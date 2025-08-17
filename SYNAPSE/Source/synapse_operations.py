from argparse import Namespace
from itertools import combinations
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F

import wandb
from Source.architecture import SparseConv2d, SparseLinear, SparseOutput
from Source.resnet18 import BasicBlock, ResNet18


class BlockSimilarityAnalyzer:
    """
    ResNetのBasicBlock間の機能的な類似度を計算するクラス。
    """
    def __init__(self, model: Any, context_detector: Any):
        self.model = model
        self.context_detector = context_detector
        # モデル内の全層の名前とインデックスの対応表
        self.layer_name_to_idx = {name: i for i, (_, name) in enumerate(model.unit_ranks)}
        print("BlockSimilarityAnalyzer is ready.")

    def _get_block_input_activations(self, block: BasicBlock) -> List[torch.Tensor]:
        """指定されたブロックへの入力となる活性化データを全タスク分集める"""
        input_activations = []
        # ブロックの最初の層の名前を取得
        first_layer_name = block.conv1.layer_name
        # その層の「前」の層のインデックスを取得
        input_layer_idx = self.layer_name_to_idx.get(first_layer_name, 1) - 1

        try:
            layer_num = int(first_layer_name.split('_')[-1])
            # block1_1のconv1 (conv_early_2) の入力は conv_early_1
            # shortcut層も考慮し、より安全に一つ前の層のインデックスを取得
            input_layer_idx = self.layer_name_to_idx.get(f"conv_early_{layer_num - 1}", -1)
            # block2_1のconv1 (conv_early_6) の入力は block1_2の最終出力 (conv_early_5)
            if "->" in first_layer_name: # shortcut層の場合
                input_layer_name = first_layer_name.split('->')[0]
                input_layer_idx = self.layer_name_to_idx.get(input_layer_name, -1)
        except (ValueError, IndexError):
            input_layer_idx = -1

        if input_layer_idx == -1: return []

        for task_id in self.context_detector.float_context_representations.keys():
            for activations_per_sample, _, _ in self.context_detector.float_context_representations[task_id]:
                for layer_idx, activation_tensor in activations_per_sample:
                    if layer_idx == input_layer_idx:
                        # ★修正点: 4Dテンソルであることを確認して追加
                        if activation_tensor.dim() == 4:
                            input_activations.append(activation_tensor)
                        # もし2Dで保存されていたら、スキップして警告を出す (デバッグ用)
                        elif activation_tensor.dim() == 2:
                            print(f"  [SYNAPSE WARNING] Skipping 2D activation tensor for layer {input_layer_idx} intended for a Conv layer.")

        return input_activations
    
    def calculate_block_similarity(self, block1: BasicBlock, block2: BasicBlock) -> float:
        """2つのBasicBlockの機能的類似度を計算する"""
        block1.eval()
        block2.eval()
        
        # 2つのブロックで共通の入力データを使う
        # ここでは簡単のため、block1への入力データをblock2にも流用する
        input_activations = self._get_block_input_activations(block1)
        if not input_activations:
            return 0.0

        output_vecs1, output_vecs2 = [], []
        with torch.no_grad():
            for input_tensor in input_activations:
                # ブロックの出力を計算
                out1, _, _ = block1(input_tensor)
                out2, _, _ = block2(input_tensor)
                
                # 出力テンソルをベクトル化
                vec1 = out1.view(out1.size(0), -1)
                vec2 = out2.view(out2.size(0), -1)
                output_vecs1.append(vec1)
                output_vecs2.append(vec2)
        
        if not output_vecs1:
            return 0.0

        # 全サンプルの出力ベクトルを結合
        vec1_full = torch.cat(output_vecs1)
        vec2_full = torch.cat(output_vecs2)

        # コサイン類似度を計算
        similarity = F.cosine_similarity(vec1_full, vec2_full, dim=1).mean()
        return similarity.item()


class StructuralOptimizationController:
    def __init__(self, model: Any, args: Namespace):
        self.model = model
        self.args = args
        print("StructuralOptimizationController is ready.")

    def prune_and_reinitialize_block(self, block_to_prune: BasicBlock):
        """指定されたブロック内の全ニューロンを剪定・再初期化する"""
        print(f"  Executing Pruning for block...")
        with torch.no_grad():
            for m in block_to_prune.modules():
                if isinstance(m, (SparseConv2d, nn.BatchNorm2d)):
                    # Kaiming Heによる初期化
                    if hasattr(m, 'weight') and m.weight is not None:
                        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if hasattr(m, 'bias') and m.bias is not None:
                        nn.init.constant_(m.bias, 0)

        # ブロック内の全ニューロンのランクをImmature(0)に戻す
        for m in block_to_prune.modules():
            if isinstance(m, SparseConv2d):
                layer_name = m.layer_name
                for i, (ranks, name) in enumerate(self.model.unit_ranks):
                    if name == layer_name:
                        self.model.unit_ranks[i] = (np.zeros_like(ranks), name)
                        print(f"    >> Neurons in {name} have been reset to Immature.")
                        break

def run_synapse_optimization(model: Any, context_detector: Any, args: Namespace, episode_index: int):
    if episode_index < args.synapse_activation_task_count:
        return
    log_metrics = {}
    pruned_block_count = 0
    shared_block_count = 0 # 将来の共有化機能のために用意   
    # ResNet以外は現在のロジックのまま（何もしない）
    if not isinstance(model, ResNet18):
        print("SYNAPSE: Block-level optimization is only implemented for ResNet18.")
        return

    print(f"\n--- SYNAPSE Block-level Phase starting after episode {episode_index} ---")
    analyzer = BlockSimilarityAnalyzer(model, context_detector)
    controller = StructuralOptimizationController(model, args)

    # 凍結済み（成熟）ブロックをタスクごとに分類
    mature_blocks_by_task: Dict[int, List[BasicBlock]] = {}
    for block in model.blocks:
        # ブロックが成熟しているかを判定（ここではブロック内の最初のconv層のニューロンが1つでも成熟していたら、と仮定）
        first_layer_name = block.conv1.layer_name
        is_mature = False
        task_id = -1
        for ranks, name in model.unit_ranks:
            if name == first_layer_name:
                if np.any(ranks > 1):
                    is_mature = True
                    # 多数決でブロックのタスクを決定
                    mature_indices = np.where(ranks > 1)[0]
                    tasks = [model.neuron_birth_task[name].get(idx) for idx in mature_indices]
                    if tasks:
                        task_id = max(set(tasks), key=tasks.count) # 最頻値
                break
        
        if is_mature and task_id != -1:
            if task_id not in mature_blocks_by_task:
                mature_blocks_by_task[task_id] = []
            mature_blocks_by_task[task_id].append(block)

    # Intra-Task Pruning (同一タスク内のブロック剪定)
    pruned_blocks = []
    for task, blocks in mature_blocks_by_task.items():
        if len(blocks) < 2: continue
        for block1, block2 in combinations(blocks, 2):
            similarity = analyzer.calculate_block_similarity(block1, block2)
            if similarity > args.threshold_intra_task_pruning:
                print(f"  >> PRUNING CANDIDATE (Intra-Task): Blocks in task {task}. Similarity: {similarity:.4f}")
                # どちらかを剪定（ここでは単純に2番目のブロック）
                controller.prune_and_reinitialize_block(block2)
                pruned_blocks.append(block2)
                pruned_block_count += 1
                break # 1ペア見つけたら次のタスクへ

    if wandb.run is not None:
        log_metrics['synapse/pruned_blocks'] = pruned_block_count
        log_metrics['synapse/shared_blocks'] = shared_block_count # 現状は常に0
        wandb.log(log_metrics, step=episode_index)
        
    print("\n--- SYNAPSE Phase finished. ---\n")