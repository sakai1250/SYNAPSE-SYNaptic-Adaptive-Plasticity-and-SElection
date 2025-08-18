# from argparse import Namespace
# from itertools import combinations
# from typing import Any, Dict, List

# import numpy as np
# import torch
# import torch.nn.functional as F

# import wandb
# from Source.architecture import SparseConv2d, SparseLinear, SparseOutput
# from Source.resnet18 import BasicBlock, ResNet18


# class BlockSimilarityAnalyzer:
#     """
#     ResNetのBasicBlock間の機能的な類似度を計算するクラス。
#     """
#     def __init__(self, model: Any, context_detector: Any):
#         self.model = model
#         self.context_detector = context_detector
#         # モデル内の全層の名前とインデックスの対応表
#         self.layer_name_to_idx = {name: i for i, (_, name) in enumerate(model.unit_ranks)}
#         print("BlockSimilarityAnalyzer is ready.")

#     def _get_block_input_activations(self, block: BasicBlock) -> List[torch.Tensor]:
#         """指定されたブロックへの入力となる活性化データを全タスク分集める"""
#         input_activations = []
#         # ブロックの最初の層の名前を取得
#         first_layer_name = block.conv1.layer_name
#         # その層の「前」の層のインデックスを取得
#         input_layer_idx = self.layer_name_to_idx.get(first_layer_name, 1) - 1

#         try:
#             layer_num = int(first_layer_name.split('_')[-1])
#             # block1_1のconv1 (conv_early_2) の入力は conv_early_1
#             # shortcut層も考慮し、より安全に一つ前の層のインデックスを取得
#             input_layer_idx = self.layer_name_to_idx.get(f"conv_early_{layer_num - 1}", -1)
#             # block2_1のconv1 (conv_early_6) の入力は block1_2の最終出力 (conv_early_5)
#             if "->" in first_layer_name: # shortcut層の場合
#                 input_layer_name = first_layer_name.split('->')[0]
#                 input_layer_idx = self.layer_name_to_idx.get(input_layer_name, -1)
#         except (ValueError, IndexError):
#             input_layer_idx = -1

#         if input_layer_idx == -1: return []

#         for task_id in self.context_detector.float_context_representations.keys():
#             for activations_per_sample, _, _ in self.context_detector.float_context_representations[task_id]:
#                 for layer_idx, activation_tensor in activations_per_sample:
#                     if layer_idx == input_layer_idx:
#                         if activation_tensor.dim() == 4:
#                             input_activations.append(activation_tensor)
#                         # もし2Dで保存されていたら、スキップして警告を出す (デバッグ用)
#                         elif activation_tensor.dim() == 2:
#                             print(f"  [SYNAPSE WARNING] Skipping 2D activation tensor for layer {input_layer_idx} intended for a Conv layer.")

#         return input_activations
    
#     def calculate_block_similarity(self, block1: BasicBlock, block2: BasicBlock) -> float:
#         """2つのBasicBlockの機能的類似度を計算する"""
#         block1.eval()
#         block2.eval()
        
#         # 2つのブロックで共通の入力データを使う
#         # ここでは簡単のため、block1への入力データをblock2にも流用する
#         input_activations = self._get_block_input_activations(block1)
#         if not input_activations:
#             return 0.0

#         output_vecs1, output_vecs2 = [], []
#         with torch.no_grad():
#             for input_tensor in input_activations:
#                 # ブロックの出力を計算
#                 out1, _, _ = block1(input_tensor)
#                 out2, _, _ = block2(input_tensor)
                
#                 # 出力テンソルをベクトル化
#                 vec1 = out1.view(out1.size(0), -1)
#                 vec2 = out2.view(out2.size(0), -1)
#                 output_vecs1.append(vec1)
#                 output_vecs2.append(vec2)
        
#         if not output_vecs1:
#             return 0.0

#         # 全サンプルの出力ベクトルを結合
#         vec1_full = torch.cat(output_vecs1)
#         vec2_full = torch.cat(output_vecs2)

#         # コサイン類似度を計算
#         similarity = F.cosine_similarity(vec1_full, vec2_full, dim=1).mean()
#         return similarity.item()


# class StructuralOptimizationController:
#     def __init__(self, model: Any, args: Namespace):
#         self.model = model
#         self.args = args
#         print("StructuralOptimizationController is ready.")

#     def prune_and_reinitialize_block(self, block_to_prune: BasicBlock):
#         """指定されたブロック内の全ニューロンを剪定・再初期化する"""
#         print(f"  Executing Pruning for block...")
#         with torch.no_grad():
#             for m in block_to_prune.modules():
#                 if isinstance(m, (SparseConv2d, nn.BatchNorm2d)):
#                     # Kaiming Heによる初期化
#                     if hasattr(m, 'weight') and m.weight is not None:
#                         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                     if hasattr(m, 'bias') and m.bias is not None:
#                         nn.init.constant_(m.bias, 0)

#         # ブロック内の全ニューロンのランクをImmature(0)に戻す
#         for m in block_to_prune.modules():
#             if isinstance(m, SparseConv2d):
#                 layer_name = m.layer_name
#                 for i, (ranks, name) in enumerate(self.model.unit_ranks):
#                     if name == layer_name:
#                         self.model.unit_ranks[i] = (np.zeros_like(ranks), name)
#                         print(f"    >> Neurons in {name} have been reset to Immature.")
#                         break

# def run_synapse_optimization(model: Any, context_detector: Any, args: Namespace, episode_index: int):
#     if episode_index < args.synapse_activation_task_count:
#         return
#     log_metrics = {}
#     pruned_block_count = 0
#     shared_block_count = 0 # 将来の共有化機能のために用意   
#     # ResNet以外は現在のロジックのまま（何もしない）
#     if not isinstance(model, ResNet18):
#         print("SYNAPSE: Block-level optimization is only implemented for ResNet18.")
#         return

#     print(f"\n--- SYNAPSE Block-level Phase starting after episode {episode_index} ---")
#     analyzer = BlockSimilarityAnalyzer(model, context_detector)
#     controller = StructuralOptimizationController(model, args)

#     # 凍結済み（成熟）ブロックをタスクごとに分類
#     mature_blocks_by_task: Dict[int, List[BasicBlock]] = {}
#     for block in model.blocks:
#         # ブロックが成熟しているかを判定（ここではブロック内の最初のconv層のニューロンが1つでも成熟していたら、と仮定）
#         first_layer_name = block.conv1.layer_name
#         is_mature = False
#         task_id = -1
#         for ranks, name in model.unit_ranks:
#             if name == first_layer_name:
#                 if np.any(ranks > 1):
#                     is_mature = True
#                     # 多数決でブロックのタスクを決定
#                     mature_indices = np.where(ranks > 1)[0]
#                     tasks = [model.neuron_birth_task[name].get(idx) for idx in mature_indices]
#                     if tasks:
#                         task_id = max(set(tasks), key=tasks.count) # 最頻値
#                 break
        
#         if is_mature and task_id != -1:
#             if task_id not in mature_blocks_by_task:
#                 mature_blocks_by_task[task_id] = []
#             mature_blocks_by_task[task_id].append(block)

#     # Intra-Task Pruning (同一タスク内のブロック剪定)
#     pruned_blocks = []
#     for task, blocks in mature_blocks_by_task.items():
#         if len(blocks) < 2: continue
#         for block1, block2 in combinations(blocks, 2):
#             similarity = analyzer.calculate_block_similarity(block1, block2)
#             if similarity > args.threshold_intra_task_pruning:
#                 print(f"  >> PRUNING CANDIDATE (Intra-Task): Blocks in task {task}. Similarity: {similarity:.4f}")
#                 # どちらかを剪定（ここでは単純に2番目のブロック）
#                 controller.prune_and_reinitialize_block(block2)
#                 pruned_blocks.append(block2)
#                 pruned_block_count += 1
#                 break # 1ペア見つけたら次のタスクへ

#     if wandb.run is not None:
#         log_metrics['synapse/pruned_blocks'] = pruned_block_count
#         log_metrics['synapse/shared_blocks'] = shared_block_count # 現状は常に0
#         wandb.log(log_metrics, step=episode_index)
        
#     print("\n--- SYNAPSE Phase finished. ---\n")
    

###########################


# # SYNAPSE/Source/synapse_operations.py

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from argparse import Namespace
# from typing import Any, Dict, List
# from itertools import combinations
# import numpy as np
# import wandb
# from Source.architecture import SparseConv2d, SparseLinear, SparseOutput
# from Source.resnet18 import ResNet18, BasicBlock
# from avalanche.benchmarks import TCLExperience
# from Source.context_detector import get_n_samples_per_class
# from Source.helper import get_device


# class BlockSimilarityAnalyzer:
#     """
#     ResNetのBasicBlock間の機能的な類似度を計算するクラス。
#     """
#     def __init__(self, model: Any, context_detector: Any, episode_data: TCLExperience, args: Namespace):
#         self.model = model
#         self.context_detector = context_detector
#         self.episode_data = episode_data
#         self.args = args
#         self.layer_name_to_idx = {name: i for i, (_, name) in enumerate(model.unit_ranks)}
#         print("BlockSimilarityAnalyzer is ready.")

#     def calculate_block_similarity(self, block1: BasicBlock, block2: BasicBlock) -> float:
#         """2つのBasicBlockの機能的類似度を計算する"""
#         block1.eval()
#         block2.eval()
        
#         # 代表サンプルを取得
#         samples_per_class = get_n_samples_per_class(self.episode_data, self.args.memo_per_class_context)
        
#         output_vecs1, output_vecs2 = [], []
#         with torch.no_grad():
#             for samples, _ in samples_per_class:
#                 samples = samples.to(get_device())
                
#                 # --- モデルを経由して、ブロックへの正しい形の入力を生成 ---
#                 x = samples
#                 x = self.model.relu(self.model.bn1(self.model.conv1(x)))
                
#                 for block in self.model.blocks:
#                     if block == block1 or block == block2:
#                         break
#                     x, _, _ = block(x)

#                 # --- 2つのブロックの出力を比較 ---
#                 out1, _, _ = block1(x)
#                 out2, _, _ = block2(x)
                
#                 vec1 = out1.view(out1.size(0), -1)
#                 vec2 = out2.view(out2.size(0), -1)
#                 output_vecs1.append(vec1)
#                 output_vecs2.append(vec2)
        
#         if not output_vecs1: 
#             self.model.train()
#             return 0.0

#         vec1_full = torch.cat(output_vecs1)
#         vec2_full = torch.cat(output_vecs2)

#         similarity = F.cosine_similarity(vec1_full, vec2_full, dim=1).mean()
#         self.model.train()
#         return similarity.item()


# class StructuralOptimizationController:
#     def __init__(self, model: Any, args: Namespace):
#         self.model = model
#         self.args = args
#         print("StructuralOptimizationController is ready.")

#     def prune_and_reinitialize_block(self, block_to_prune: BasicBlock):
#         """指定されたブロック内の全ニューロンを剪定・再初期化する"""
#         print(f"  Executing Pruning for block...")
#         with torch.no_grad():
#             for m in block_to_prune.modules():
#                 if isinstance(m, (SparseConv2d, nn.BatchNorm2d)):
#                     if hasattr(m, 'weight') and m.weight is not None:
#                         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                     if hasattr(m, 'bias') and m.bias is not None:
#                         nn.init.constant_(m.bias, 0)

#         for m in block_to_prune.modules():
#             if isinstance(m, SparseConv2d):
#                 layer_name = m.layer_name
#                 for i, (ranks, name) in enumerate(self.model.unit_ranks):
#                     if name == layer_name:
#                         self.model.unit_ranks[i] = (np.zeros_like(ranks), name)
#                         print(f"    >> Neurons in {name} have been reset to Immature.")
#                         break
    
#     # =================================================================
#     # Inter-Task Sharing を行う
#     # =================================================================
#     def share_blocks(self, source_block: BasicBlock, target_block: BasicBlock):
#         """source_block の知識を target_block に統合し、source_block を再初期化する"""
#         print(f"  Executing Sharing: Merging knowledge from one block to another...")
#         with torch.no_grad():
#             # sourceの重みをtargetに加算していく
#             for (name_s, param_s), (name_t, param_t) in zip(source_block.named_parameters(), target_block.named_parameters()):
#                 if param_s.data.shape == param_t.data.shape:
#                     param_t.data += param_s.data
        
#         # 知識の転移が完了したので、sourceブロックは再初期化してプールに戻す
#         self.prune_and_reinitialize_block(source_block)
#     # =================================================================


# def run_synapse_optimization(model: Any, context_detector: Any, args: Namespace, episode_index: int, train_episode: TCLExperience) -> dict:
#     if episode_index < args.synapse_activation_task_count:
#         return {}
    
#     if not isinstance(model, ResNet18):
#         print("SYNAPSE: Block-level optimization is currently only implemented for ResNet18.")
#         return {}

#     print(f"\n--- SYNAPSE Block-level Phase starting after episode {episode_index} ---")
#     analyzer = BlockSimilarityAnalyzer(model, context_detector, train_episode, args)
#     controller = StructuralOptimizationController(model, args)

#     # =================================================================
#     # ★変更点: 比較可能なブロックのグループを定義
#     # =================================================================
#     compatible_block_groups = [
#         [model.block1_1, model.block1_2],
#         [model.block2_2], # チャンネル数が変わるブロックは単独
#         [model.block3_2],
#         [model.block4_2]
#     ]
#     # =================================================================


#     synapse_metrics = {
#         'synapse/pruned_blocks': 0,
#         'synapse/shared_blocks': 0 
#     }

#     mature_blocks_by_task: Dict[int, List[BasicBlock]] = {}
#     for block in model.blocks:
#         first_layer_name = block.conv1.layer_name
#         is_mature, task_id = False, -1
#         for ranks, name in model.unit_ranks:
#             if name == first_layer_name:
#                 if np.any(ranks > 1):
#                     is_mature = True
#                     mature_indices = np.where(ranks > 1)[0]
#                     tasks = [model.neuron_birth_task[name].get(idx) for idx in mature_indices if model.neuron_birth_task[name].get(idx) is not None]
#                     if tasks:
#                         task_id = max(set(tasks), key=tasks.count)
#                 break
        
#         if is_mature and task_id != -1:
#             if task_id not in mature_blocks_by_task:
#                 mature_blocks_by_task[task_id] = []
#             mature_blocks_by_task[task_id].append(block)

#     # # --- 1. Intra-Task Pruning (同一タスク内の剪定) ---
#     # pruned_blocks = []
#     # for task, blocks in mature_blocks_by_task.items():
#     #     if len(blocks) < 2: continue
#     #     for block1, block2 in combinations(blocks, 2):
#     #         if any(b in pruned_blocks for b in [block1, block2]): continue
#     #         similarity = analyzer.calculate_block_similarity(block1, block2)
#     #         if similarity > args.threshold_intra_task_pruning:
#     #             print(f"  >> PRUNING TRIGGERED (Intra-Task): Blocks in task {task}. Similarity: {similarity:.4f}")
#     #             controller.prune_and_reinitialize_block(block2)
#     #             pruned_blocks.append(block2)
#     #             synapse_metrics['synapse/pruned_blocks'] += 1
#     #             break 

#     # --- Intra-Task Pruning ---
#     pruned_blocks = []
#     # ★変更点: グループ内で比較
#     for group in compatible_block_groups:
#         for task, blocks_in_task in mature_blocks_by_task.items():
#             # 現在のグループに属する、このタスクのブロックを抽出
#             relevant_blocks = [b for b in blocks_in_task if b in group]
#             if len(relevant_blocks) < 2: continue
            
#             for block1, block2 in combinations(relevant_blocks, 2):
#                 if any(b in pruned_blocks for b in [block1, block2]): continue
#                 similarity = analyzer.calculate_block_similarity(block1, block2)
#                 if similarity > args.threshold_intra_task_pruning:
#                     print(f"  >> PRUNING TRIGGERED (Intra-Task): Blocks in task {task}. Similarity: {similarity:.4f}")
#                     controller.prune_and_reinitialize_block(block2)
#                     pruned_blocks.append(block2)
#                     synapse_metrics['synapse/pruned_blocks'] += 1
#                     break


#     # =================================================================
#     # Inter-Task Sharing (異種タスク間の共有化)
#     # =================================================================
#     if len(mature_blocks_by_task) > 1:
#         print("  Analyzing interactions between tasks for Inter-Task Sharing...")
#         shared_blocks = list(pruned_blocks) # 剪定済みブロックは共有化の対象外
#         # ... (既存のロジックを、同様にグループ内で比較するように修正) ...
#         for group in compatible_block_groups:
#             # 異なるタスクのブロックペアを全て評価
#             for (task1, blocks1), (task2, blocks2) in combinations(mature_blocks_by_task.items(), 2):
#                 # 各タスクから、現在のグループに属するブロックを抽出
#                 relevant_blocks1 = [b for b in blocks1 if b in group and b not in shared_blocks]
#                 relevant_blocks2 = [b for b in blocks2 if b in group and b not in shared_blocks]

#                 for b1 in relevant_blocks1:
#                     for b2 in relevant_blocks2:
#                         if b1 in shared_blocks or b2 in shared_blocks: continue

#                         similarity = analyzer.calculate_block_similarity(b1, b2)
#                         if similarity > args.threshold_inter_task_sharing:
#                             print(f"  >> SHARING TRIGGERED (Inter-Task): Blocks from tasks ({task1}, {task2}). Similarity: {similarity:.4f}")
                            
#                             # より新しいタスクのブロックをsource、古い方をtargetとする
#                             if task1 > task2:
#                                 source_block, target_block = b1, b2
#                             else:
#                                 source_block, target_block = b2, b1
                            
#                             controller.share_blocks(source_block, target_block)
#                             shared_blocks.extend([source_block, target_block])
#                             synapse_metrics['synapse/shared_blocks'] += 1
#                             break # このペアの処理は終了
#                     if b1 in shared_blocks:
#                         break
                            
#         # # 異なるタスクのブロックペアを全て評価
#         # for (task1, blocks1), (task2, blocks2) in combinations(mature_blocks_by_task.items(), 2):
#         #     for b1 in blocks1:
#         #         if b1 in shared_blocks: continue
#         #         for b2 in blocks2:
#         #             if b2 in shared_blocks: continue

#         #             similarity = analyzer.calculate_block_similarity(b1, b2)
#         #             if similarity > args.threshold_inter_task_sharing:
#         #                 print(f"  >> SHARING TRIGGERED (Inter-Task): Blocks from tasks ({task1}, {task2}). Similarity: {similarity:.4f}")
                        
#         #                 # より新しいタスクのブロックをsource、古い方をtargetとする
#         #                 if task1 > task2:
#         #                     source_block, target_block = b1, b2
#         #                 else:
#         #                     source_block, target_block = b2, b1
                        
#         #                 controller.share_blocks(source_block, target_block)
#         #                 shared_blocks.extend([source_block, target_block])
#         #                 synapse_metrics['synapse/shared_blocks'] += 1
#         #                 break # このペアの処理は終了
#         #         if b1 in shared_blocks:
#         #             break
#     # =================================================================

#     print("\n--- SYNAPSE Phase finished. ---\n")
#     return synapse_metrics


# SYNAPSE/Source/synapse_operations.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from argparse import Namespace
from typing import Any, Dict, List
from itertools import combinations
import numpy as np
import wandb
from Source.architecture import SparseConv2d, SparseLinear, SparseOutput
from Source.resnet18 import ResNet18, BasicBlock
from avalanche.benchmarks import TCLExperience
from Source.context_detector import get_n_samples_per_class
from Source.helper import get_device


class BlockSimilarityAnalyzer:
    """
    ResNetのBasicBlock間の機能的な類似度を計算するクラス。
    """
    def __init__(self, model: Any, context_detector: Any, episode_data: TCLExperience, args: Namespace):
        self.model = model
        self.context_detector = context_detector
        self.episode_data = episode_data
        self.args = args
        self.layer_name_to_idx = {name: i for i, (_, name) in enumerate(model.unit_ranks)}
        print("BlockSimilarityAnalyzer is ready.")

    def calculate_block_similarity(self, block1: BasicBlock, block2: BasicBlock) -> float:
        """2つのBasicBlockの機能的類似度を計算する"""
        block1.eval()
        block2.eval()
        
        # 代表サンプルを取得
        samples_per_class = get_n_samples_per_class(self.episode_data, self.args.memo_per_class_context)
        
        output_vecs1, output_vecs2 = [], []
        with torch.no_grad():
            for samples, _ in samples_per_class:
                samples = samples.to(get_device())
                
                # --- モデルを経由して、ブロックへの正しい形の入力を生成 ---
                x = samples
                x = self.model.relu(self.model.bn1(self.model.conv1(x)))
                
                # 比較対象のブロックの直前までデータを流す
                for block in self.model.blocks:
                    if block == block1 or block == block2:
                        break
                    x, _, _ = block(x)

                # --- 2つのブロックの出力を比較 ---
                out1, _, _ = block1(x)
                out2, _, _ = block2(x)
                
                vec1 = out1.view(out1.size(0), -1)
                vec2 = out2.view(out2.size(0), -1)
                output_vecs1.append(vec1)
                output_vecs2.append(vec2)
        
        if not output_vecs1: 
            self.model.train()
            return 0.0

        vec1_full = torch.cat(output_vecs1)
        vec2_full = torch.cat(output_vecs2)

        similarity = F.cosine_similarity(vec1_full, vec2_full, dim=1).mean()
        self.model.train()
        return similarity.item()


class StructuralOptimizationController:
    def __init__(self, model: Any, args: Namespace):
        self.model = model
        self.args = args
        print("StructuralOptimizationController is ready.")

    def prune_and_reinitialize_block(self, block_to_prune: BasicBlock):
        """指定されたブロック内の全ニューロンを剪定・再初期化する"""
        print(f"  Executing Pruning for a block...")
        with torch.no_grad():
            for m in block_to_prune.modules():
                if isinstance(m, (SparseConv2d, nn.BatchNorm2d)):
                    if hasattr(m, 'weight') and m.weight is not None:
                        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if hasattr(m, 'bias') and m.bias is not None:
                        nn.init.constant_(m.bias, 0)

        for m in block_to_prune.modules():
            if isinstance(m, SparseConv2d):
                layer_name = m.layer_name
                for i, (ranks, name) in enumerate(self.model.unit_ranks):
                    if name == layer_name:
                        self.model.unit_ranks[i] = (np.zeros_like(ranks), name)
                        print(f"    >> Neurons in {name} have been reset to Immature.")
                        break
    
    def share_blocks(self, source_block: BasicBlock, target_block: BasicBlock):
        """source_block の知識を target_block に統合し、source_block を再初期化する"""
        print(f"  Executing Sharing: Merging knowledge from one block to another...")
        with torch.no_grad():
            # sourceの重みをtargetに加算していく (簡単な平均化)
            for (name_s, param_s), (name_t, param_t) in zip(source_block.named_parameters(), target_block.named_parameters()):
                if param_s.data.shape == param_t.data.shape:
                    param_t.data = (param_t.data + param_s.data) / 2.0
        
        # 知識の転移が完了したので、sourceブロックは再初期化してプールに戻す
        self.prune_and_reinitialize_block(source_block)


def run_synapse_optimization(model: Any, context_detector: Any, args: Namespace, episode_index: int, train_episode: TCLExperience) -> dict:
    if episode_index < args.synapse_activation_task_count:
        return {}
    
    if not isinstance(model, ResNet18):
        print("SYNAPSE: Block-level optimization is currently only implemented for ResNet18.")
        return {}

    print(f"\n--- SYNAPSE Block-level Phase starting after episode {episode_index} ---")
    analyzer = BlockSimilarityAnalyzer(model, context_detector, train_episode, args)
    controller = StructuralOptimizationController(model, args)

    # 互換性のあるブロックのグループを定義
    # ResNet18の構造に基づき、同じ入出力形状を持つブロックをグループ化
    compatible_block_groups = [
        [model.block1_1, model.block1_2],
        [model.block2_2], 
        [model.block3_2],
        [model.block4_2]
    ]

    synapse_metrics = {
        'synapse/pruned_blocks': 0,
        'synapse/shared_blocks': 0 
    }

    # 成熟したブロックをタスクごとに分類
    mature_blocks_by_task: Dict[int, List[BasicBlock]] = {}
    for block in model.blocks:
        first_layer_name = block.conv1.layer_name
        is_mature, task_id = False, -1
        for ranks, name in model.unit_ranks:
            if name == first_layer_name:
                if np.any(ranks > 1): # ランクが1より大きい=成熟
                    is_mature = True
                    mature_indices = np.where(ranks > 1)[0]
                    tasks = [model.neuron_birth_task[name].get(idx) for idx in mature_indices if model.neuron_birth_task[name].get(idx) is not None]
                    if tasks:
                        task_id = max(set(tasks), key=tasks.count) # 最も多いタスクIDを所属タスクとする
                break
        
        if is_mature and task_id != -1:
            if task_id not in mature_blocks_by_task:
                mature_blocks_by_task[task_id] = []
            mature_blocks_by_task[task_id].append(block)

    # --- Intra-Task Pruning (同一タスク内の剪定) ---
    print("  Analyzing interactions within tasks for Intra-Task Pruning...")
    pruned_blocks = []
    for group in compatible_block_groups:
        for task, blocks_in_task in mature_blocks_by_task.items():
            # 現在のグループに属し、かつこのタスクに属するブロックを抽出
            relevant_blocks = [b for b in blocks_in_task if b in group]
            if len(relevant_blocks) < 2: continue
            
            for block1, block2 in combinations(relevant_blocks, 2):
                if any(b in pruned_blocks for b in [block1, block2]): continue
                similarity = analyzer.calculate_block_similarity(block1, block2)
                if similarity > args.threshold_intra_task_pruning:
                    print(f"  >> PRUNING TRIGGERED (Intra-Task): Blocks in task {task}. Similarity: {similarity:.4f}")
                    # より新しいタスクのブロックを剪定（ここでは単純にblock2）
                    controller.prune_and_reinitialize_block(block2)
                    pruned_blocks.append(block2)
                    synapse_metrics['synapse/pruned_blocks'] += 1
                    break

    # --- Inter-Task Sharing (異種タスク間の共有化) ---
    if len(mature_blocks_by_task) > 1:
        print("  Analyzing interactions between tasks for Inter-Task Sharing...")
        shared_blocks = list(pruned_blocks) # 剪定済みブロックは共有化の対象外
        
        for group in compatible_block_groups:
            # 異なるタスクのブロックペアを全て評価
            for (task1, blocks1), (task2, blocks2) in combinations(mature_blocks_by_task.items(), 2):
                # 各タスクから、現在のグループに属するブロックを抽出
                relevant_blocks1 = [b for b in blocks1 if b in group and b not in shared_blocks]
                relevant_blocks2 = [b for b in blocks2 if b in group and b not in shared_blocks]

                for b1 in relevant_blocks1:
                    for b2 in relevant_blocks2:
                        if b1 in shared_blocks or b2 in shared_blocks: continue

                        similarity = analyzer.calculate_block_similarity(b1, b2)
                        if similarity > args.threshold_inter_task_sharing:
                            print(f"  >> SHARING TRIGGERED (Inter-Task): Blocks from tasks ({task1}, {task2}). Similarity: {similarity:.4f}")
                            
                            # より新しいタスクのブロックをsource、古い方をtargetとする
                            if task1 > task2:
                                source_block, target_block = b1, b2
                            else:
                                source_block, target_block = b2, b1
                            
                            controller.share_blocks(source_block, target_block)
                            shared_blocks.extend([source_block, target_block])
                            synapse_metrics['synapse/shared_blocks'] += 1
                            break
                    if b1 in shared_blocks:
                        break
                            
    print("\n--- SYNAPSE Phase finished. ---\n")
    return synapse_metrics