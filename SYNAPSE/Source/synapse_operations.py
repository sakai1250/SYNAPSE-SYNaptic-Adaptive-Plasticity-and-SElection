# NICE/Source/synapse_operations.py
import torch
from argparse import Namespace
from typing import Any
from itertools import combinations

class NeuronSimilarityAnalyzer:
    """
    ニューロン間の機能的な類似度を計算するクラス。
    """
    def __init__(self, model: Any, context_detector: Any):
        """
        Args:
            model (Any): 分析対象のネットワークモデル。
            context_detector (Any): NICEのContextDetector。代表サンプルの活性化パターンを保持。
        """
        self.model = model
        self.context_detector = context_detector
        print("NeuronSimilarityAnalyzer is ready.")

    def calculate_cosine_similarity(self, layer_name: str, neuron_idx1: int, neuron_idx2: int) -> float:
        """
        指定された層内の2つのニューロンの機能的類似度（コサイン類似度）を計算します。
        全タスクの代表サンプルに対する活性化ベクトルを集約して比較します。
        """
        vec1_list, vec2_list = [], []

        # ContextDetectorに保存されている全タスクの代表サンプル活性化データをループ
        for task_id in self.context_detector.float_context_representations.keys():
            for activations_per_sample, _, _ in self.context_detector.float_context_representations[task_id]:
                # layer_name に対応する活性化記録を探す
                target_activation = None
                for layer_idx, activation_tensor in activations_per_sample:
                    # モデルの持つレイヤー名リストからインデックスを取得して照合
                    # (ResNet18を想定した簡易的な照合)
                    if self.model.layers[layer_idx][1] == layer_name:
                         target_activation = activation_tensor
                         break
                
                if target_activation is None:
                    continue

                # 活性化ベクトルの形状をチェック
                # [batch, channels, height, width] の場合 (Conv層)
                if len(target_activation.shape) == 4:
                    # 空間方向は平均をとってベクトル化
                    act1 = target_activation[:, neuron_idx1, :, :].mean(dim=[1, 2])
                    act2 = target_activation[:, neuron_idx2, :, :].mean(dim=[1, 2])
                # [batch, features] の場合 (Linear層)
                else:
                    act1 = target_activation[:, neuron_idx1]
                    act2 = target_activation[:, neuron_idx2]
                
                vec1_list.append(act1)
                vec2_list.append(act2)

        if not vec1_list or not vec2_list:
            return 0.0

        # 全サンプルの活性化ベクトルを結合して一つの長いベクトルにする
        vec1 = torch.cat(vec1_list)
        vec2 = torch.cat(vec2_list)

        # コサイン類似度を計算して返す
        similarity = F.cosine_similarity(vec1, vec2, dim=0)
        return similarity.item()


class StructuralOptimizationController:
    """
    ニューロンの剪定（Pruning）や共有化（Sharing）といった構造最適化を実行するクラス。
    """
    def __init__(self, model: Any, args: Namespace):
        """
        Args:
            model (Any): 対象のネットワークモデル。
            args (Namespace): 閾値などのハイパーパラメータ。
        """
        self.model = model
        self.args = args
        # すぐに層の重みなどを名前で引けるように、モジュールの辞書を作っておくと便利
        self.modules_dict = {name: module for name, module in model.named_modules()}
        print("StructuralOptimizationController is ready.")
            
    def _clear_neuron_connections(self, layer_name: str, neuron_idx: int):
        """
        指定されたニューロンの入出力接続をすべてゼロにする（内部ヘルパー関数）。
        """
        # --- 出力接続をゼロに ---
        target_module = self.modules_dict.get(layer_name)
        if target_module and hasattr(target_module, 'weight'):
            with torch.no_grad():
                target_module.weight.data[neuron_idx].zero_()
                if target_module.bias is not None:
                    target_module.bias.data[neuron_idx].zero_()

        # --- 入力接続をゼロに ---
        found_next_layer = False
        module_list = list(self.model.modules())
        for i, module in enumerate(module_list):
            if module == target_module and i + 1 < len(module_list):
                for next_module in module_list[i+1:]:
                    if isinstance(next_module, (SparseConv2d, SparseLinear, SparseOutput)):
                        with torch.no_grad():
                            if len(next_module.weight.data.shape) == 4: # Conv層
                                next_module.weight.data[:, neuron_idx, :, :].zero_()
                            else: # Linear層
                                next_module.weight.data[:, neuron_idx].zero_()
                        found_next_layer = True
                        break
            if found_next_layer:
                break

    def prune_and_reinitialize(self, layer_name: str, neuron_idx: int):
        """
        指定されたニューロンを剪定し、戦略的に再初期化して immature プールに戻します。
        """
        print(f"  Executing Pruning for neuron {neuron_idx} in layer {layer_name}...")
        
        # --- 1. 接続をクリア ---
        self._clear_neuron_connections(layer_name, neuron_idx)

        # --- 2. 戦略的に再初期化 ---
        target_module = self.modules_dict.get(layer_name)
        if target_module and hasattr(target_module, 'weight'):
             with torch.no_grad():
                weight_data = target_module.weight.data
                other_neurons_mask = torch.ones(weight_data.size(0), dtype=torch.bool, device=weight_data.device)
                other_neurons_mask[neuron_idx] = False
                avg_weight = weight_data[other_neurons_mask].mean(dim=0)
                
                target_module.weight.data[neuron_idx] = -avg_weight
        
        # --- 3. 状態を「未熟 (Immature)」に戻す ---
        for i, (ranks, name) in enumerate(self.model.unit_ranks):
            if name == layer_name:
                self.model.unit_ranks[i][0][neuron_idx] = 0
                print(f"  >> Neuron {neuron_idx} in {layer_name} has been reset to Immature.")
                break

    def share_neurons(self, layer_name: str, source_neuron_idx: int, target_neuron_idx: int):
        """
        source_neuron の接続を target_neuron に統合し、source_neuron を削除・再初期化します。
        """
        print(f"  Executing Sharing: Merging neuron {source_neuron_idx} into {target_neuron_idx} in layer {layer_name}...")
        target_module = self.modules_dict.get(layer_name)
        if not (target_module and hasattr(target_module, 'weight')):
            return

        # --- 1. 出力接続を統合 ---
        # source の重みを target に加算する
        with torch.no_grad():
            target_module.weight.data[target_neuron_idx] += target_module.weight.data[source_neuron_idx]
            if target_module.bias is not None:
                target_module.bias.data[target_neuron_idx] += target_module.bias.data[source_neuron_idx]

        # --- 2. 入力接続を統合 ---
        found_next_layer = False
        module_list = list(self.model.modules())
        for i, module in enumerate(module_list):
            if module == target_module and i + 1 < len(module_list):
                for next_module in module_list[i+1:]:
                    if isinstance(next_module, (SparseConv2d, SparseLinear, SparseOutput)):
                        with torch.no_grad():
                            # 次の層の重み行列で、source_neuron の列を target_neuron の列に加算
                            if len(next_module.weight.data.shape) == 4: # Conv層
                                next_module.weight.data[:, target_neuron_idx, :, :] += next_module.weight.data[:, source_neuron_idx, :, :]
                            else: # Linear層
                                next_module.weight.data[:, target_neuron_idx] += next_module.weight.data[:, source_neuron_idx]
                        found_next_layer = True
                        break
            if found_next_layer:
                break
        
        # --- 3. sourceニューロンを剪定・再初期化 ---
        # 接続の統合が完了したので、sourceは不要になる
        self.prune_and_reinitialize(layer_name, source_neuron_idx)

def run_synapse_optimization(model: Any, context_detector: Any, args: Namespace, episode_index: int):
    """
    NICEのタスク学習完了後に呼び出されるSYNAPSEのメイン関数。
    
    Args:
        model (Any): 対象のネットワークモデル。
        context_detector (Any): NICEのContextDetector。
        args (Namespace): ハイパーパラメータ。
        episode_index (int): 現在のタスク（エピソード）番号。
    """
    print(f"\n--- SYNAPSE Phase starting after episode {episode_index} ---")
    
    # --- Step 0: 実行条件のチェック ---
    if episode_index < args.synapse_activation_task_count:
        print(f"SYNAPSE is not activated yet. (Current task: {episode_index}, Activation threshold: {args.synapse_activation_task_count})")
        return

    # --- Step 1: 初期化 ---
    analyzer = NeuronSimilarityAnalyzer(model, context_detector)
    controller = StructuralOptimizationController(model, args)
    
    # --- Step 2: 各層で最適化を実行 ---
    print("Starting layer-wise optimization...")
    for layer_idx, (ranks, layer_name) in enumerate(model.unit_ranks):
        if layer_idx == 0: continue # 入力層はスキップ

        print(f"\n[Layer: {layer_name}]")

        # --- Step 2a: Intra-Task Pruning (同一タスク内での剪定) ---
        # 凍結済み(Mature, rank > 1) かつ所属タスクが記録されているニューロンを収集
        mature_neurons_by_task = {}
        mature_indices = np.where(ranks > 1)[0]

        for neuron_idx in mature_indices:
            task = model.neuron_birth_task[layer_name].get(neuron_idx)
            if task is not None:
                if task not in mature_neurons_by_task:
                    mature_neurons_by_task[task] = []
                mature_neurons_by_task[task].append(neuron_idx)

        print(f"  Found mature neurons in {len(mature_neurons_by_task)} tasks.")

        # --- Step 2b: Inter-Task Sharing (異種タスク間での共有化) ---
        # mature_neurons_by_task は Step 2a で計算済みのものを再利用
        if len(mature_neurons_by_task.keys()) < 2: continue # 比較対象のタスクがなければスキップ

        print(f"  Analyzing interactions between {len(mature_neurons_by_task.keys())} tasks...")
        # 異なるタスクのニューロンペアをすべて作成
        shared_neurons = [] # この層で既に共有化に関与したニューロン
        for (task1, neurons1), (task2, neurons2) in combinations(mature_neurons_by_task.items(), 2):
            for n1 in neurons1:
                for n2 in neurons2:
                    if n1 in shared_neurons or n2 in shared_neurons:
                        continue
                    
                    similarity = analyzer.calculate_cosine_similarity(layer_name, n1, n2)
                    
                    if similarity > args.threshold_inter_task_sharing:
                        print(f"  >> SHARING TRIGGERED (Inter-Task): Neurons ({n1}, {n2}) from tasks ({task1}, {task2}). Similarity: {similarity:.4f}")
                        # より新しく学習した（タスク番号が大きい）ニューロンを、古い方に統合する
                        source_neuron = max(n1, n2, key=lambda n: model.neuron_birth_task[layer_name].get(n, -1))
                        target_neuron = min(n1, n2, key=lambda n: model.neuron_birth_task[layer_name].get(n, -1))
                        
                        # 念のため、sourceとtargetが同じにならないようにチェック
                        if source_neuron != target_neuron:
                            controller.share_neurons(layer_name, source_neuron, target_neuron)
                            shared_neurons.extend([source_neuron, target_neuron])


    print("\n--- SYNAPSE Phase finished. ---\n")