# Source/synapse_operations.py

import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Any, List, Dict

from Source.helper import get_device, SparseLinear, SparseConv2d, SparseOutput

# SYNAPSE: ニューロンの状態を定義します。
# Immature: 待機中のニューロン
# Transitional: 新規タスクへの適応中に訓練されるニューロン
# Mature: 特定クラスに特化し、重みが保護されるニューロン
IMMATURE = -1
TRANSITIONAL = 0
MATURE_BASE_RANK = 1  # Matureニューロンは1以上のランク値を持ちます

def set_neuron_state(network: Any, layer_index: int, neuron_indices: list, new_state: int) -> Any:
    """
    指定されたニューロンの状態（ランク）を変更します。

    Args:
        network: ネットワークのインスタンス
        layer_index: 対象となる層のインデックス
        neuron_indices: 状態を変更するニューロンのインデックスリスト
        new_state: 新しい状態 (IMMATURE, TRANSITIONAL, またはMatureランク)

    Returns:
        状態が更新されたネットワーク
    """
    if len(neuron_indices) == 0:
        return network

    ranks, name = network.unit_ranks[layer_index]
    ranks[neuron_indices] = new_state
    network.unit_ranks[layer_index] = (ranks, name)

    # 変更を反映させるために、状態ごとのニューロンリストを更新します
    network.update_neuron_state_lists()
    return network

def add_new_neuron(network: Any, layer_index: int, num_to_add: int = 1) -> Any:
    """
    Immature Poolからニューロンを取り出し、Transitional状態に設定します。
    これはSYNAPSEの「Add」操作の核となる部分です。

    Args:
        network: ネットワークのインスタンス
        layer_index: ニューロンを追加する層のインデックス
        num_to_add: 追加するニューロンの数

    Returns:
        ニューロンが追加（状態変更）されたネットワーク
    """
    immature_pool = network.immature_neurons[layer_index]
    if len(immature_pool) < num_to_add:
        print(f"警告: 層 {layer_index} のImmatureニューロンが不足しています！")
        # 本来はここで補充(rebalancing)処理を呼び出すことも考えられます
        return network

    # Poolの先頭から必要な数だけニューロンを選択します
    new_neurons = immature_pool[:num_to_add]

    # 選択したニューロンの状態をTransitionalに変更します
    network = set_neuron_state(network, layer_index, new_neurons, TRANSITIONAL)
    print(f"層 {layer_index} に {len(new_neurons)} 個のニューロンをTransitionalとして追加しました。")
    return network

def mature_transitional_neurons(network: Any) -> Any:
    """
    現在Transitional状態にある全てのニューロンをMature状態へ遷移させます。
    学習フェーズの完了後に呼び出されます。

    Returns:
        ニューロンが成熟化されたネットワーク
    """
    print("Transitional状態のニューロンをMature化します...")
    for layer_idx, (ranks, name) in enumerate(network.unit_ranks):
        transitional_indices = (ranks == TRANSITIONAL).nonzero()[0]
        if len(transitional_indices) > 0:
            # 新しいMatureランクを決定します。
            # 例えば、既存のMatureニューロンの最大ランク+1、といった戦略が考えられます。
            max_rank = np.max(ranks) if np.any(ranks >= MATURE_BASE_RANK) else 0
            new_rank = max(MATURE_BASE_RANK, int(max_rank) + 1)
            network = set_neuron_state(network, layer_idx, transitional_indices, new_rank)
    print("ニューロンの成熟化が完了しました。")
    return network


def update_freeze_masks_synapse(network: Any) -> Any:
    """
    SYNAPSEの仕様に基づき、Matureニューロンの重みを凍結（保護）するためのマスクを更新します。
    """
    print("Matureニューロンの重みを保護するためのフリーズマスクを更新します。")
    
    # 1. unit_ranksから「名札」をキーにした対応表を作成
    ranks_by_name = {name: ranks for ranks, name in network.unit_ranks}
    mature_neurons_by_name = {
        name: list(np.where(ranks >= MATURE_BASE_RANK)[0]) 
        for name, ranks in ranks_by_name.items()
    }

    freeze_masks = []
    
    # 2. ネットワークの全モジュールを巡回
    for module in network.modules():
        if isinstance(module, (SparseLinear, SparseConv2d, SparseOutput)):
            # モジュールの「名札」を取得
            layer_name = module.layer_name
            
            # 名札が対応表に無ければスキップ
            if layer_name not in mature_neurons_by_name:
                continue

            # 対応表から、この層のMatureニューロンのリストを取得
            mature_neuron_indices = mature_neurons_by_name[layer_name]

            weight_mask = torch.zeros_like(module.weight.data, dtype=torch.bool)
            bias_mask = torch.zeros_like(module.bias.data, dtype=torch.bool)

            if mature_neuron_indices:
                # テンソルの次元数に応じてインデックス指定を切り替える
                if module.weight.data.dim() == 4: # 畳み込み層
                    weight_mask[mature_neuron_indices, :, :, :] = True
                else: # 全結合層
                    weight_mask[mature_neuron_indices, :] = True
                
                bias_mask[mature_neuron_indices] = True
            
            freeze_masks.append((weight_mask.to(get_device()), bias_mask.to(get_device())))
            
    network.freeze_masks = freeze_masks
    
    # BatchNorm層の凍結ロジック (こちらもlayer_nameベースでより堅牢に)
    for module in network.modules():
        if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
            if hasattr(module, 'layer_name'):
                layer_name = module.layer_name
                if layer_name in ranks_by_name:
                    ranks = ranks_by_name[layer_name]
                    frozen_units = torch.tensor(ranks >= MATURE_BASE_RANK, dtype=torch.bool).to(get_device())
                    if hasattr(module, 'freeze_units'):
                        module.freeze_units(frozen_units)

    return network


def get_most_activated_mature_neuron(network: Any, data_loader: DataLoader) -> tuple[int, int] | None:
    """
    データセットに対して、最も平均活性度が高かったMatureニューロンを見つけます。
    """
    print("最も活性化したMatureニューロンを探索します...")
    network.eval()
    
    best_neuron_info = None
    max_activation = -1.0

    # 全ての層のMatureニューロンの活性度を保持する辞書
    activations_sum = {layer_idx: torch.zeros(len(neurons)).to(get_device()) for layer_idx, neurons in enumerate(network.mature_neurons) if neurons}
    data_count = 0

    with torch.no_grad():
        for data, _, _ in data_loader:
            data = data.to(get_device())
            data_count += data.shape[0]
            _, all_layer_activations = network.get_activations(data, return_output=True)

            for layer_idx, mature_indices in enumerate(network.mature_neurons):
                if not mature_indices: continue
                
                layer_activations = all_layer_activations[layer_idx]
                if len(layer_activations.shape) > 2:
                    layer_activations = layer_activations.mean(dim=[2, 3])
                
                # このバッチでの活性度を合計に加算
                activations_sum[layer_idx] += layer_activations[:, mature_indices].sum(dim=0)
    
    # 平均活性度が最大のニューロンを探す
    for layer_idx, sums in activations_sum.items():
        if data_count > 0:
            avg_activations = sums / data_count
            current_max = torch.max(avg_activations)
            if current_max > max_activation:
                max_activation = current_max
                # 元のニューロンインデックスを取得
                original_neuron_idx = network.mature_neurons[layer_idx][torch.argmax(avg_activations).item()]
                best_neuron_info = (layer_idx, original_neuron_idx)

    network.train()
    if best_neuron_info:
        print(f"探索完了: 層 {best_neuron_info[0]}, ニューロン {best_neuron_info[1]} が最も活性化しました。")
    else:
        print("探索失敗: Matureニューロンが見つかりませんでした。")
        
    return best_neuron_info

def get_activation_patterns(network: Any, data_loader: DataLoader, layer_idx: int, neuron_indices: List[int]) -> torch.Tensor | None:
    if not neuron_indices: return None
    network.eval()
    all_activations = []
    with torch.no_grad():
        for data, _, _ in data_loader:
            data = data.to(get_device())
            _, activations = network.get_activations(data, return_output=True)
            target_layer_activations = activations[layer_idx]
            if len(target_layer_activations.shape) > 2:
                target_layer_activations = target_layer_activations.mean(dim=[2, 3])
            all_activations.append(target_layer_activations[:, neuron_indices])
    network.train()
    if not all_activations: return None
    return torch.cat(all_activations, dim=0).T

def integrate_neurons(network: Any, data_loader: DataLoader, threshold: float = 0.95) -> Any:
    print("Integrate操作: 冗長なニューロンの統合を開始します...")
    module_map = {idx+1: m for idx, m in enumerate(network.modules()) if isinstance(m, (SparseLinear, SparseConv2d, SparseOutput))}

    for layer_idx, mature_indices in enumerate(network.mature_neurons):
        if len(mature_indices) < 2 or layer_idx not in module_map: continue

        patterns = get_activation_patterns(network, data_loader, layer_idx, mature_indices)
        if patterns is None: continue

        patterns_norm = F.normalize(patterns, p=2, dim=1)
        similarity_matrix = patterns_norm @ patterns_norm.T
        redundant_pairs = (similarity_matrix > threshold).nonzero()

        processed_neurons = set()
        for pair in redundant_pairs:
            u, v = pair[0].item(), pair[1].item()
            if u >= v or u in processed_neurons or v in processed_neurons: continue

            neuron_u_idx = mature_indices[u]
            neuron_v_idx = mature_indices[v]
            print(f"  - 層 {layer_idx} のニューロン {neuron_v_idx} を {neuron_u_idx} に統合します。")

            module = module_map[layer_idx]
            with torch.no_grad():
                # 重みを平均化
                avg_weight = (module.weight.data[neuron_u_idx] + module.weight.data[neuron_v_idx]) / 2.0
                module.weight.data[neuron_u_idx] = avg_weight
                if module.bias is not None:
                    avg_bias = (module.bias.data[neuron_u_idx] + module.bias.data[neuron_v_idx]) / 2.0
                    module.bias.data[neuron_u_idx] = avg_bias
            
            # v を引退させ、Immature Poolに戻す
            network = set_neuron_state(network, layer_idx, [neuron_v_idx], IMMATURE)
            processed_neurons.add(u)
            processed_neurons.add(v)
            
    return network

def share_neuron(network: Any, layer_index: int, neuron_index: int) -> Any:
    """
    「Share」操作。既存のMatureニューロンを再学習の対象にします。
    状態をMatureからTransitionalへ変更します。
    """
    print(f"Share操作: 層 {layer_index} のMatureニューロン {neuron_index} をTransitional状態に変更します。")
    network = set_neuron_state(network, layer_index, [neuron_index], TRANSITIONAL)
    return network

def duplicate_neuron(network: Any, source_layer_idx: int, source_neuron_idx: int) -> Any:
    """
    「Duplicate」操作。既存のMatureニューロンを複製します。
    """
    print(f"Duplicate操作: 層 {source_layer_idx} のニューロン {source_neuron_idx} を複製します。")
    # 1. Immature Poolから新しいニューロンを取得
    immature_pool = network.immature_neurons[source_layer_idx]
    if not immature_pool:
        print(f"警告: Duplicate操作失敗。層 {source_layer_idx} にImmatureニューロンがありません。")
        return network
    target_neuron_idx = immature_pool[0]

    # 2. 重みをコピー
    #    ネットワークのモジュールをたどり、対象の重みを見つける必要があります。
    module_found = False
    module_idx = 0
    for module in network.modules():
        if isinstance(module, (torch.nn.Linear, SparseConv2d)):
            if module_idx == source_layer_idx -1: # unit_ranksのインデックスと合わせる
                # 重みとバイアスをコピー
                with torch.no_grad():
                    # sourceの重みを取得
                    source_weight = module.weight.data[source_neuron_idx]
                    # targetにコピー
                    module.weight.data[target_neuron_idx] = source_weight.clone()

                    if module.bias is not None:
                        source_bias = module.bias.data[source_neuron_idx]
                        module.bias.data[target_neuron_idx] = source_bias.clone()
                module_found = True
                break
            module_idx += 1
    
    if not module_found:
        print("警告: 重みコピー対象のモジュールが見つかりませんでした。")
        return network

    # 3. 複製したニューロンの状態をTransitionalに設定
    network = set_neuron_state(network, source_layer_idx, [target_neuron_idx], TRANSITIONAL)
    
    return network


def initialize_strategically(network: Any, new_neurons_info: Dict[int, List[int]]):
    """
    「Add」操作で追加されたニューロンを戦略的に初期化します。
    既存のMatureニューロン群の重みの平均から逆方向に初期化することで、
    特徴空間上で未探索の領域を向くように促します。
    """
    print("戦略的初期化を実行します...")
    
    module_idx = 0
    for module in network.modules():
        if isinstance(module, (SparseLinear, SparseConv2d, SparseOutput)):
            layer_idx = module_idx + 1 # unit_ranksのインデックスに合わせる
            
            # この層が初期化対象か確認
            if layer_idx in new_neurons_info:
                new_neuron_indices = new_neurons_info[layer_idx]
                mature_neuron_indices = network.mature_neurons[layer_idx]
                
                # 初期化の基準となる重みを取得
                if mature_neuron_indices:
                    # Matureニューロンが存在すれば、その重みの平均を計算
                    base_weights = module.weight.data[mature_neuron_indices]
                    mean_weight = torch.mean(base_weights, dim=0)
                else:
                    # 存在しなければ、ゼロベクトルを基準とする
                    mean_weight = torch.zeros_like(module.weight.data[0])

                # 新しいニューロンを初期化
                with torch.no_grad():
                    for neuron_idx in new_neuron_indices:
                        # 平均の逆方向ベクトルに、少しランダム性を加える
                        noise = torch.randn_like(mean_weight) * 0.01
                        module.weight.data[neuron_idx] = -mean_weight + noise
                        
                        if module.bias is not None:
                            module.bias.data[neuron_idx] = 0.0
                
                print(f"  - 層 {layer_idx} のニューロン {new_neuron_indices} を戦略的に初期化しました。")

            module_idx += 1
            
    return network

def replenish_immature_pool(network: Any) -> Any:
    """
    「Replenish」操作。Immatureニューロンが不足していたら補充します。
    """
    print("Replenish操作: Immature Poolの補充を確認します...")
    for layer_idx, (ranks, name) in enumerate(network.unit_ranks):
        if layer_idx == 0: continue # 入力層はスキップ

        num_units = len(ranks)
        num_immature = len(network.immature_neurons[layer_idx])
        required_immature = int(num_units * network.immature_pool_ratio)
        
        if num_immature < required_immature:
            num_to_replenish = required_immature - num_immature
            
            # Matureニューロンの中から引退させる候補を選ぶ
            # ここでは単純にランクが一番低い（=一番新しい）Matureニューロンを引退させる
            mature_indices = network.mature_neurons[layer_idx]
            if len(mature_indices) > num_to_replenish:
                
                # ランクでソートして、ランクが低い（新しい）ものから選ぶ
                mature_ranks = ranks[mature_indices]
                sorted_mature_indices = [x for _, x in sorted(zip(mature_ranks, mature_indices))]
                
                neurons_to_retire = sorted_mature_indices[:num_to_replenish]
                network = set_neuron_state(network, layer_idx, neurons_to_retire, IMMATURE)
                print(f"  - 層 {layer_idx} で {len(neurons_to_retire)} 個のニューロンを補充のためにImmatureにしました。")

    return network