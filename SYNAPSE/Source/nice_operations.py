# SYNAPSE/Source/nice_operations.py

import copy
from typing import Any, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from Source.architecture import SparseConv2d, SparseLinear, SparseOutput
from Source.helper import get_device
from Source.resnet18 import ResNet18


def get_current_young_neurons(unit_ranks: List) -> List:
    """未熟ニューロン（タスクリストが空）のインデックスを取得する"""
    current_young_neurons = []
    for ranks_in_layer, _ in unit_ranks:
        young_indices = [idx for idx, rank_list in enumerate(ranks_in_layer) if not rank_list]
        current_young_neurons.append(young_indices)
    return current_young_neurons


def get_current_learner_neurons(unit_ranks: List, current_task_id: int) -> List:
    """現在のタスクの学習者ニューロンのインデックスを取得する"""
    current_learner_neurons = []
    for ranks_in_layer, _ in unit_ranks:
        learner_indices = [idx for idx, rank_list in enumerate(ranks_in_layer) if rank_list == [current_task_id]]
        current_learner_neurons.append(learner_indices)
    return current_learner_neurons


def increase_unit_ranks(network: Any) -> Any:
    """タスク完了後、学習者ニューロンを成熟済みにする（何もしない）"""
    return network


def update_freeze_masks(network: Any) -> Any:
    """成熟済みニューロンを凍結するためのマスクを更新する"""
    if isinstance(network, ResNet18):
        return network

    weights = network.get_weight_bias_masks_numpy()
    freeze_masks = []
    
    ranks_without_input = [ranks for ranks, name in network.unit_ranks[1:]]
    
    if len(weights) != len(ranks_without_input):
        print(f"[WARNING] Layer count mismatch in update_freeze_masks. Weights: {len(weights)}, Ranks: {len(ranks_without_input)}")
        return network

    for i, ranks_in_layer in enumerate(ranks_without_input):
        weight_pair = weights[i]
        mature_indices = np.array([idx for idx, r in enumerate(ranks_in_layer) if r], dtype=np.int32)
        
        mask_w = np.zeros(weight_pair[0].shape)
        mask_b = np.zeros(weight_pair[1].shape)

        if len(mature_indices) > 0:
            mask_w[mature_indices, :] = 1
            mask_b[mature_indices] = 1
        
        freeze_masks.append((mask_w * weight_pair[0], mask_b))

    freeze_masks_tensors = [(torch.tensor(w, dtype=torch.bool).to(get_device()),
                             torch.tensor(b, dtype=torch.bool).to(get_device()))
                            for w, b in freeze_masks]
    network.freeze_masks = freeze_masks_tensors
    return network


def pick_top_neurons(scores, selection_ratio) -> List[int]:
    """活性化スコア上位のニューロンを選択するヘルパー関数"""
    if torch.sum(scores).item() == 0:
        return []
    total = torch.sum(scores)
    accumulate = 0
    indices = []
    sort_indices = torch.argsort(-scores)
    for index in sort_indices:
        index_val = index.item()
        if scores[index_val] == 0:
            continue
        accumulate += scores[index_val]
        indices.append(index_val)
        if accumulate >= total * selection_ratio / 100.0:
            break
    return indices


def select_learner_units(network: Any, stable_selection_perc: float, train_episode: Any, episode_index: int) -> Any:
    if isinstance(network, ResNet18):
        return network

    top_unit_indices = []
    
    intermediate_ranks = [ranks for ranks, name in network.unit_ranks[1:-1]]
    
    if stable_selection_perc == 100.0:
        for ranks in intermediate_ranks:
             selectable = [idx for idx, r in enumerate(ranks) if not r]
             top_unit_indices.append(selectable)
    else:
        loader = DataLoader(train_episode.dataset, batch_size=1024, shuffle=False)
        data, _, _ = next(iter(loader))
        data = data.to(get_device())
        
        _, layer_activations = network.get_activations(data, return_output=True)
        
        for act, ranks in zip(layer_activations[1:-1], intermediate_ranks):
            if act.dim() > 2:
                scores = torch.sum(act, dim=(0, 2, 3))
            else:
                scores = torch.sum(act, dim=0)
            
            mature_indices = [idx for idx, r in enumerate(ranks) if r and r != [episode_index]]
            if mature_indices:
                scores[mature_indices] = 0.0
            
            selected = pick_top_neurons(scores, stable_selection_perc)
            current_learners = [idx for idx, r in enumerate(ranks) if r == [episode_index]]
            final_selection = sorted(list(set(selected + current_learners)))
            top_unit_indices.append(final_selection)
    
    output_layer_ranks, _ = network.unit_ranks[-1]
    output_learners = [idx for idx, r in enumerate(output_layer_ranks) if r == [episode_index]]
    output_units = sorted(list(set(train_episode.classes_in_this_experience + output_learners)))
    top_unit_indices.append(output_units)

    new_unit_ranks_list = network.unit_ranks[1:] 
    for i, selected_for_layer in enumerate(top_unit_indices):
        ranks, name = new_unit_ranks_list[i]
        new_ranks = copy.deepcopy(ranks)
        for unit_idx in selected_for_layer:
            if unit_idx < len(new_ranks):
                if not new_ranks[unit_idx]:
                    new_ranks[unit_idx] = [episode_index]
        new_unit_ranks_list[i] = (new_ranks, name)
    
    network.unit_ranks = [network.unit_ranks[0]] + new_unit_ranks_list
    return network


def grow_all_to_young(network: Any) -> Any:
    """
    次の層の未熟ニューロンへの接続をすべて有効化（成長）させる。
    """
    if isinstance(network, ResNet18):
        return network

    all_young_indices = get_current_young_neurons(network.unit_ranks)
    module_list = [m for m in network.modules() if isinstance(m, (SparseLinear, SparseConv2d, SparseOutput))]

    for i, module in enumerate(module_list):
        # 次の層のインデックスは i+2 (入力層をunit_ranks[0]で考慮するため)
        if i + 2 >= len(all_young_indices):
            continue
        next_layer_young_idx = all_young_indices[i + 2]
        if not next_layer_young_idx:
            continue

        weight_mask, bias_mask = module.get_mask()
        device = weight_mask.device
        
        # =================================================================
        # === 曖昧なインデックス指定を index_fill_ に変更 ===
        # =================================================================
        rows_to_fill = torch.tensor(next_layer_young_idx, dtype=torch.long, device=device)
        
        # weight_maskの最初の次元（出力チャンネル/特徴）を指定して1で埋める
        weight_mask.index_fill_(0, rows_to_fill, 1)
        
        # bias_maskも同様に更新
        if bias_mask is not None:
            bias_mask.index_fill_(0, rows_to_fill, 1)
        # =================================================================

        module.set_mask(weight_mask, bias_mask)

    return network


def drop_young_to_learner(network: Any) -> Any:
    """
    現在の層の未熟ニューロンから、次の層の成熟済みニューロンへの接続を削除する。
    """
    if isinstance(network, ResNet18):
        return network

    all_young_indices = get_current_young_neurons(network.unit_ranks)
    
    mature_neurons_per_layer = []
    for ranks, _ in network.unit_ranks:
        mature_indices = [idx for idx, r in enumerate(ranks) if r]
        mature_neurons_per_layer.append(mature_indices)

    module_list = [m for m in network.modules() if isinstance(m, (SparseLinear, SparseConv2d, SparseOutput))]

    for i, module in enumerate(module_list):
        # 現在の層は i+1
        current_layer_young_idx = all_young_indices[i + 1]
        
        # 次の層は i+2
        if i + 2 >= len(mature_neurons_per_layer):
            continue
        next_layer_mature_idx = mature_neurons_per_layer[i + 2]

        if not current_layer_young_idx or not next_layer_mature_idx:
            continue

        weight_mask, _ = module.get_mask() # bias_maskは変更しないので不要

        # =================================================================
        # =2D/4Dテンソルに対応した、より安全なインデックス指定 ===
        # =================================================================
        rows = torch.tensor(next_layer_mature_idx, dtype=torch.long)
        cols = torch.tensor(current_layer_young_idx, dtype=torch.long)

        # 選択された行(rows)の、特定の列(cols)だけを0にする
        if weight_mask.dim() == 4: # Conv層: (out_c, in_c, kH, kW)
             for r in rows:
                 weight_mask[r, cols, :, :] = 0
        else: # Linear層: (out_f, in_f)
            for r in rows:
                 weight_mask[r, cols] = 0
        # =================================================================
        # set_maskはループの外で行う方が効率的だが、可読性のためここで行う
        # module.set_mask(weight_mask, bias_mask) # weight_maskしか変更していない

    return network