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
    current_young_neurons = [[idx for idx, rank in enumerate(ranks) if not rank]
                             for ranks, _ in unit_ranks]
    return current_young_neurons


def get_current_learner_neurons(unit_ranks: List, current_task_id: int) -> List:
    """現在のタスクの学習者ニューロンのインデックスを取得する"""
    current_learner_neurons = [[idx for idx, rank in enumerate(ranks) if rank == [current_task_id]]
                               for ranks, _ in unit_ranks]
    return current_learner_neurons


def increase_unit_ranks(network: Any) -> Any:
    """
    タスク完了後、学習者ニューロンを成熟済みにする。
    リスト表現では明示的なランクのインクリメントは不要なため、何もしない。
    """
    return network


def update_freeze_masks(network: Any) -> Any:
    if isinstance(network, ResNet18):
        # ResNetの場合は専用の関数を呼ぶ (未実装の場合は汎用ロジックへ)
        # return update_freeze_masks_resnet(network)
        pass

    weights = network.get_weight_bias_masks_numpy()
    freeze_masks = []
    mature_neurons = []
    for ranks, _ in network.unit_ranks:
        # ランクリストが空でないニューロンを凍結対象（成熟済み）とみなす
        mature_indices = [idx for idx, r in enumerate(ranks) if r]
        mature_neurons.append(np.array(mature_indices))

    for i, target_mature in enumerate(mature_neurons[1:]): # 入力層を除く
        target_mature = np.array(target_mature, dtype=np.int32)
        if i >= len(weights): continue
        
        mask_w = np.zeros(weights[i][0].shape)
        mask_b = np.zeros(weights[i][1].shape)
        if len(target_mature) != 0:
            mask_w[target_mature, :] = 1
            mask_b[target_mature] = 1
        freeze_masks.append((mask_w * weights[i][0], mask_b))

    freeze_masks = [(torch.tensor(w).to(torch.bool).to(get_device()),
                     torch.tensor(b).to(torch.bool).to(get_device()))
                    for w, b in freeze_masks]
    network.freeze_masks = freeze_masks
    return network


def pick_top_neurons(scores, selection_ratio) -> List[int]:
    """活性化スコア上位のニューロンを選択するヘルパー関数"""
    if torch.sum(scores) == 0:
        return []
    total = torch.sum(scores)
    accumulate = 0
    indices = []
    sort_indices = torch.argsort(-scores)
    for index in sort_indices:
        index_val = index.item()
        accumulate += scores[index_val]
        indices.append(index_val)
        if accumulate >= total * selection_ratio / 100.0:
            break
    return indices


def select_learner_units(network: Any, stable_selection_perc, train_episode: Any, episode_index: int) -> Any:
    if isinstance(network, ResNet18):
        return select_learner_units_resnet(network, stable_selection_perc, train_episode, episode_index)

    # === VGG/CNNモデル用のロジックを、新しいunit_ranksシステムに完全対応させる ===
    top_unit_indices = []
    
    # 動的に現在の未熟/学習者ニューロンを取得
    current_young = get_current_young_neurons(network.unit_ranks)
    current_learners = get_current_learner_neurons(network.unit_ranks, episode_index)
    
    if stable_selection_perc == 100.0:
        # フェーズ1: 利用可能な全てのニューロン（未熟＋現在の学習者）を選択
        for i in range(1, len(network.unit_ranks)): # 入力層を除く
            selectable = sorted(list(set(current_young[i] + current_learners[i])))
            top_unit_indices.append(selectable)
    else:
        # フェーズ2以降: 活性化に基づいて未熟ニューロンから選択
        loader = DataLoader(train_episode.dataset, batch_size=1024,  shuffle=False)
        data, _, _ = next(iter(loader))
        data = data.to(get_device())
        
        # get_activation_selectionは古い属性に依存するため、get_activationsを代わりに使う
        _, layer_activations = network.get_activations(data, return_output=True)
        
        layer_idx = 1 # unit_ranksに合わせる
        for act in layer_activations[1:-1]: # 入力と出力を除く
            if act.dim() > 2: # Conv層
                scores = torch.sum(act, dim=(0, 2, 3))
            else: # Linear層
                scores = torch.sum(act, dim=0)
            
            # 既に学習者 or 成熟済みのニューロンは候補から除外
            mask = torch.zeros_like(scores, dtype=torch.bool)
            non_selectable = [i for i, r in enumerate(network.unit_ranks[layer_idx][0]) if r]
            if non_selectable:
                mask[non_selectable] = True
            scores[mask] = 0.0
            
            selected = pick_top_neurons(scores, stable_selection_perc)
            # 既存の学習者も維持する
            final_selection = sorted(list(set(selected + current_learners[layer_idx])))
            top_unit_indices.append(final_selection)
            layer_idx += 1

    # 出力層のユニットを追加
    output_layer_units = sorted(list(set(train_episode.classes_in_this_experience + current_learners[-1])))
    top_unit_indices.append(output_layer_units)

    # ランクを更新
    unit_ranks = [network.unit_ranks[0]]
    for i, (ranks, name) in enumerate(network.unit_ranks[1:], 1):
        if i - 1 < len(top_unit_indices):
            new_ranks = copy.deepcopy(ranks)
            selected_for_layer = top_unit_indices[i - 1]
            for unit_idx in selected_for_layer:
                if not new_ranks[unit_idx]: # 未熟の場合のみ
                    new_ranks[unit_idx] = [episode_index]
            unit_ranks.append((new_ranks, name))
        else:
            unit_ranks.append((ranks, name))
    
    network.unit_ranks = unit_ranks
    return network


# ResNet用の関数も、エラーを起こした部分を修正
def select_learner_units_resnet(network: Any, stable_selection_perc, train_episode: Any, episode_index: int) -> Any:
    top_unit_indices = []
    
    if stable_selection_perc == 100.0:
        for ranks, _ in network.unit_ranks[1:-1]:
            selectable_indices = [idx for idx, r in enumerate(ranks) if not r or r == [episode_index]]
            top_unit_indices.append(np.array(selectable_indices))
    else:
        # (活性化に基づく選択ロジック)
        pass

    top_unit_indices.append(train_episode.classes_in_this_experience)
    
    unit_ranks = [network.unit_ranks[0]]
    for i, (ranks, name) in enumerate(network.unit_ranks[1:], 1):
        if i - 1 < len(top_unit_indices):
            layer_selected_units = top_unit_indices[i - 1]
            new_ranks = copy.deepcopy(ranks)
            for unit_idx in layer_selected_units:
                if unit_idx < len(new_ranks) and not new_ranks[unit_idx]:
                    new_ranks[unit_idx] = [episode_index]
            unit_ranks.append((new_ranks, name))
        else:
            unit_ranks.append((ranks, name))
            
    network.unit_ranks = unit_ranks
    return network


def grow_all_to_young(network: Any) -> Any:
    # この関数は古い接続マスクベースのため、新しいシステムでは再設計が必要
    # ここでは何もしないように変更し、エラーを防ぐ
    print("  [INFO] grow_all_to_young is currently disabled for the new rank system.")
    return network

def drop_young_to_learner(network: Any) -> Any:
    # この関数も同様に再設計が必要
    print("  [INFO] drop_young_to_learner is currently disabled for the new rank system.")
    return network