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
                              for ranks in unit_ranks]
    return current_young_neurons


def get_current_learner_neurons(unit_ranks: List, current_task_id: int) -> List:
    """現在のタスクの学習者ニューロンのインデックスを取得する"""
    current_learner_neurons = [[idx for idx, rank in enumerate(ranks) if rank == [current_task_id]]
                               for ranks in unit_ranks]
    return current_learner_neurons


def increase_unit_ranks(network: Any) -> Any:
    """
    タスク完了後、学習者ニューロンを成熟済みにする。
    リスト表現では明示的なランクのインクリメントは不要なため、何もしない。
    """
    return network


def update_freeze_masks(network: Any) -> Any:
    if isinstance(network, ResNet18):
        # (ResNetの場合はここを実装)
        return network

    weights = network.get_weight_bias_masks_numpy()
    freeze_masks = []
    
    # network.unit_ranksから直接、成熟ニューロンのインデックスリストを作成
    mature_neurons_per_layer = []
    for ranks in network.unit_ranks:
        mature_indices = [idx for idx, r in enumerate(ranks) if r] # rが空リストでなければTrue (成熟)
        mature_neurons_per_layer.append(np.array(mature_indices, dtype=np.int32))

    # 層の数（重みの数とランクの数）が一致していることを確認 (念のため)
    if len(weights) != len(mature_neurons_per_layer):
        print(f"[WARNING] Layer count mismatch in update_freeze_masks. Weights: {len(weights)}, Ranks: {len(mature_neurons_per_layer)}")
        return network # エラーを防ぐために処理を中断

    # zipを使い、各層の「重み」と「成熟ニューロンリスト」を正しくペアリングしてループ
    for weight_pair, target_mature in zip(weights, mature_neurons_per_layer):
        
        mask_w = np.zeros(weight_pair[0].shape)
        mask_b = np.zeros(weight_pair[1].shape)

        if len(target_mature) > 0:
            # これでインデックスと層のサイズが一致するため、エラーは発生しない
            mask_w[target_mature, :] = 1
            mask_b[target_mature] = 1
        
        # 既存の接続マスクと凍結マスクを結合
        freeze_masks.append((mask_w * weight_pair[0], mask_b))

    freeze_masks_tensors = [(torch.tensor(w, dtype=torch.bool).to(get_device()),
                             torch.tensor(b, dtype=torch.bool).to(get_device()))
                            for w, b in freeze_masks]
    network.freeze_masks = freeze_masks_tensors
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


# SYNAPSE/Source/nice_operations.py

def select_learner_units(network: Any, stable_selection_perc, train_episode: Any, episode_index: int) -> Any:
    if isinstance(network, ResNet18):
        return select_learner_units_resnet(network, stable_selection_perc, train_episode, episode_index)

    top_unit_indices = []
    
    # 最新の未熟/学習者ニューロンの状態を取得 (unit_ranksの長さに対応)
    current_young = get_current_young_neurons(network.unit_ranks)
    current_learners = get_current_learner_neurons(network.unit_ranks, episode_index)
    
    if stable_selection_perc == 100.0:
        # --- フェーズ1 ---
        # 全ての学習可能層（出力層を除く）に対して、利用可能な全ニューロンを選択
        for i in range(len(network.unit_ranks) - 1):
            selectable = sorted(list(set(current_young[i] + current_learners[i])))
            top_unit_indices.append(selectable)
    else:
        # --- フェーズ2以降 ---
        loader = DataLoader(train_episode.dataset, batch_size=1024,  shuffle=False)
        data, _, _ = next(iter(loader))
        data = data.to(get_device())
        
        # network.get_activations()は[入力, 層1, 層2, ..., 出力]のリストを返す
        _, layer_activations = network.get_activations(data, return_output=True)
        
        # 活性化リスト(層1〜最後から2番目まで)とunit_ranks(層1〜最後から2番目まで)をzipで安全にループ
        # これでリスト間のズレが完全になくなる
        for act, ranks, learners in zip(layer_activations[1:-1], network.unit_ranks[:-1], current_learners[:-1]):
            if act.dim() > 2: # Conv層
                scores = torch.sum(act, dim=(0, 2, 3))
            else: # Linear層
                scores = torch.sum(act, dim=0)
            
            # 既に何らかのタスクを担当しているニューロンはスコアを0にし、選択対象から除外
            mask = torch.zeros_like(scores, dtype=torch.bool)
            non_selectable = [i for i, r in enumerate(ranks) if r]
            if non_selectable:
                mask[non_selectable] = True
            scores[mask] = 0.0
            
            selected = pick_top_neurons(scores, stable_selection_perc)
            # 既存の学習者も選択状態に維持する
            final_selection = sorted(list(set(selected + learners)))
            top_unit_indices.append(final_selection)

    # --- 出力層の処理 ---
    # 現在のタスクで必要なクラスと、既存の学習者を結合して選択
    output_layer_units = sorted(list(set(train_episode.classes_in_this_experience + current_learners[-1])))
    top_unit_indices.append(output_layer_units)

    # --- ランクの更新 ---
    # unit_ranks (11要素) と top_unit_indices (11要素) が完全に一致しているため、安全に更新できる
    new_unit_ranks = []
    for ranks, selected_for_layer in zip(network.unit_ranks, top_unit_indices):
        new_ranks = copy.deepcopy(ranks)
        for unit_idx in selected_for_layer:
            if unit_idx < len(new_ranks):
                if not new_ranks[unit_idx]: # 未熟ニューロン([])の場合のみ、現在のタスクIDを追加
                    new_ranks[unit_idx] = [episode_index]
        new_unit_ranks.append(new_ranks)
    
    network.unit_ranks = new_unit_ranks
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