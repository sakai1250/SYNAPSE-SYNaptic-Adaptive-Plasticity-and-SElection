# SYNAPSE/Source/synapse_operations.py

import torch
import torch.nn as nn
import numpy as np
from itertools import combinations, product
from typing import Any, Dict, List
import copy

from argparse import Namespace
from avalanche.benchmarks import TCLExperience
from Source.helper import get_device, reduce_or_flat_convs, SparseConv2d, SparseLinear, SparseOutput
from Source.context_detector import get_n_samples_per_class


def centered_kernel_alignment(X, Y):
    """
    2つの活性化行列間の線形CKAを計算する。
    """
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    XTX = X.T @ X
    YTY = Y.T @ Y

    numerator = torch.norm(Y.T @ X, p='fro')**2
    denominator = torch.norm(XTX, p='fro') * torch.norm(YTY, p='fro')

    return (numerator / denominator).item() if denominator > 0 else 0.0

def reinitialize_neurons(model: Any, layer_idx: int, neuron_indices: List[int]):
    """
    指定された層の特定のニューロンの重みとバイアスを再初期化する。
    """
    if not neuron_indices:
        return

    all_modules = [m for m in model.modules() if isinstance(m, (SparseConv2d, SparseLinear, SparseOutput))]
    target_module_idx = layer_idx - 1
    if target_module_idx < 0 or target_module_idx >= len(all_modules):
        return

    target_module = all_modules[target_module_idx]
    
    with torch.no_grad():
        nn.init.kaiming_normal_(target_module.weight.data[neuron_indices, :], mode='fan_out', nonlinearity='relu')
        if target_module.bias is not None:
            nn.init.constant_(target_module.bias.data[neuron_indices], 0.0)
        if target_module_idx + 1 < len(all_modules):
            next_module = all_modules[target_module_idx + 1]
            if next_module.weight.dim() > 1:
                 nn.init.kaiming_normal_(next_module.weight.data[:, neuron_indices], mode='fan_in', nonlinearity='relu')

def run_synapse_optimization(model: Any, context_detector: Any, args: Namespace, episode_index: int, train_episode: TCLExperience) -> dict:
    if episode_index < args.synapse_activation_task_count:
        return {}

    print(f"\n--- SYNAPSE Self-Optimization Phase starting after episode {episode_index} ---")
    synapse_metrics = {'synapse/pruned_count': 0, 'synapse/shared_count': 0}

    # 1. 全体の可塑性をチェック
    total_neurons, immature_neurons = 0, 0
    # unit_ranksは (ranks, name) のタプルのリストなので、ranksを取り出す
    for ranks, _ in model.unit_ranks[1:-1]:
        total_neurons += len(ranks)
        immature_neurons += sum(1 for r in ranks if not r)
    
    current_immature_ratio = immature_neurons / total_neurons if total_neurons > 0 else 0
    print(f"  Overall immature neuron ratio: {current_immature_ratio:.2%} (Target: {args.target_immature_pool_ratio:.2%})")
    if current_immature_ratio >= args.target_immature_pool_ratio:
        print("  Sufficient overall plasticity. No optimization needed.")
        return synapse_metrics

    # 2. 活性化データを収集
    print("  Collecting activations for similarity analysis...")
    model.eval()
    activations_for_cka = {}
    representative_samples = get_n_samples_per_class(train_episode, n=10)

    with torch.no_grad():
        for samples, _ in representative_samples:
            samples = samples.to(get_device())
            _, layer_activations = model.get_activations(samples, return_output=True)
            for layer_idx, activation in enumerate(layer_activations):
                if layer_idx not in activations_for_cka: activations_for_cka[layer_idx] = []
                activations_for_cka[layer_idx].append(activation)
    for layer_idx in activations_for_cka:
        activations_for_cka[layer_idx] = torch.cat(activations_for_cka[layer_idx], dim=0)
    model.train()

    # 3. 類似度を計算し、候補ペアをリストアップ
    candidate_pairs = []
    PRUNING_THRESHOLD = 0.90
    SHARING_THRESHOLD = 0.85
    
    for layer_idx, (ranks, layer_name) in enumerate(model.unit_ranks):
        if not ranks or layer_idx not in activations_for_cka: continue

        cohorts = {task_id: [] for task_id in range(1, episode_index + 1)}
        for neuron_idx, task_list in enumerate(ranks):
            for task_id in task_list:
                if task_id in cohorts: cohorts[task_id].append(neuron_idx)
        cohorts = {k: v for k, v in cohorts.items() if v}

        if len(cohorts) < 1: continue
        activation_tensor = activations_for_cka.get(layer_idx)
        if activation_tensor is None: continue

        act_reshaped = activation_tensor.permute(0, 2, 3, 1).reshape(-1, activation_tensor.shape[1]) if activation_tensor.dim() == 4 else activation_tensor

        for task_id, neurons in cohorts.items():
            if len(neurons) < 2: continue
            for n1_idx, n2_idx in combinations(neurons, 2):
                similarity = centered_kernel_alignment(act_reshaped[:, n1_idx].unsqueeze(1), act_reshaped[:, n2_idx].unsqueeze(1))
                if similarity > PRUNING_THRESHOLD:
                    candidate_pairs.append({'similarity': similarity, 'layer_idx': layer_idx, 'neurons': [n1_idx, n2_idx], 'type': 'pruning'})

        if len(cohorts) < 2: continue
        for (task1, neurons1), (task2, neurons2) in combinations(cohorts.items(), 2):
            for n1_idx, n2_idx in product(neurons1, neurons2):
                similarity = centered_kernel_alignment(act_reshaped[:, n1_idx].unsqueeze(1), act_reshaped[:, n2_idx].unsqueeze(1))
                if similarity > SHARING_THRESHOLD:
                    candidate_pairs.append({'similarity': similarity, 'layer_idx': layer_idx, 'neurons': [n1_idx, n2_idx], 'type': 'sharing'})
    
    candidate_pairs.sort(key=lambda x: x['similarity'], reverse=True)

    # 4. 可塑性が目標に達するまで、類似度の高いペアから順にリサイクルを実行
    recycled_neurons_this_phase = set()
    neurons_to_reinit = {}

    for pair in candidate_pairs:
        if any((pair['layer_idx'], n) in recycled_neurons_this_phase for n in pair['neurons']):
            continue

        l_idx, n_indices = pair['layer_idx'], pair['neurons']
        
        if pair['type'] == 'pruning':
            target_neuron_idx = max(n_indices)
            model.unit_ranks[l_idx][0][target_neuron_idx] = []
            recycled_neurons_this_phase.add((l_idx, target_neuron_idx))
            if l_idx not in neurons_to_reinit: neurons_to_reinit[l_idx] = []
            neurons_to_reinit[l_idx].append(target_neuron_idx)
            synapse_metrics['synapse/pruned_count'] += 1
            print(f"  [Pruning] Layer {l_idx}, Neuron {target_neuron_idx} recycled. (Similarity: {pair['similarity']:.4f})")
            
        elif pair['type'] == 'sharing':
            n1_idx, n2_idx = n_indices
            task1_list = model.unit_ranks[l_idx][0][n1_idx]
            task2_list = model.unit_ranks[l_idx][0][n2_idx]
            
            if not task1_list or not task2_list: continue

            source_idx, target_idx = (n1_idx, n2_idx) if max(task1_list) < max(task2_list) else (n2_idx, n1_idx)
            
            # =================================================================
            # === 修正箇所: 参照に頼らず、新しいリストを作成して再代入する ===
            # =================================================================
            source_tasks = model.unit_ranks[l_idx][0][source_idx]
            # ターゲットのタスクリストをコピーして、新しいリストを作成
            new_target_task_list = model.unit_ranks[l_idx][0][target_idx][:]
            
            for task_id in source_tasks:
                if task_id not in new_target_task_list:
                    new_target_task_list.append(task_id)

            # 新しく作成したリストで、元のデータを上書きする
            model.unit_ranks[l_idx][0][target_idx] = sorted(new_target_task_list)
            # =================================================================

            model.unit_ranks[l_idx][0][source_idx] = []
            recycled_neurons_this_phase.add((l_idx, source_idx))
            if l_idx not in neurons_to_reinit: neurons_to_reinit[l_idx] = []
            neurons_to_reinit[l_idx].append(source_idx)
            synapse_metrics['synapse/shared_count'] += 1
            print(f"  [Sharing] Layer {l_idx}, Neuron {source_idx} merged into {target_idx} and recycled. (Similarity: {pair['similarity']:.4f})")

    # 5. リサイクル対象のニューロンの重みを一括で再初期化
    if neurons_to_reinit:
        print("  Reinitializing recycled neurons...")
        for layer_idx, indices in neurons_to_reinit.items():
            reinitialize_neurons(model, layer_idx, list(set(indices)))
            print(f"    Layer {layer_idx}: Reinitialized {len(set(indices))} neurons.")

    print("\n--- SYNAPSE Phase finished. ---\n")
    return synapse_metrics