
# SYNAPSE/Source/synapse_operations.py

import torch
import numpy as np
from itertools import combinations, product
from typing import Any, Dict, List
import copy

from argparse import Namespace
from avalanche.benchmarks import TCLExperience
from Source.helper import get_device, reduce_or_flat_convs
from Source.context_detector import get_n_samples_per_class


# --- CKAの実装 ---
# この関数は新たに追加します
def centered_kernel_alignment(X, Y):
    """
    2つの活性化行列間の線形CKAを計算する。
    Args:
        X (torch.Tensor): 最初の活性化行列 (サンプル数 x ニューロン数)
        Y (torch.Tensor): 2番目の活性化行列 (サンプル数 x ニューロン数)
    Returns:
        float: CKAスコア
    """
    # グラム行列を計算する前に平均を引く
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    # グラム行列 (K = X @ X.T, L = Y @ Y.T)
    # HSIC(K, L) = tr(KHLH) / ((tr(KHK))^1/2 * (tr(LHL))^1/2) とほぼ等価
    # Hはセンタリング行列だが、既に行ったので不要

    XTX = X.T @ X
    YTY = Y.T @ Y
    
    # 分子: HSIC(X, Y)
    numerator = torch.norm(Y.T @ X, p='fro')**2
    
    # 分母: sqrt(HSIC(X, X) * HSIC(Y, Y))
    denominator = torch.norm(XTX, p='fro') * torch.norm(YTY, p='fro')
    
    return (numerator / denominator).item() if denominator > 0 else 0.0


# --- メイン関数 ---
def run_synapse_optimization(model: Any, context_detector: Any, args: Namespace, episode_index: int, train_episode: TCLExperience) -> dict:
    if episode_index < args.synapse_activation_task_count:
        return {}

    print(f"\n--- SYNAPSE Self-Optimization Phase starting after episode {episode_index} ---")
    synapse_metrics = {'synapse/pruned_count': 0, 'synapse/shared_count': 0}

    # 1. 可塑性のチェック (トリガー)
    total_neurons = 0
    immature_neurons = 0
    # ### 修正箇所 1: unit_ranks のタプルを正しく展開 ###
    for ranks, _ in model.unit_ranks[1:-1]: # 入力層と出力層を除く
        total_neurons += len(ranks)
        immature_neurons += sum(1 for r in ranks if not r)
    
    current_immature_ratio = immature_neurons / total_neurons if total_neurons > 0 else 0
    print(f"  Current immature neuron ratio: {current_immature_ratio:.2%} (Target: {args.target_immature_pool_ratio:.2%})")

    if current_immature_ratio >= args.target_immature_pool_ratio:
        print("  Sufficient plasticity. No optimization needed.")
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
                if layer_idx not in activations_for_cka:
                    activations_for_cka[layer_idx] = []
                activations_for_cka[layer_idx].append(activation)

    for layer_idx in activations_for_cka:
        activations_for_cka[layer_idx] = torch.cat(activations_for_cka[layer_idx], dim=0)
    
    model.train()

    # 3. 全候補の類似度を計算 & 優先順位付け
    candidate_pairs = []
    # ### 修正箇所 2: unit_ranks のタプルを正しく展開 ###
    for layer_idx, (ranks, layer_name) in enumerate(model.unit_ranks):
        if not any(ranks) or layer_idx not in activations_for_cka: continue

        cohorts = {}
        for neuron_idx, task_list in enumerate(ranks):
            if task_list and task_list != [episode_index]:
                for task_id in task_list:
                    if task_id not in cohorts: cohorts[task_id] = []
                    cohorts[task_id].append(neuron_idx)

        if len(cohorts) < 1 : continue

        activation_tensor = activations_for_cka.get(layer_idx)
        if activation_tensor is None: continue

        # 活性化テンソルをCKA計算用に整形
        if activation_tensor.dim() == 4:
            act_reshaped = activation_tensor.permute(0, 2, 3, 1).reshape(-1, activation_tensor.shape[1])
        else:
            act_reshaped = activation_tensor
            
        # Intra-Task
        for task_id, neurons in cohorts.items():
            if len(neurons) < 2: continue
            for n1_idx, n2_idx in combinations(neurons, 2):
                act1 = act_reshaped[:, n1_idx].unsqueeze(1)
                act2 = act_reshaped[:, n2_idx].unsqueeze(1)
                similarity = centered_kernel_alignment(act1, act2)
                candidate_pairs.append({
                    'similarity': similarity, 'layer_idx': layer_idx,
                    'tasks': [task_id], 'neurons': [n1_idx, n2_idx],
                    'type': 'pruning'
                })

        # Inter-Task
        if len(cohorts) < 2: continue
        for (task1, neurons1), (task2, neurons2) in combinations(cohorts.items(), 2):
            for n1_idx, n2_idx in product(neurons1, neurons2):
                act1 = act_reshaped[:, n1_idx].unsqueeze(1)
                act2 = act_reshaped[:, n2_idx].unsqueeze(1)
                similarity = centered_kernel_alignment(act1, act2)
                candidate_pairs.append({
                    'similarity': similarity, 'layer_idx': layer_idx,
                    'tasks': sorted([task1, task2]), 'neurons': [n1_idx, n2_idx],
                    'type': 'sharing'
                })
    
    candidate_pairs.sort(key=lambda x: x['similarity'], reverse=True)

    # 4 & 5. トップダウンで逐次リサイクル
    recycled_neurons_this_phase = set()
    for pair in candidate_pairs:
        # ### 修正箇所 3: unit_ranks のタプルを正しく展開 ###
        ranks, _ = model.unit_ranks[pair['layer_idx']]
        immature_count = sum(1 for r in ranks if not r)
        total_in_layer = len(ranks)
        current_layer_immature_ratio = immature_count / total_in_layer

        if current_layer_immature_ratio >= args.target_immature_pool_ratio:
            continue

        if any((pair['layer_idx'], n) in recycled_neurons_this_phase for n in pair['neurons']):
            continue

        if pair['type'] == 'pruning':
            target_neuron_idx = pair['neurons'][1]
            model.unit_ranks[pair['layer_idx']][0][target_neuron_idx] = []
            recycled_neurons_this_phase.add((pair['layer_idx'], target_neuron_idx))
            synapse_metrics['synapse/pruned_count'] += 1
            print(f"  [Pruning] Layer {pair['layer_idx']}, Neuron {target_neuron_idx} recycled. (Similarity: {pair['similarity']:.4f})")
            
        elif pair['type'] == 'sharing':
            n1_idx, n2_idx = pair['neurons']
            task1_list = model.unit_ranks[pair['layer_idx']][0][n1_idx]
            task2_list = model.unit_ranks[pair['layer_idx']][0][n2_idx]
            
            if max(task1_list) > max(task2_list):
                source_idx, target_idx = n1_idx, n2_idx
            else:
                source_idx, target_idx = n2_idx, n1_idx
            
            source_tasks = model.unit_ranks[pair['layer_idx']][0][source_idx]
            for task_id in source_tasks:
                if task_id not in model.unit_ranks[pair['layer_idx']][0][target_idx]:
                    model.unit_ranks[pair['layer_idx']][0][target_idx].append(task_id)

            model.unit_ranks[pair['layer_idx']][0][source_idx] = []
            recycled_neurons_this_phase.add((pair['layer_idx'], source_idx))
            synapse_metrics['synapse/shared_count'] += 1
            print(f"  [Sharing] Layer {pair['layer_idx']}, Neuron {source_idx} merged into {target_idx} and recycled. (Similarity: {pair['similarity']:.4f})")

    print("\n--- SYNAPSE Phase finished. ---\n")
    return synapse_metrics
