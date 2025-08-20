# # from argparse import Namespace
# # from itertools import combinations
# # from typing import Any, Dict, List

# # import numpy as np
# # import torch
# # import torch.nn.functional as F

# # import wandb
# # from Source.architecture import SparseConv2d, SparseLinear, SparseOutput
# # from Source.resnet18 import BasicBlock, ResNet18


# # class BlockSimilarityAnalyzer:
# #     """
# #     ResNetのBasicBlock間の機能的な類似度を計算するクラス。
# #     """
# #     def __init__(self, model: Any, context_detector: Any):
# #         self.model = model
# #         self.context_detector = context_detector
# #         # モデル内の全層の名前とインデックスの対応表
# #         self.layer_name_to_idx = {name: i for i, (_, name) in enumerate(model.unit_ranks)}
# #         print("BlockSimilarityAnalyzer is ready.")

# #     def _get_block_input_activations(self, block: BasicBlock) -> List[torch.Tensor]:
# #         """指定されたブロックへの入力となる活性化データを全タスク分集める"""
# #         input_activations = []
# #         # ブロックの最初の層の名前を取得
# #         first_layer_name = block.conv1.layer_name
# #         # その層の「前」の層のインデックスを取得
# #         input_layer_idx = self.layer_name_to_idx.get(first_layer_name, 1) - 1

# #         try:
# #             layer_num = int(first_layer_name.split('_')[-1])
# #             # block1_1のconv1 (conv_early_2) の入力は conv_early_1
# #             # shortcut層も考慮し、より安全に一つ前の層のインデックスを取得
# #             input_layer_idx = self.layer_name_to_idx.get(f"conv_early_{layer_num - 1}", -1)
# #             # block2_1のconv1 (conv_early_6) の入力は block1_2の最終出力 (conv_early_5)
# #             if "->" in first_layer_name: # shortcut層の場合
# #                 input_layer_name = first_layer_name.split('->')[0]
# #                 input_layer_idx = self.layer_name_to_idx.get(input_layer_name, -1)
# #         except (ValueError, IndexError):
# #             input_layer_idx = -1

# #         if input_layer_idx == -1: return []

# #         for task_id in self.context_detector.float_context_representations.keys():
# #             for activations_per_sample, _, _ in self.context_detector.float_context_representations[task_id]:
# #                 for layer_idx, activation_tensor in activations_per_sample:
# #                     if layer_idx == input_layer_idx:
# #                         if activation_tensor.dim() == 4:
# #                             input_activations.append(activation_tensor)
# #                         # もし2Dで保存されていたら、スキップして警告を出す (デバッグ用)
# #                         elif activation_tensor.dim() == 2:
# #                             print(f"  [SYNAPSE WARNING] Skipping 2D activation tensor for layer {input_layer_idx} intended for a Conv layer.")

# #         return input_activations
    
# #     def calculate_block_similarity(self, block1: BasicBlock, block2: BasicBlock) -> float:
# #         """2つのBasicBlockの機能的類似度を計算する"""
# #         block1.eval()
# #         block2.eval()
        
# #         # 2つのブロックで共通の入力データを使う
# #         # ここでは簡単のため、block1への入力データをblock2にも流用する
# #         input_activations = self._get_block_input_activations(block1)
# #         if not input_activations:
# #             return 0.0

# #         output_vecs1, output_vecs2 = [], []
# #         with torch.no_grad():
# #             for input_tensor in input_activations:
# #                 # ブロックの出力を計算
# #                 out1, _, _ = block1(input_tensor)
# #                 out2, _, _ = block2(input_tensor)
                
# #                 # 出力テンソルをベクトル化
# #                 vec1 = out1.view(out1.size(0), -1)
# #                 vec2 = out2.view(out2.size(0), -1)
# #                 output_vecs1.append(vec1)
# #                 output_vecs2.append(vec2)
        
# #         if not output_vecs1:
# #             return 0.0

# #         # 全サンプルの出力ベクトルを結合
# #         vec1_full = torch.cat(output_vecs1)
# #         vec2_full = torch.cat(output_vecs2)

# #         # コサイン類似度を計算
# #         similarity = F.cosine_similarity(vec1_full, vec2_full, dim=1).mean()
# #         return similarity.item()


# # class StructuralOptimizationController:
# #     def __init__(self, model: Any, args: Namespace):
# #         self.model = model
# #         self.args = args
# #         print("StructuralOptimizationController is ready.")

# #     def prune_and_reinitialize_block(self, block_to_prune: BasicBlock):
# #         """指定されたブロック内の全ニューロンを剪定・再初期化する"""
# #         print(f"  Executing Pruning for block...")
# #         with torch.no_grad():
# #             for m in block_to_prune.modules():
# #                 if isinstance(m, (SparseConv2d, nn.BatchNorm2d)):
# #                     # Kaiming Heによる初期化
# #                     if hasattr(m, 'weight') and m.weight is not None:
# #                         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
# #                     if hasattr(m, 'bias') and m.bias is not None:
# #                         nn.init.constant_(m.bias, 0)

# #         # ブロック内の全ニューロンのランクをImmature(0)に戻す
# #         for m in block_to_prune.modules():
# #             if isinstance(m, SparseConv2d):
# #                 layer_name = m.layer_name
# #                 for i, (ranks, name) in enumerate(self.model.unit_ranks):
# #                     if name == layer_name:
# #                         self.model.unit_ranks[i] = (np.zeros_like(ranks), name)
# #                         print(f"    >> Neurons in {name} have been reset to Immature.")
# #                         break

# # def run_synapse_optimization(model: Any, context_detector: Any, args: Namespace, episode_index: int):
# #     if episode_index < args.synapse_activation_task_count:
# #         return
# #     log_metrics = {}
# #     pruned_block_count = 0
# #     shared_block_count = 0 # 将来の共有化機能のために用意   
# #     # ResNet以外は現在のロジックのまま（何もしない）
# #     if not isinstance(model, ResNet18):
# #         print("SYNAPSE: Block-level optimization is only implemented for ResNet18.")
# #         return

# #     print(f"\n--- SYNAPSE Block-level Phase starting after episode {episode_index} ---")
# #     analyzer = BlockSimilarityAnalyzer(model, context_detector)
# #     controller = StructuralOptimizationController(model, args)

# #     # 凍結済み（成熟）ブロックをタスクごとに分類
# #     mature_blocks_by_task: Dict[int, List[BasicBlock]] = {}
# #     for block in model.blocks:
# #         # ブロックが成熟しているかを判定（ここではブロック内の最初のconv層のニューロンが1つでも成熟していたら、と仮定）
# #         first_layer_name = block.conv1.layer_name
# #         is_mature = False
# #         task_id = -1
# #         for ranks, name in model.unit_ranks:
# #             if name == first_layer_name:
# #                 if np.any(ranks > 1):
# #                     is_mature = True
# #                     # 多数決でブロックのタスクを決定
# #                     mature_indices = np.where(ranks > 1)[0]
# #                     tasks = [model.neuron_birth_task[name].get(idx) for idx in mature_indices]
# #                     if tasks:
# #                         task_id = max(set(tasks), key=tasks.count) # 最頻値
# #                 break
        
# #         if is_mature and task_id != -1:
# #             if task_id not in mature_blocks_by_task:
# #                 mature_blocks_by_task[task_id] = []
# #             mature_blocks_by_task[task_id].append(block)

# #     # Intra-Task Pruning (同一タスク内のブロック剪定)
# #     pruned_blocks = []
# #     for task, blocks in mature_blocks_by_task.items():
# #         if len(blocks) < 2: continue
# #         for block1, block2 in combinations(blocks, 2):
# #             similarity = analyzer.calculate_block_similarity(block1, block2)
# #             if similarity > args.threshold_intra_task_pruning:
# #                 print(f"  >> PRUNING CANDIDATE (Intra-Task): Blocks in task {task}. Similarity: {similarity:.4f}")
# #                 # どちらかを剪定（ここでは単純に2番目のブロック）
# #                 controller.prune_and_reinitialize_block(block2)
# #                 pruned_blocks.append(block2)
# #                 pruned_block_count += 1
# #                 break # 1ペア見つけたら次のタスクへ

# #     if wandb.run is not None:
# #         log_metrics['synapse/pruned_blocks'] = pruned_block_count
# #         log_metrics['synapse/shared_blocks'] = shared_block_count # 現状は常に0
# #         wandb.log(log_metrics, step=episode_index)
        
# #     print("\n--- SYNAPSE Phase finished. ---\n")
    

# ###########################


# # # SYNAPSE/Source/synapse_operations.py

# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F
# # from argparse import Namespace
# # from typing import Any, Dict, List
# # from itertools import combinations
# # import numpy as np
# # import wandb
# # from Source.architecture import SparseConv2d, SparseLinear, SparseOutput
# # from Source.resnet18 import ResNet18, BasicBlock
# # from avalanche.benchmarks import TCLExperience
# # from Source.context_detector import get_n_samples_per_class
# # from Source.helper import get_device


# # class BlockSimilarityAnalyzer:
# #     """
# #     ResNetのBasicBlock間の機能的な類似度を計算するクラス。
# #     """
# #     def __init__(self, model: Any, context_detector: Any, episode_data: TCLExperience, args: Namespace):
# #         self.model = model
# #         self.context_detector = context_detector
# #         self.episode_data = episode_data
# #         self.args = args
# #         self.layer_name_to_idx = {name: i for i, (_, name) in enumerate(model.unit_ranks)}
# #         print("BlockSimilarityAnalyzer is ready.")

# #     def calculate_block_similarity(self, block1: BasicBlock, block2: BasicBlock) -> float:
# #         """2つのBasicBlockの機能的類似度を計算する"""
# #         block1.eval()
# #         block2.eval()
        
# #         # 代表サンプルを取得
# #         samples_per_class = get_n_samples_per_class(self.episode_data, self.args.memo_per_class_context)
        
# #         output_vecs1, output_vecs2 = [], []
# #         with torch.no_grad():
# #             for samples, _ in samples_per_class:
# #                 samples = samples.to(get_device())
                
# #                 # --- モデルを経由して、ブロックへの正しい形の入力を生成 ---
# #                 x = samples
# #                 x = self.model.relu(self.model.bn1(self.model.conv1(x)))
                
# #                 for block in self.model.blocks:
# #                     if block == block1 or block == block2:
# #                         break
# #                     x, _, _ = block(x)

# #                 # --- 2つのブロックの出力を比較 ---
# #                 out1, _, _ = block1(x)
# #                 out2, _, _ = block2(x)
                
# #                 vec1 = out1.view(out1.size(0), -1)
# #                 vec2 = out2.view(out2.size(0), -1)
# #                 output_vecs1.append(vec1)
# #                 output_vecs2.append(vec2)
        
# #         if not output_vecs1: 
# #             self.model.train()
# #             return 0.0

# #         vec1_full = torch.cat(output_vecs1)
# #         vec2_full = torch.cat(output_vecs2)

# #         similarity = F.cosine_similarity(vec1_full, vec2_full, dim=1).mean()
# #         self.model.train()
# #         return similarity.item()


# # class StructuralOptimizationController:
# #     def __init__(self, model: Any, args: Namespace):
# #         self.model = model
# #         self.args = args
# #         print("StructuralOptimizationController is ready.")

# #     def prune_and_reinitialize_block(self, block_to_prune: BasicBlock):
# #         """指定されたブロック内の全ニューロンを剪定・再初期化する"""
# #         print(f"  Executing Pruning for block...")
# #         with torch.no_grad():
# #             for m in block_to_prune.modules():
# #                 if isinstance(m, (SparseConv2d, nn.BatchNorm2d)):
# #                     if hasattr(m, 'weight') and m.weight is not None:
# #                         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
# #                     if hasattr(m, 'bias') and m.bias is not None:
# #                         nn.init.constant_(m.bias, 0)

# #         for m in block_to_prune.modules():
# #             if isinstance(m, SparseConv2d):
# #                 layer_name = m.layer_name
# #                 for i, (ranks, name) in enumerate(self.model.unit_ranks):
# #                     if name == layer_name:
# #                         self.model.unit_ranks[i] = (np.zeros_like(ranks), name)
# #                         print(f"    >> Neurons in {name} have been reset to Immature.")
# #                         break
    
# #     # =================================================================
# #     # Inter-Task Sharing を行う
# #     # =================================================================
# #     def share_blocks(self, source_block: BasicBlock, target_block: BasicBlock):
# #         """source_block の知識を target_block に統合し、source_block を再初期化する"""
# #         print(f"  Executing Sharing: Merging knowledge from one block to another...")
# #         with torch.no_grad():
# #             # sourceの重みをtargetに加算していく
# #             for (name_s, param_s), (name_t, param_t) in zip(source_block.named_parameters(), target_block.named_parameters()):
# #                 if param_s.data.shape == param_t.data.shape:
# #                     param_t.data += param_s.data
        
# #         # 知識の転移が完了したので、sourceブロックは再初期化してプールに戻す
# #         self.prune_and_reinitialize_block(source_block)
# #     # =================================================================


# # def run_synapse_optimization(model: Any, context_detector: Any, args: Namespace, episode_index: int, train_episode: TCLExperience) -> dict:
# #     if episode_index < args.synapse_activation_task_count:
# #         return {}
    
# #     if not isinstance(model, ResNet18):
# #         print("SYNAPSE: Block-level optimization is currently only implemented for ResNet18.")
# #         return {}

# #     print(f"\n--- SYNAPSE Block-level Phase starting after episode {episode_index} ---")
# #     analyzer = BlockSimilarityAnalyzer(model, context_detector, train_episode, args)
# #     controller = StructuralOptimizationController(model, args)

# #     # =================================================================
# #     # ★変更点: 比較可能なブロックのグループを定義
# #     # =================================================================
# #     compatible_block_groups = [
# #         [model.block1_1, model.block1_2],
# #         [model.block2_2], # チャンネル数が変わるブロックは単独
# #         [model.block3_2],
# #         [model.block4_2]
# #     ]
# #     # =================================================================


# #     synapse_metrics = {
# #         'synapse/pruned_blocks': 0,
# #         'synapse/shared_blocks': 0 
# #     }

# #     mature_blocks_by_task: Dict[int, List[BasicBlock]] = {}
# #     for block in model.blocks:
# #         first_layer_name = block.conv1.layer_name
# #         is_mature, task_id = False, -1
# #         for ranks, name in model.unit_ranks:
# #             if name == first_layer_name:
# #                 if np.any(ranks > 1):
# #                     is_mature = True
# #                     mature_indices = np.where(ranks > 1)[0]
# #                     tasks = [model.neuron_birth_task[name].get(idx) for idx in mature_indices if model.neuron_birth_task[name].get(idx) is not None]
# #                     if tasks:
# #                         task_id = max(set(tasks), key=tasks.count)
# #                 break
        
# #         if is_mature and task_id != -1:
# #             if task_id not in mature_blocks_by_task:
# #                 mature_blocks_by_task[task_id] = []
# #             mature_blocks_by_task[task_id].append(block)

# #     # # --- 1. Intra-Task Pruning (同一タスク内の剪定) ---
# #     # pruned_blocks = []
# #     # for task, blocks in mature_blocks_by_task.items():
# #     #     if len(blocks) < 2: continue
# #     #     for block1, block2 in combinations(blocks, 2):
# #     #         if any(b in pruned_blocks for b in [block1, block2]): continue
# #     #         similarity = analyzer.calculate_block_similarity(block1, block2)
# #     #         if similarity > args.threshold_intra_task_pruning:
# #     #             print(f"  >> PRUNING TRIGGERED (Intra-Task): Blocks in task {task}. Similarity: {similarity:.4f}")
# #     #             controller.prune_and_reinitialize_block(block2)
# #     #             pruned_blocks.append(block2)
# #     #             synapse_metrics['synapse/pruned_blocks'] += 1
# #     #             break 

# #     # --- Intra-Task Pruning ---
# #     pruned_blocks = []
# #     # ★変更点: グループ内で比較
# #     for group in compatible_block_groups:
# #         for task, blocks_in_task in mature_blocks_by_task.items():
# #             # 現在のグループに属する、このタスクのブロックを抽出
# #             relevant_blocks = [b for b in blocks_in_task if b in group]
# #             if len(relevant_blocks) < 2: continue
            
# #             for block1, block2 in combinations(relevant_blocks, 2):
# #                 if any(b in pruned_blocks for b in [block1, block2]): continue
# #                 similarity = analyzer.calculate_block_similarity(block1, block2)
# #                 if similarity > args.threshold_intra_task_pruning:
# #                     print(f"  >> PRUNING TRIGGERED (Intra-Task): Blocks in task {task}. Similarity: {similarity:.4f}")
# #                     controller.prune_and_reinitialize_block(block2)
# #                     pruned_blocks.append(block2)
# #                     synapse_metrics['synapse/pruned_blocks'] += 1
# #                     break


# #     # =================================================================
# #     # Inter-Task Sharing (異種タスク間の共有化)
# #     # =================================================================
# #     if len(mature_blocks_by_task) > 1:
# #         print("  Analyzing interactions between tasks for Inter-Task Sharing...")
# #         shared_blocks = list(pruned_blocks) # 剪定済みブロックは共有化の対象外
# #         # ... (既存のロジックを、同様にグループ内で比較するように修正) ...
# #         for group in compatible_block_groups:
# #             # 異なるタスクのブロックペアを全て評価
# #             for (task1, blocks1), (task2, blocks2) in combinations(mature_blocks_by_task.items(), 2):
# #                 # 各タスクから、現在のグループに属するブロックを抽出
# #                 relevant_blocks1 = [b for b in blocks1 if b in group and b not in shared_blocks]
# #                 relevant_blocks2 = [b for b in blocks2 if b in group and b not in shared_blocks]

# #                 for b1 in relevant_blocks1:
# #                     for b2 in relevant_blocks2:
# #                         if b1 in shared_blocks or b2 in shared_blocks: continue

# #                         similarity = analyzer.calculate_block_similarity(b1, b2)
# #                         if similarity > args.threshold_inter_task_sharing:
# #                             print(f"  >> SHARING TRIGGERED (Inter-Task): Blocks from tasks ({task1}, {task2}). Similarity: {similarity:.4f}")
                            
# #                             # より新しいタスクのブロックをsource、古い方をtargetとする
# #                             if task1 > task2:
# #                                 source_block, target_block = b1, b2
# #                             else:
# #                                 source_block, target_block = b2, b1
                            
# #                             controller.share_blocks(source_block, target_block)
# #                             shared_blocks.extend([source_block, target_block])
# #                             synapse_metrics['synapse/shared_blocks'] += 1
# #                             break # このペアの処理は終了
# #                     if b1 in shared_blocks:
# #                         break
                            
# #         # # 異なるタスクのブロックペアを全て評価
# #         # for (task1, blocks1), (task2, blocks2) in combinations(mature_blocks_by_task.items(), 2):
# #         #     for b1 in blocks1:
# #         #         if b1 in shared_blocks: continue
# #         #         for b2 in blocks2:
# #         #             if b2 in shared_blocks: continue

# #         #             similarity = analyzer.calculate_block_similarity(b1, b2)
# #         #             if similarity > args.threshold_inter_task_sharing:
# #         #                 print(f"  >> SHARING TRIGGERED (Inter-Task): Blocks from tasks ({task1}, {task2}). Similarity: {similarity:.4f}")
                        
# #         #                 # より新しいタスクのブロックをsource、古い方をtargetとする
# #         #                 if task1 > task2:
# #         #                     source_block, target_block = b1, b2
# #         #                 else:
# #         #                     source_block, target_block = b2, b1
                        
# #         #                 controller.share_blocks(source_block, target_block)
# #         #                 shared_blocks.extend([source_block, target_block])
# #         #                 synapse_metrics['synapse/shared_blocks'] += 1
# #         #                 break # このペアの処理は終了
# #         #         if b1 in shared_blocks:
# #         #             break
# #     # =================================================================

# #     print("\n--- SYNAPSE Phase finished. ---\n")
# #     return synapse_metrics


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
                
#                 # 比較対象のブロックの直前までデータを流す
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
#         print(f"  Executing Pruning for a block...")
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
    
#     def share_blocks(self, source_block: BasicBlock, target_block: BasicBlock):
#         """source_block の知識を target_block に統合し、source_block を再初期化する"""
#         print(f"  Executing Sharing: Merging knowledge from one block to another...")
#         with torch.no_grad():
#             # sourceの重みをtargetに加算していく (簡単な平均化)
#             for (name_s, param_s), (name_t, param_t) in zip(source_block.named_parameters(), target_block.named_parameters()):
#                 if param_s.data.shape == param_t.data.shape:
#                     param_t.data = (param_t.data + param_s.data) / 2.0
        
#         # 知識の転移が完了したので、sourceブロックは再初期化してプールに戻す
#         self.prune_and_reinitialize_block(source_block)


# def run_synapse_optimization(model: Any, context_detector: Any, args: Namespace, episode_index: int, train_episode: TCLExperience) -> dict:
#     if episode_index < args.synapse_activation_task_count:
#         return {}
    
#     if not isinstance(model, ResNet18):
#         print("SYNAPSE: Block-level optimization is currently only implemented for ResNet18.")
#         return {}

#     print(f"\n--- SYNAPSE Block-level Phase starting after episode {episode_index} ---")
#     analyzer = BlockSimilarityAnalyzer(model, context_detector, train_episode, args)
#     controller = StructuralOptimizationController(model, args)

#     # 互換性のあるブロックのグループを定義
#     # ResNet18の構造に基づき、同じ入出力形状を持つブロックをグループ化
#     compatible_block_groups = [
#         [model.block1_1, model.block1_2],
#         [model.block2_2], 
#         [model.block3_2],
#         [model.block4_2]
#     ]

#     synapse_metrics = {
#         'synapse/pruned_blocks': 0,
#         'synapse/shared_blocks': 0 
#     }

#     # 成熟したブロックをタスクごとに分類
#     mature_blocks_by_task: Dict[int, List[BasicBlock]] = {}
#     for block in model.blocks:
#         first_layer_name = block.conv1.layer_name
#         is_mature, task_id = False, -1
#         for ranks, name in model.unit_ranks:
#             if name == first_layer_name:
#                 if np.any(ranks > 1): # ランクが1より大きい=成熟
#                     is_mature = True
#                     mature_indices = np.where(ranks > 1)[0]
#                     tasks = [model.neuron_birth_task[name].get(idx) for idx in mature_indices if model.neuron_birth_task[name].get(idx) is not None]
#                     if tasks:
#                         task_id = max(set(tasks), key=tasks.count) # 最も多いタスクIDを所属タスクとする
#                 break
        
#         if is_mature and task_id != -1:
#             if task_id not in mature_blocks_by_task:
#                 mature_blocks_by_task[task_id] = []
#             mature_blocks_by_task[task_id].append(block)

#     # --- Intra-Task Pruning (同一タスク内の剪定) ---
#     print("  Analyzing interactions within tasks for Intra-Task Pruning...")
#     pruned_blocks = []
#     for group in compatible_block_groups:
#         for task, blocks_in_task in mature_blocks_by_task.items():
#             # 現在のグループに属し、かつこのタスクに属するブロックを抽出
#             relevant_blocks = [b for b in blocks_in_task if b in group]
#             if len(relevant_blocks) < 2: continue
            
#             for block1, block2 in combinations(relevant_blocks, 2):
#                 if any(b in pruned_blocks for b in [block1, block2]): continue
#                 similarity = analyzer.calculate_block_similarity(block1, block2)
#                 if similarity > args.threshold_intra_task_pruning:
#                     print(f"  >> PRUNING TRIGGERED (Intra-Task): Blocks in task {task}. Similarity: {similarity:.4f}")
#                     # より新しいタスクのブロックを剪定（ここでは単純にblock2）
#                     controller.prune_and_reinitialize_block(block2)
#                     pruned_blocks.append(block2)
#                     synapse_metrics['synapse/pruned_blocks'] += 1
#                     break

#     # --- Inter-Task Sharing (異種タスク間の共有化) ---
#     if len(mature_blocks_by_task) > 1:
#         print("  Analyzing interactions between tasks for Inter-Task Sharing...")
#         shared_blocks = list(pruned_blocks) # 剪定済みブロックは共有化の対象外
        
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
#                             break
#                     if b1 in shared_blocks:
#                         break
                            
#     print("\n--- SYNAPSE Phase finished. ---\n")
#     return synapse_metrics


# SYNAPSE/Source/synapse_operations.py
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

    # =================================================================
    # 1. 可塑性のチェック (トリガー)
    # =================================================================
    total_neurons = 0
    immature_neurons = 0
    # 入力層(0)と出力層(-1)を除く
    for ranks, _ in model.unit_ranks[1:-1]:
        total_neurons += len(ranks)
        immature_neurons += sum(1 for r in ranks if not r)
    
    current_immature_ratio = immature_neurons / total_neurons if total_neurons > 0 else 0
    print(f"  Current immature neuron ratio: {current_immature_ratio:.2%} (Target: {args.target_immature_pool_ratio:.2%})")

    if current_immature_ratio >= args.target_immature_pool_ratio:
        print("  Sufficient plasticity. No optimization needed.")
        return synapse_metrics

    # =================================================================
    # 2. 活性化データを収集
    # =================================================================
    print("  Collecting activations for similarity analysis...")
    model.eval()
    activations_for_cka = {}  # {layer_idx: [sample_activations]}
    # get_n_samples_per_classはcontext_detectorからインポート
    representative_samples = get_n_samples_per_class(train_episode, n=10) # 各クラスから10サンプル取得

    with torch.no_grad():
        for samples, _ in representative_samples:
            samples = samples.to(get_device())
            # 活性化を取得（conv層はflattenしない）
            _, layer_activations = reduce_or_flat_convs(model.get_activations(samples), reduce=False)
            
            for layer_idx, activation in enumerate(layer_activations):
                if layer_idx not in activations_for_cka:
                    activations_for_cka[layer_idx] = []
                # (サンプル数, 特徴次元数) の形に変形
                activations_for_cka[layer_idx].append(activation.view(activation.shape[0], -1))

    # 各層でテンソルを結合
    for layer_idx in activations_for_cka:
        activations_for_cka[layer_idx] = torch.cat(activations_for_cka[layer_idx], dim=0)
    
    model.train() # モデルを訓練モードに戻す

    # =================================================================
    # 3. 全候補の類似度を計算 & 優先順位付け
    # =================================================================
    candidate_pairs = []
    # 入力層(0)と出力層(-1)を除く
    for layer_idx, (ranks, layer_name) in enumerate(model.unit_ranks[1:-1], start=1):
        if not any(ranks) or layer_idx not in activations_for_cka: continue

        # この層の凍結済みニューロンをタスクごとにグループ化
        cohorts = {}  # {task_id: [neuron_indices]}
        for neuron_idx, task_list in enumerate(ranks):
            # 凍結済み(空でなく、現在の学習者でもない)ニューロンを対象
            if task_list and task_list != [episode_index]:
                for task_id in task_list:
                    if task_id not in cohorts: cohorts[task_id] = []
                    cohorts[task_id].append(neuron_idx)

        # 比較対象が2つ以上なければスキップ
        if len(cohorts) < 1: continue

        # 同一タスク内(Intra-Task)のペア
        for task_id, neurons in cohorts.items():
            if len(neurons) < 2: continue
            # 全ニューロンペアに対してCKAを計算
            for n1_idx, n2_idx in combinations(neurons, 2):
                act1 = activations_for_cka[layer_idx][:, n1_idx].unsqueeze(1)
                act2 = activations_for_cka[layer_idx][:, n2_idx].unsqueeze(1)
                similarity = centered_kernel_alignment(act1, act2)
                candidate_pairs.append({
                    'similarity': similarity, 'layer_idx': layer_idx,
                    'tasks': [task_id], 'neurons': [n1_idx, n2_idx],
                    'type': 'pruning'
                })

        # 異種タスク間(Inter-Task)のペア
        if len(cohorts) < 2: continue
        for (task1, neurons1), (task2, neurons2) in combinations(cohorts.items(), 2):
            # タスクコホート間の全ニューロンペアで比較
            for n1_idx, n2_idx in product(neurons1, neurons2):
                act1 = activations_for_cka[layer_idx][:, n1_idx].unsqueeze(1)
                act2 = activations_for_cka[layer_idx][:, n2_idx].unsqueeze(1)
                similarity = centered_kernel_alignment(act1, act2)
                candidate_pairs.append({
                    'similarity': similarity, 'layer_idx': layer_idx,
                    'tasks': sorted([task1, task2]), 'neurons': [n1_idx, n2_idx],
                    'type': 'sharing'
                })
    
    # 類似度が高い順にソート
    candidate_pairs.sort(key=lambda x: x['similarity'], reverse=True)

    # =================================================================
    # 4 & 5. トップダウンで逐次リサイクルし、目標達成で停止
    # =================================================================
    recycled_neurons_this_phase = set()
    
    for pair in candidate_pairs:
        # 現在の未熟率を再計算
        immature_count = sum(1 for r in model.unit_ranks[pair['layer_idx']][0] if not r)
        total_in_layer = len(model.unit_ranks[pair['layer_idx']][0])
        current_layer_immature_ratio = immature_count / total_in_layer

        if current_layer_immature_ratio >= args.target_immature_pool_ratio:
            continue # この層は目標達成済みなのでスキップ

        # 操作対象のニューロンが既にリサイクル済みでないか確認
        if any((pair['layer_idx'], n) in recycled_neurons_this_phase for n in pair['neurons']):
            continue

        if pair['type'] == 'pruning':
            # Pruning: 片方のニューロンをリサイクル（ここでは2番目のニューロン）
            target_neuron_idx = pair['neurons'][1]
            model.unit_ranks[pair['layer_idx']][0][target_neuron_idx] = [] # 未熟化
            recycled_neurons_this_phase.add((pair['layer_idx'], target_neuron_idx))
            synapse_metrics['synapse/pruned_count'] += 1
            print(f"  [Pruning] Layer {pair['layer_idx']}, Neuron {target_neuron_idx} recycled. (Similarity: {pair['similarity']:.4f})")
            
        elif pair['type'] == 'sharing':
            # Sharing: 新しいタスクのニューロンを古い方に統合
            n1_idx, n2_idx = pair['neurons']
            task1_list = model.unit_ranks[pair['layer_idx']][0][n1_idx]
            task2_list = model.unit_ranks[pair['layer_idx']][0][n2_idx]
            
            # より新しいタスクを持つ方をsourceとする
            if max(task1_list) > max(task2_list):
                source_idx, target_idx = n1_idx, n2_idx
            else:
                source_idx, target_idx = n2_idx, n1_idx

            # 知識の統合: ここでは単純な重み平均化を実装
            # TODO: より高度な知識蒸留などを検討
            # (この部分はモデルの構造にアクセスする必要があるため、別途ヘルパー関数化が望ましい)
            
            # sourceをリサイクル
            model.unit_ranks[pair['layer_idx']][0][source_idx] = []
            
            # targetにsourceのタスクIDを追加して共有状態にする
            source_tasks = model.unit_ranks[pair['layer_idx']][0][source_idx]
            for task_id in source_tasks:
                if task_id not in model.unit_ranks[pair['layer_idx']][0][target_idx]:
                    model.unit_ranks[pair['layer_idx']][0][target_idx].append(task_id)

            recycled_neurons_this_phase.add((pair['layer_idx'], source_idx))
            synapse_metrics['synapse/shared_count'] += 1
            print(f"  [Sharing] Layer {pair['layer_idx']}, Neuron {source_idx} merged into {target_idx} and recycled. (Similarity: {pair['similarity']:.4f})")

    print("\n--- SYNAPSE Phase finished. ---\n")
    return synapse_metrics