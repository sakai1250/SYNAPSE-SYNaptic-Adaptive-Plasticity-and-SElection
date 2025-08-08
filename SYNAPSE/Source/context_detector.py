# # Source/context_detector.py

# from argparse import Namespace
# from typing import Any, Dict, List
# import torch
# import numpy as np
# from torch.utils.data import DataLoader, Subset

# from Source.helper import get_device, reduce_or_flat_convs
# # SYNAPSE: コサイン類似度を計算するためにF.normalizeをインポートします
# import torch.nn.functional as F

# # SYNAPSE: 状態定義をインポートします
# from .synapse_operations import MATURE_BASE_RANK


# class SimilarityAnalyzer():
#     """
#     SYNAPSEの頭脳。入力データの活性化パターンを分析し、
#     既知のクラスとの類似度を計算します。
#     """
#     def __init__(self, args: Namespace, penultimate_layer_size: int, task2classes: Dict):
#         self.args = args
#         # self.task2classes = task2classes ... (必要に応じて残す)

#         # SYNAPSE: クラスごとの「顔」（プロトタイプ）を保存する辞書
#         # {class_id: prototype_vector} の形式で保存します
#         self.class_prototypes = dict()

#         # SYNAPSE: どの層の活性化を分析に使うかを指定します
#         # ここでは例として最後から2番目の層 (-2) を使います
#         self.prototype_layer_idx = -2

#     def update_prototypes(self, network: Any, train_episode: Any, episode_index: int):
#         """
#         学習済みのデータから、クラスごとのプロトタイプ（平均活性化ベクトル）を計算・更新します。
#         この関数は、学習が完了しニューロンがMature化した後に呼び出されます。
#         """
#         print("クラスプロトタイプを更新します...")
#         network.eval()
        
#         # このエピソードで学習したクラスを取得
#         classes_in_this_episode = train_episode.classes_in_this_experience
        
#         with torch.no_grad():
#             for class_id in classes_in_this_episode:
#                 # このクラスに属するMatureニューロンのインデックスを取得
#                 # 全ての層でMatureニューロンを考慮するか、特定の層に限定するかは戦略によります
#                 mature_neurons_indices = network.mature_neurons[self.prototype_layer_idx]

#                 if len(mature_neurons_indices) == 0:
#                     # print(f"警告: クラス {class_id} に対応するMatureニューロンが層 {self.prototype_layer_idx} に見つかりません。")
#                     continue
                
#                 # クラスごとのデータを取得して、平均活性化を計算
#                 # NICEのget_n_samples_per_classを流用します
#                 samples, _ = get_n_samples_per_class(train_episode, n=50, target_class=class_id)
#                 if samples is None: continue

#                 # 活性化を取得
#                 _, activations = network.get_activations(samples.to(get_device()), return_output=True)
                
#                 # 指定した層の活性化ベクトルを取得
#                 target_activations = activations[self.prototype_layer_idx]
                
#                 # 活性化ベクトルをフラット化 (conv層の場合)
#                 if len(target_activations.shape) > 2:
#                     target_activations = target_activations.mean(dim=[2,3]) # Global Average Pooling
                
#                 # このクラスに対応するMatureニューロンの活性化部分だけを抽出
#                 class_specific_activations = target_activations[:, mature_neurons_indices]
                
#                 # 平均を取ってプロトタイプとする
#                 prototype = class_specific_activations.mean(dim=0)
                
#                 # 計算したプロトタイプを保存
#                 self.class_prototypes[class_id] = prototype
#                 print(f"クラス {class_id} のプロトタイプを更新しました。サイズ: {prototype.shape}")

#         network.train()
#         return

#     def calculate_similarity_score(self, network: Any, data_loader: DataLoader) -> float:
#         """
#         新しいデータに対して、既存の全クラスとの最大類似度スコアを計算します。

#         Returns:
#             類似度スコア (0から1の範囲)。既知クラスがなければ0を返す。
#         """
#         if not self.class_prototypes:
#             print("類似度計算スキップ: 既知のクラスプロトタイプがありません。")
#             return 0.0

#         network.eval()
#         # データローダーから1バッチだけ取得して分析
#         data, _, _ = next(iter(data_loader))
#         data = data.to(get_device())

#         max_similarity = 0.0

#         with torch.no_grad():
#             # 入力データの活性化を取得
#             _, activations = network.get_activations(data, return_output=True)
#             target_activations = activations[self.prototype_layer_idx]
#             if len(target_activations.shape) > 2:
#                 target_activations = target_activations.mean(dim=[2,3])

#             # 各プロトタイプと比較
#             for class_id, prototype in self.class_prototypes.items():
#                 mature_neurons_indices = network.mature_neurons[self.prototype_layer_idx]
                
#                 # 比較対象の活性化を抽出
#                 current_activations = target_activations[:, mature_neurons_indices]
                
#                 # コサイン類似度を計算 (バッチ内の平均を取る)
#                 # F.normalizeでベクトルを正規化してから内積を取る
#                 # prototypeも同様に正規化
#                 prototype_norm = F.normalize(prototype.unsqueeze(0), p=2, dim=1)
#                 current_activations_norm = F.normalize(current_activations, p=2, dim=1)
                
#                 similarity = (current_activations_norm @ prototype_norm.T).mean()
                
#                 if similarity > max_similarity:
#                     max_similarity = similarity.item()
        
#         network.train()
#         print(f"計算された最大類似度スコア: {max_similarity:.4f}")
#         return max_similarity


#     def tree_preds(self, activations) -> List[int]:
#         pos_probs = []
#         for index, model in enumerate(self.context_learners, 1):
#             context_masks = self.context_layers_masks[index]
#             X = self.process_and_stack(context_masks, activations)
#             preds = model.predict_proba(X.cpu().numpy())
#             pos_probs.append(preds[:, 1])
#         pos_probs = np.array(pos_probs).T
#         neg_probs = 1 - pos_probs  # type: ignore

#         chain_probs = np.zeros((activations[0].shape[0], len(self.context_learners) + 1))
#         for episode_index in range(len(self.context_learners)):
#             if episode_index == 0:
#                 chain_probs[:, 0] = pos_probs[:, 0]
#             else:
#                 prev_neg_prob = np.prod(neg_probs[:, :episode_index], axis=1)
#                 current_pos_prob = prev_neg_prob * pos_probs[:, episode_index]
#                 chain_probs[:, episode_index] = current_pos_prob

#         chain_probs[:, -1] = 1.0 - chain_probs.sum(axis=1)
#         return list(chain_probs.argmax(axis=1) + 1)



#     def predict_context(self, activations: List[torch.Tensor], episode_index: int):
#         if episode_index is None:
#             _, activations = reduce_or_flat_convs(activations)
#             preds = self.tree_preds(activations)
#             return [self.task2classes[i] for i in preds], [i for i in preds]
#         else:
#             return ([self.task2classes[episode_index] for _ in range(activations[0].shape[0])],
#                     [episode_index for _ in range(activations[0].shape[0])])


# def get_n_samples_per_class(dataset, n: int, target_class: int):
#     indices = []
#     # dataset.datasetから対象クラスのインデックスだけを取得
#     for i, (_, y, _) in enumerate(dataset.dataset):
#         if y == target_class:
#             indices.append(i)
    
#     if len(indices) == 0:
#         return None, None

#     # n個のサンプルを取得
#     subset_indices = indices[:n]
#     dataloader = DataLoader(Subset(dataset.dataset, subset_indices), batch_size=len(subset_indices))
#     samples, _, _ = next(iter(dataloader))
#     return samples, target_class

from argparse import Namespace
from typing import Any, Dict, List
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from Source.helper import get_device
import torch.nn.functional as F

class SimilarityAnalyzer():
    def __init__(self, args: Namespace, penultimate_layer_size: int, task2classes: Dict):
        self.args = args
        self.class_prototypes = dict()

    def update_prototypes(self, network: Any, train_episode: Any, episode_index: int):
        print("Updating class prototypes...")
        network.eval()
        classes_in_this_episode = train_episode.classes_in_this_experience
        with torch.no_grad():
            for class_id in classes_in_this_episode:
                samples, _ = get_n_samples_per_class(train_episode, n=50, target_class=class_id)
                if samples is None: continue
                embeddings = network.forward(samples.to(get_device()))
                prototype = embeddings.mean(dim=0)
                self.class_prototypes[class_id] = prototype
                print(f"  - Prototype for class {class_id} updated.")
        network.train()

    def calculate_similarity_score(self, network: Any, data_loader: DataLoader) -> float:
        if not self.class_prototypes:
            print("Similarity calculation skipped: No known class prototypes.")
            return 0.0

        network.eval()
        data, _, _ = next(iter(data_loader))
        data = data.to(get_device())
        max_similarity = 0.0
        with torch.no_grad():
            current_embeddings = network.forward(data)
            prototypes_tensor = torch.stack(list(self.class_prototypes.values()))
            cos_sim = F.cosine_similarity(current_embeddings.unsqueeze(1), prototypes_tensor.unsqueeze(0), dim=2)
            max_similarity = cos_sim.max().item()
        network.train()
        print(f"Calculated Max Similarity Score: {max_similarity:.4f}")
        return max_similarity

def get_n_samples_per_class(dataset, n: int, target_class: int):
    indices = [i for i, (_, y, _) in enumerate(dataset.dataset) if y == target_class]
    if not indices: return None, None
    subset_indices = indices[:n]
    dataloader = DataLoader(Subset(dataset.dataset, subset_indices), batch_size=len(subset_indices))
    samples, _, _ = next(iter(dataloader))
    return samples, target_class