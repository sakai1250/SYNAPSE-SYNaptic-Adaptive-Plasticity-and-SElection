# SYNAPSE/Source/context_detector.py

import copy
from argparse import Namespace
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
# Do not delete the following import line, it is needed for the correct functioning of the code
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from Source.helper import get_device, reduce_or_flat_convs
from Source.resnet18 import ResNet18

def inv_dict(d):
    return {vi: k for k, v in d.items() for vi in v}


def concat_tensors(tensor_lists):
    transposed_lists = list(zip(*tensor_lists))
    concatenated_tensors = [torch.cat(tensors, dim=0) for tensors in transposed_lists]
    return concatenated_tensors


class NaiveCLF():
    def predict_proba(self, X):
        a = np.zeros((X.shape[0], 2))
        a[:, 1] = 1.0
        return a
    

class ContextDetector():
    def __init__(self, args: Namespace, penultimate_layer_size: int, task2classes: Dict):
        self.args = args
        self.penultimate_layer_size = penultimate_layer_size
        self.task2classes = task2classes
        self.class2task = inv_dict(task2classes)
        self.context_learner_prototype = eval(self.args.context_learner)

        self.quantized_context_representations = dict()
        self.context_layers_masks = dict()
        self.context_learners = []
        self.layer_binarizers = [Binarizer() for _ in self.args.context_layers]

        self.float_context_representations = dict()

    def train_models(self, current_episode_index: int):
        context_learners = []
        if current_episode_index == 1:
            context_learners.append(NaiveCLF())
        else:
            for prev_episode in tqdm(range(1, current_episode_index), desc="Training Context Learners", unit="episode"):
                X, y = [], []
                # Positive Samples
                pos_activations = [self.layer_binarizers[index].dequantize(acti)
                                   for index, acti in enumerate(self.quantized_context_representations[prev_episode])]
                features = torch.hstack(pos_activations)
                X.append(features)
                y = y + [1]*len(features)

                # Negative Samples
                for neg_episode_indices in tqdm(range(prev_episode + 1, current_episode_index + 1), desc="Negative Samples", unit="episode"):
                    neg_activations = [self.layer_binarizers[index].dequantize(acti)
                                       for index, acti in enumerate(self.quantized_context_representations[neg_episode_indices])]
                    features = torch.hstack(neg_activations)
                    X.append(features)
                    y = y + [0] * len(features)

                m = np.concatenate([a for a, _ in self.context_layers_masks[prev_episode]])
                X = torch.concat(X)
                X_train = np.array(X[:, m].cpu())
                y_train = np.array(y)
                clf = copy.deepcopy(self.context_learner_prototype)
                clf.fit(X_train, y_train)
                context_learners.append(clf)
        self.context_learners = context_learners

    def train_quantizers(self, network: Any, train_episode: Any):
        network.eval()
        subsets = get_n_samples_per_class(train_episode, self.args.memo_per_class_context)
        with torch.no_grad():
            activations_all_classes = []
            for samples, _ in subsets:
                _, layer_activations = reduce_or_flat_convs(network.get_activations(samples.to(get_device())))
                context_layer_activations = [(index, layer_activations[index]) 
                                             for index in self.args.context_layers]
                activations = []
                for _, activation in context_layer_activations:
                    activation = activation.detach().cpu()
                    activations.append(activation)
                activations_all_classes.append(activations)
            layer_activations_all_classes = concat_tensors(activations_all_classes)
            for index, acti in enumerate(layer_activations_all_classes):
                self.layer_binarizers[index].fit(acti)
        network.train()

    def push_activations(self, network: Any, train_episode: Any, episode_index: int):
        if episode_index == 1:
            self.train_quantizers(network, train_episode)
        
        network.eval()
        episode_quantized_context_representations = []
        episodes_float_context_representations = []
        subsets = get_n_samples_per_class(train_episode,
                                          self.args.memo_per_class_context)
        with torch.no_grad():
            for samples, class_ in subsets:
                is_conv, layer_activations = reduce_or_flat_convs(
                                             network.get_activations(samples.to(get_device())))
                context_layer_activations = [(index, layer_activations[index])
                                             for index in self.args.context_layers]

                quantized_activations = []
                context_masks = []
                
                # `layer_binarizers` のインデックスとして使用するカウンター
                binarizer_counter = 0
                for index, activation in context_layer_activations:
                    
                    # ### 修正箇所 1 ###
                    # `unit_ranks` のデータ構造に合わせてマスクを正しく生成
                    mask = np.ones(activation.shape[1], dtype=bool) # デフォルトは全ニューロンを使用
                    
                    # `unit_ranks` は (ranks, name) のタプルのリストであると想定
                    if index < len(network.unit_ranks):
                        # タプルからランク情報(リスト)のみを正しく取り出す
                        ranks_for_layer, _ = network.unit_ranks[index]
                        
                        # ランク情報がリスト形式であることを確認 (VGG11_SLIMの古い形式への後方互換性)
                        if isinstance(ranks_for_layer, list):
                            current_mask = np.array([bool(r) for r in ranks_for_layer], dtype=bool)
                            # マスクと活性化テンソルの次元が一致するか検証
                            if current_mask.shape[0] == activation.shape[1]:
                                mask = current_mask
                            else:
                                print(f"[WARNING] Mask shape mismatch in push_activations for activation index {index}. Using all neurons.")
                    
                    context_masks.append((mask, index))
                    quantized_activations.append(self.layer_binarizers[binarizer_counter].quantize(activation))
                    binarizer_counter += 1

                episodes_float_context_representations.append((context_layer_activations, class_, is_conv))
                self.context_layers_masks[episode_index] = context_masks
                episode_quantized_context_representations.append(quantized_activations)

        self.float_context_representations[episode_index] = episodes_float_context_representations
        self.quantized_context_representations[episode_index] = concat_tensors(episode_quantized_context_representations)
        self.train_models(episode_index)
        network.train()

    def process_and_stack(self, context_masks, activations):
        activation_list = []
        
        # ### 修正箇所 2 ###
        # `layer_binarizers` のインデックスとしてカウンターを使用
        binarizer_counter = 0
        for mask, index in context_masks:
            activation = activations[index]
            quantized = self.layer_binarizers[binarizer_counter].quantize(activation)
            dequantized = self.layer_binarizers[binarizer_counter].dequantize(quantized)
            
            # NumPyのbool配列をPyTorchのboolテンソルに変換
            mask_tensor = torch.from_numpy(mask).to(dequantized.device)
            
            # 安全のため、ここでも形状をチェック
            if mask_tensor.shape[0] == dequantized.shape[1]:
                activation_list.append(dequantized[:, mask_tensor])
            else:
                print(f"[WARNING] Mismatch in process_and_stack for activation index {index}. Using full tensor.")
                activation_list.append(dequantized)
                
            binarizer_counter += 1
        
        # マスク適用後に空になったテンソルを除外
        non_empty_activations = [act for act in activation_list if act.shape[1] > 0]
        if not non_empty_activations:
            return torch.zeros((activations[0].shape[0], 0), device=activations[0].device)

        return torch.hstack(non_empty_activations)
    def tree_preds(self, activations) -> List[int]:
        pos_probs = []
        for index, model in enumerate(self.context_learners, 1):
            context_masks = self.context_layers_masks[index]
            X = self.process_and_stack(context_masks, activations)
            preds = model.predict_proba(X.cpu().numpy())
            pos_probs.append(preds[:, 1])
        pos_probs = np.array(pos_probs).T
        neg_probs = 1 - pos_probs  # type: ignore

        chain_probs = np.zeros((activations[0].shape[0], len(self.context_learners) + 1))
        for episode_index in range(len(self.context_learners)):
            if episode_index == 0:
                chain_probs[:, 0] = pos_probs[:, 0]
            else:
                prev_neg_prob = np.prod(neg_probs[:, :episode_index], axis=1)
                current_pos_prob = prev_neg_prob * pos_probs[:, episode_index]
                chain_probs[:, episode_index] = current_pos_prob

        chain_probs[:, -1] = 1.0 - chain_probs.sum(axis=1)
        return list(chain_probs.argmax(axis=1) + 1)


    def predict_context(self, activations: List[torch.Tensor], episode_index: int):
        if episode_index is None:
            _, activations = reduce_or_flat_convs(activations)
            preds = self.tree_preds(activations)
            return [self.task2classes[i] for i in preds], [i for i in preds]
        else:
            return ([self.task2classes[episode_index] for _ in range(activations[0].shape[0])],
                    [episode_index for _ in range(activations[0].shape[0])])


def get_n_samples_per_class(dataset, n: int) -> List:  # type: ignore
    indices = {i: [] for i in dataset.classes_in_this_experience}
    for i, (_, y, _) in enumerate(dataset.dataset):
        indices[y].append(i)

    subsets = []
    for i in dataset.classes_in_this_experience:
        dataloader = DataLoader(Subset(dataset.dataset, indices[i][:n]), batch_size=n)
        samples, _, _ = next(iter(dataloader))
        subsets.append((samples, i))
    return subsets

class Binarizer:
    def __init__(self):
        self.mean_val = 0.0

    def fit(self, x):
        self.mean_val = x.mean() + x.std()

    def quantize(self, x):
        return x > self.mean_val

    def dequantize(self, quantized):
        return quantized.int()
