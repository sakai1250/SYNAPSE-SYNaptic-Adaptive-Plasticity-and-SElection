from argparse import Namespace
from typing import Any, Dict, List
import torch
from Source.helper import get_device, reduce_or_flat_convs
import numpy as np
import copy
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F

# Do not delete the following import line, it is needed for the correct functioning of the code
from sklearn.linear_model import LogisticRegression

from tqdm import tqdm

def inv_dict(d):
    return {vi: k for k, v in d.items() for vi in v}


def concat_tensors(tensor_lists):
    # Transpose the list of lists
    transposed_lists = list(zip(*tensor_lists))
    # Concatenate tensors with the same index
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

        # Episode id -> list of layer activations
        self.quantized_context_representations = dict()
        # Episode id -> list of binary vectors of varying sizes (neuron stable/candidate by time = 1)
        self.context_layers_masks = dict()
        self.context_learners = []  # List of episode predictors
        self.layer_binarizers = [Binarizer() for _ in self.args.context_layers]  # List of quantizers

        self.float_context_representations = dict()

        # ADAN用: 各クラスの「お手本」（平均活性化ベクトル）を保存する場所
        self.class_prototypes = dict()


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
        # Binary context Part
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
                if len(is_conv) != len(context_layer_activations):
                    raise Exception("We did not tested using layers partially for context")

                quantized_activations = []
                context_masks = []
                for index, activation in context_layer_activations:
                    activation = activation.detach().cpu()
                    context_masks.append((network.unit_ranks[index][0] > 0, index))
                    quantized_activations.append(self.layer_binarizers[index].quantize(activation))

                episodes_float_context_representations.append((context_layer_activations, class_, is_conv))
                self.context_layers_masks[episode_index] = context_masks
                episode_quantized_context_representations.append(quantized_activations)

        self.float_context_representations[episode_index] = episodes_float_context_representations
        self.quantized_context_representations[episode_index] = concat_tensors(episode_quantized_context_representations)
        self.train_models(episode_index)

        # ADAN用: タスク学習後、Matureになったニューロンの活性化を基にお手本を更新する
        network.eval()
        with torch.no_grad():
            classes_in_experience = train_episode.classes_in_this_experience
            for class_id in classes_in_experience:
                # データセットから、このクラスのサンプルをいくつか取得します
                # get_n_samples_per_classはNICEに元からある関数です
                class_subset = get_n_samples_per_class(train_episode, n=self.args.memo_per_class_context, classes_to_get=[class_id])
                if not class_subset:
                    continue
                
                samples, _ = class_subset[0]
                # サンプルをモデルに入力し、各層の活性化ベクトルを取得
                _, activations = reduce_or_flat_convs(network.get_activations(samples.to(get_device())))
                
                # 最終層の一つ手前（penultimate layer）の活性化ベクトルを使います
                # この層がデータの特徴を最もよく表していると考えられるためです
                penultimate_activations = activations[-2]
                
                # そのクラスの「お手本」として、活性化ベクトルの平均値を保存します
                if penultimate_activations.numel() > 0:
                    self.class_prototypes[class_id] = penultimate_activations.mean(dim=0)

        network.train()

    def process_and_stack(self, context_masks, activations):
        activation_list = []
        for mask, index in context_masks:
            activation = activations[index]
            quantized = self.layer_binarizers[index].quantize(activation)
            dequantized = self.layer_binarizers[index].dequantize(quantized)
            activation_list.append(dequantized[:, mask])
        X = torch.hstack(activation_list)
        return X

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

    def calculate_similarity_score(self, activations: list[torch.Tensor]) -> tuple[float, int | None]:
        """
        新しいデータの活性化ベクトルと、記憶している全クラスの「お手本」との
        類似度を計算します。

        戻り値:
            (float): 最も高い類似度スコア (0〜1の範囲)
            (int | None): 最も似ていたクラスのID (お手本がまだ無い場合はNone)
        """
        # まだ比較対象のお手本が一つも無ければ、類似度0を返す
        if not self.class_prototypes:
            return 0.0, None

        # 比較しやすいように、入力された活性化ベクトルを整形します
        _, flattened_activations = reduce_or_flat_convs(activations)
        current_activation = flattened_activations[-2].mean(dim=0)

        # 保存されているお手本の一覧を取得
        class_ids = list(self.class_prototypes.keys())
        prototypes = torch.stack(list(self.class_prototypes.values()))
        
        # コサイン類似度を計算して、最も似ているものを探します
        # ベクトルを正規化してから内積を取ると、コサイン類似度になります
        cos_sim = F.linear(F.normalize(current_activation.unsqueeze(0)), F.normalize(prototypes))
        
        # 最も高い類似度スコアと、それがどのクラスだったかを取得
        max_sim_val, max_idx = torch.max(cos_sim, dim=1)
        best_class_id = class_ids[max_idx.item()]
        
        # コサイン類似度 (-1〜1) を 0〜1 の範囲に変換して返す
        score = (max_sim_val.item() + 1) / 2
        
        return score, best_class_id


# def get_n_samples_per_class(dataset, n: int) -> List:  # type: ignore
#     indices = {i: [] for i in dataset.classes_in_this_experience}
#     for i, (_, y, _) in enumerate(dataset.dataset):
#         indices[y].append(i)

#     subsets = []
#     for i in dataset.classes_in_this_experience:
#         dataloader = DataLoader(Subset(dataset.dataset, indices[i][:n]), batch_size=n)
#         samples, _, _ = next(iter(dataloader))
#         subsets.append((samples, i))
#     return subsets

def get_n_samples_per_class(dataset, n: int, classes_to_get: list | None = None) -> List:  # type: ignore
    """
    データセットからクラスごとにn個のサンプルを取得する。
    classes_to_getを指定すると、そのクラスのサンプルのみを取得する。
    """
    all_classes = dataset.classes_in_this_experience
    target_classes = classes_to_get if classes_to_get is not None else all_classes
    
    indices = {i: [] for i in all_classes}
    # dataset.datasetからインデックスを収集
    # datasetがTensorDatasetの場合など、.dataset属性がない場合も考慮
    try:
        data_source = dataset.dataset
    except AttributeError:
        data_source = dataset

    for i, (_, y, *_) in enumerate(data_source):
        # yがTensorの場合、数値に変換
        y_val = y.item() if isinstance(y, torch.Tensor) else y
        if y_val in indices:
            indices[y_val].append(i)

    subsets = []
    for i in target_classes:
        if i not in indices or not indices[i]:
            continue
        
        # サンプル数がnより少ない場合、存在する全てのサンプルを使う
        sample_indices = indices[i][:n]
        if not sample_indices:
            continue

        dataloader = DataLoader(Subset(data_source, sample_indices), batch_size=len(sample_indices))
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
