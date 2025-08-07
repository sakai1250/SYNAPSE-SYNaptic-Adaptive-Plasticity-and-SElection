from argparse import Namespace
from avalanche.benchmarks import GenericCLScenario, TCLExperience
from typing import Dict, Any
import torch
from Source.helper import random_prune, get_device, get_data_loaders
from Source.context_detector import ContextDetector
from Source.nice_operations import increase_unit_ranks, update_freeze_masks, select_learner_units
from Source.nice_operations import drop_young_to_learner, grow_all_to_young
from Source.log import log_end_of_episode, log_end_of_sequence, log_end_of_phase
from Source.train_eval import test, phase_training_ce

from Source.synapse_operations import add_new_neuron, mature_transitional_neurons, update_freeze_masks_synapse
from Source.train_eval import phase_training_scl
from Source.context_detector import SimilarityAnalyzer
from Source.synapse_operations import (
    add_new_neuron, mature_transitional_neurons, update_freeze_masks_synapse,
    share_neuron, duplicate_neuron, get_most_activated_mature_neuron,
    initialize_strategically,
    integrate_neurons, replenish_immature_pool

)

from tqdm import tqdm
import os

class Learner():

    def __init__(self, args: Namespace, network: Any, scenario: GenericCLScenario,
                 input_size: int, task2classes: Dict, log_dirpath: str):
        self.args, self.input_size = args, input_size
        self.task2classes = task2classes
        self.log_dirpath = log_dirpath
        self.optim_obj = getattr(torch.optim, args.optimizer)
        # this is needed to assign masks for later
        self.network = random_prune(network.to(get_device()), 0.0)
        self.context_detector = ContextDetector(args, network.penultimate_layer_size, task2classes)
        self.original_scenario = scenario
        print("Model: \n", self.network)
        # SYNAPSE: 
        self.similarity_analyzer = SimilarityAnalyzer(args, network.penultimate_layer_size, task2classes)

    def start_episode(self, train_episode: TCLExperience, val_episode: TCLExperience, test_episode: TCLExperience, episode_index: int):
        print("****** Starting Episode-{}   Classes: {} ******".format(episode_index, train_episode.classes_in_this_experience))

    def end_episode(self, train_episode: TCLExperience, val_episode: TCLExperience, test_episode: TCLExperience, episode_index: int):
        print("******  Ending Episode ****** ")
        self.similarity_analyzer.update_prototypes(self.network, train_episode, episode_index)
        # self.context_detector.push_activations(self.network, train_episode, episode_index)
        self.network = increase_unit_ranks(self.network)
        self.network = update_freeze_masks(self.network)
        self.network.freeze_bn_layers()
        log_end_of_episode(self.args, self.network, self.context_detector,
                           self.original_scenario, episode_index, self.log_dirpath)


    def learn_episode(self, train_episode: TCLExperience, ..., episode_index: int):
        train_loader, val_loader, test_loader = get_data_loaders(...)

        # === SYNAPSE Step 1: 類似度分析 ===
        similarity_score = self.similarity_analyzer.calculate_similarity_score(self.network, train_loader)
        
        # === SYNAPSE Step 2: 操作決定 ===
        newly_added_neurons = {} # 戦略的初期化のために、追加されたニューロンを記録

        if similarity_score > 0.8:
            print("SYNAPSE 操作: Share を選択しました。")
            # 最も関連性の高いMatureニューロンを探して共有
            target_info = get_most_activated_mature_neuron(self.network, train_loader)
            if target_info:
                layer_idx, neuron_idx = target_info
                self.network = share_neuron(self.network, layer_idx, neuron_idx)
            else:
                print("Share対象が見つからず、Add操作にフォールバックします。")
                # Add操作を実行
                layer_idx, num_added = 4, 1 # 仮
                self.network = add_new_neuron(self.network, layer_idx, num_added)
                newly_added_neurons[layer_idx] = self.network.transitional_neurons[layer_idx]

        elif 0.4 < similarity_score <= 0.8:
            print("SYNAPSE 操作: Duplicate を選択しました。")
            # 最も関連性の高いMatureニューロンを探して複製
            target_info = get_most_activated_mature_neuron(self.network, train_loader)
            if target_info:
                layer_idx, neuron_idx = target_info
                self.network = duplicate_neuron(self.network, layer_idx, neuron_idx)
            else:
                print("Duplicate対象が見つからず、Add操作にフォールバックします。")
                # Add操作を実行
                layer_idx, num_added = 4, 1 # 仮
                self.network = add_new_neuron(self.network, layer_idx, num_added)
                newly_added_neurons[layer_idx] = self.network.transitional_neurons[layer_idx]
        else: # スコア <= 0.4
            print("SYNAPSE 操作: Add を選択しました。")
            # 新しいニューロンを追加
            layer_idx, num_added = 4, 5 # 例: 中間層に5つ追加
            self.network = add_new_neuron(self.network, layer_idx, num_added)
            newly_added_neurons[layer_idx] = self.network.transitional_neurons[layer_idx]

        # === SYNAPSE Step 3: 戦略的初期化 ===
        if newly_added_neurons:
            self.network = initialize_strategically(self.network, newly_added_neurons)

        # === SYNAPSE Step 4: 訓練 (Training) ===
        print("訓練フェーズを開始します...")
        if self.args.optimizer == "SGD":
            optimizer = self.optim_obj(self.network.parameters(), lr=self.args.learning_rate, weight_decay=0.0,
                                       momentum=self.args.sgd_momentum)
        else:
            optimizer = self.optim_obj(self.network.parameters(), lr=self.args.learning_rate, weight_decay=0.0)

        # SYNAPSE: 新しいSCL訓練関数を呼び出します
        self.network = phase_training_scl(self.network, self.args.phase_epochs, optimizer, train_loader, self.args)

        # === SYNAPSE Step 4: 成熟化 (Maturation) ===
        # 学習が完了したら、TransitionalニューロンをMature状態へ遷移させます
        self.network = mature_transitional_neurons(self.network)

        # 重み保護マスクを更新します
        self.network = update_freeze_masks_synapse(self.network)
        self.network.freeze_bn_layers()

        # (ログ記録などの後処理)
        log_end_of_phase(self.args, self.network, self.similarity_analyzer, episode_index, 1,
                         train_loader, val_loader, test_loader, self.log_dirpath)


    def learn_all_episodes(self):
        for episode_index, (train_task, val_task, test_task) in enumerate(zip(self.original_scenario.train_stream,
                                                                              self.original_scenario.val_stream,
                                                                              self.original_scenario.test_stream), 1):
            self.start_episode(train_task, val_task, test_task, episode_index)
            self.learn_episode(train_task, val_task, test_task, episode_index)
            self.end_episode(train_task, val_task, test_task, episode_index)

            # === SYNAPSE Step 5: 維持管理 (Network Maintenance) ===
            if episode_index > 0 and episode_index % 5 == 0: # 5エピソードごとに実行
                print("\n" + "="*50)
                print(f"エピソード {episode_index} 完了: 定期的なネットワーク維持管理を実行します。")
                
                # 統合処理には代表的なデータが必要なので、現在のタスクのデータローダーを流用
                maintenance_loader, _, _ = get_data_loaders(self.args, train_task, val_task, test_task)

                # 1. 冗長なニューロンを統合
                self.network = integrate_neurons(self.network, maintenance_loader)
                
                # 2. Immature Poolを補充
                self.network = replenish_immature_pool(self.network)
                print("="*50 + "\n")

        log_end_of_sequence(self.args, self.network, self.similarity_analyzer,
                            self.original_scenario, self.log_dirpath)
        
    def save_model(self):
        model_save_path = os.path.join(self.log_dirpath, "model.pth")
        torch.save(self.network.state_dict(), model_save_path)
        print("Model saved to {}".format(model_save_path))