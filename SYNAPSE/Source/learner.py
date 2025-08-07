from argparse import Namespace
from avalanche.benchmarks import GenericCLScenario, TCLExperience
from typing import Dict, Any
import torch
from Source.helper import get_device, get_data_loaders, initialize_masks
from Source.log import log_end_of_episode, log_end_of_sequence, log_end_of_phase
from Source.train_eval import test, phase_training_scl
from Source.context_detector import SimilarityAnalyzer
from Source.synapse_operations import (
    add_new_neuron, mature_transitional_neurons, update_freeze_masks_synapse,
    share_neuron, duplicate_neuron, get_most_activated_mature_neuron,
    initialize_strategically, integrate_neurons, replenish_immature_pool
)
import os
from tqdm import tqdm

class Learner():
    def __init__(self, args: Namespace, network: Any, scenario: GenericCLScenario,
                 input_size: int, task2classes: Dict, log_dirpath: str):
        self.args = args
        network = network.to(get_device())
        self.network = initialize_masks(network)
        self.task2classes = task2classes
        self.log_dirpath = log_dirpath
        self.optim_obj = getattr(torch.optim, args.optimizer)
        self.similarity_analyzer = SimilarityAnalyzer(args, 0, task2classes)
        self.original_scenario = scenario
        print("Model: \n", self.network)

    def start_episode(self, train_episode: TCLExperience, episode_index: int):
        print(f"****** Starting Episode-{episode_index}   Classes: {train_episode.classes_in_this_experience} ******")

    def end_episode(self, train_episode: TCLExperience, episode_index: int):
        print("****** Ending Episode ****** ")
        self.similarity_analyzer.update_prototypes(self.network, train_episode, episode_index)
        log_end_of_episode(self.args, self.network, self.original_scenario, episode_index, self.log_dirpath)

    def learn_episode(self, train_episode: TCLExperience, val_episode: TCLExperience, test_episode: TCLExperience, episode_index: int):
        train_loader, val_loader, test_loader = get_data_loaders(self.args, train_episode, val_episode, test_episode)
        
        # === Step 1 & 2: 分析と戦略決定 (エピソードの最初に1回だけ実行) ===
        similarity_score = self.similarity_analyzer.calculate_similarity_score(self.network, train_loader)
        newly_added_neurons = {}
        if similarity_score > 0.8:
            print("SYNAPSE Operation: Share")
            target_info = get_most_activated_mature_neuron(self.network, train_loader)
            if target_info: self.network = share_neuron(self.network, *target_info)
        elif 0.4 < similarity_score <= 0.8:
            print("SYNAPSE Operation: Duplicate")
            target_info = get_most_activated_mature_neuron(self.network, train_loader)
            if target_info: self.network = duplicate_neuron(self.network, *target_info)
        else:
            print("SYNAPSE Operation: Add")
            layer_idx_output = len(self.network.unit_ranks) - 1
            num_new_classes = len(train_episode.classes_in_this_experience)
            self.network = add_new_neuron(self.network, layer_idx_output, num_new_classes)
            newly_added_neurons[layer_idx_output] = self.network.transitional_neurons[layer_idx_output]

        # === Step 3: 戦略的初期化 (エピソードの最初に1回だけ実行) ===
        if newly_added_neurons:
            self.network = initialize_strategically(self.network, newly_added_neurons)
        
        # === Step 4 (前半): 訓練と中間評価 (フェーズごとに繰り返し) ===
        for phase_index in range(1, self.args.max_phases + 1):
            print(f"\n--- Starting Phase {phase_index}/{self.args.max_phases} ---")
            optimizer = self.optim_obj(self.network.parameters(), lr=self.args.learning_rate)
            self.network = phase_training_scl(self.network, self.args.phase_epochs, optimizer, train_loader, self.args)
            
            # 各フェーズ終了直後にログを記録
            log_end_of_phase(self.args, self.network, episode_index, phase_index, test_loader, self.log_dirpath)

        # === Step 4 (後半): 成熟化 (全フェーズ完了後に1回だけ実行) ===
        self.network = mature_transitional_neurons(self.network)
        self.network = update_freeze_masks_synapse(self.network)

    def learn_all_episodes(self):
        ep_bar = tqdm(enumerate(zip(
            self.original_scenario.train_stream, self.original_scenario.val_stream, self.original_scenario.test_stream
        ), 1), total=len(self.original_scenario.train_stream), desc="Episodes")

        for episode_index, (train_task, val_task, test_task) in ep_bar:
            ep_bar.set_description(f"Episode {episode_index}")
            self.start_episode(train_task, episode_index)
            self.learn_episode(train_task, val_task, test_task, episode_index)
            self.end_episode(train_task, episode_index)

            if episode_index > 0 and episode_index % 5 == 0:
                print(f"\n--- Network Maintenance after Episode {episode_index} ---")
                maintenance_loader, _, _ = get_data_loaders(self.args, train_task, val_task, test_task)
                self.network = integrate_neurons(self.network, maintenance_loader)
                self.network = replenish_immature_pool(self.network)
                print("--- Maintenance Complete ---\n")

        log_end_of_sequence(self.args, self.network, self.original_scenario, self.log_dirpath)

    def save_model(self):
        model_save_path = os.path.join(self.log_dirpath, "model.pth")
        torch.save(self.network.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")