import os
from argparse import Namespace
from typing import Any, Dict

import torch
from avalanche.benchmarks import GenericCLScenario, TCLExperience
from tqdm import tqdm

import wandb
import pickle
from Source.context_detector import ContextDetector
from Source.helper import get_data_loaders, get_device, random_prune
from Source.log import (log_end_of_episode, log_end_of_phase,
                        log_end_of_sequence)
from Source.nice_operations import (drop_young_to_learner, grow_all_to_young,
                                    increase_unit_ranks, select_learner_units,
                                    update_freeze_masks)
from Source.synapse_operations import run_synapse_optimization
from Source.train_eval import phase_training_ce, test

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
        
        self.all_episode_metrics = []
        
        print("Model: \n", self.network)
        # =================================================================
    def start_episode(self, train_episode: TCLExperience, val_episode: TCLExperience, test_episode: TCLExperience, episode_index: int):
        print("****** Starting Episode-{}   Classes: {} ******".format(episode_index, train_episode.classes_in_this_experience))

    def end_episode(self, train_episode: TCLExperience, val_episode: TCLExperience, test_episode: TCLExperience, episode_index: int):
        print("******  Ending Episode ****** ")
        self.context_detector.push_activations(self.network, train_episode, episode_index)
        self.network = increase_unit_ranks(self.network)
        self.network = update_freeze_masks(self.network)
        self.network.freeze_bn_layers()
        # log_end_of_episode(self.args, self.network, self.context_detector,
        #                    self.original_scenario, episode_index, self.log_dirpath)
        # # =================================================================
        # # SYNAPSE: 最適化フェーズを実行
        # # =================================================================
        # run_synapse_optimization(self.network, self.context_detector, self.args, episode_index)
        # # =================================================================

        # Step 1: 通常のログメトリクスを取得
        episode_metrics = log_end_of_episode(self.args, self.network, self.context_detector,
                                           self.original_scenario, episode_index, self.log_dirpath)

        # Step 2: SYNAPSEを実行し、活動記録メトリクスを取得
        synapse_metrics = run_synapse_optimization(self.network, self.context_detector, self.args, episode_index, train_episode)

        # =================================================================
        # Step 3: 今回のエピソードのメトリクスを結合してリストに追加
        combined_metrics = {**episode_metrics, **synapse_metrics}
        self.all_episode_metrics.append(combined_metrics)


        
    def learn_episode(self, train_episode: TCLExperience, val_episode: TCLExperience, test_episode: TCLExperience, episode_index: int):
        train_loader, val_loader, test_loader = get_data_loaders(self.args, train_episode, val_episode, test_episode, episode_index)
        phase_index = 1
        selection_perc = 100.0
        loss = torch.nn.CrossEntropyLoss()
        while (True):
            print('Selecting Units (Ratio: {})'.format(selection_perc))
            self.network = select_learner_units(self.network, selection_perc, train_episode, episode_index)
            print('Dropping connections')
            self.network = drop_young_to_learner(self.network)
            print('Fixing Young connections')
            self.network = grow_all_to_young(self.network)
            print("Sparsity phase-{}: {:.2f}".format(phase_index,self.network.compute_weight_sparsity()))

            if self.args.optimizer == "SGD":
                optimizer = self.optim_obj(self.network.parameters(), lr=self.args.learning_rate, weight_decay=0.0,
                                           momentum=self.args.sgd_momentum)
            else:
                optimizer = self.optim_obj(self.network.parameters(), lr=self.args.learning_rate, weight_decay=0.0)

            print("Phase-{}: Training".format(phase_index))
            self.network = phase_training_ce(self.network, self.args.phase_epochs, loss, optimizer, train_loader, self.args)

            print("Phase-{}: Testing".format(phase_index))
            # Push context_detector
            self.context_detector.push_activations(self.network, train_episode, episode_index)

            print("Phase-{}: Computing validation accuracy".format(phase_index))
            # Compute validation accuracy
            test_accuracy = test(self.network, self.context_detector, test_loader)

            print("Episode-{}, Phase-{}, Episode Test Class Accuracy: {}".format(episode_index, phase_index, test_accuracy))

            log_end_of_phase(self.args, self.network, self.context_detector, episode_index, phase_index,
                             train_loader, val_loader, test_loader, self.log_dirpath)
            phase_index = phase_index + 1
            selection_perc = self.args.activation_perc

            # Use all neurons in the last episode, selection_perc = 100
            if episode_index == self.args.number_of_tasks:
                selection_perc = 100.0
            if phase_index > self.args.max_phases:
                break

    def learn_all_episodes(self):
        #for episode_index, (train_task, val_task, test_task) in enumerate(
                #zip(self.original_scenario.train_stream, self.original_scenario.val_stream, self.original_scenario.test_stream)):
        # use tqdm
        for episode_index, (train_task, val_task, test_task) in enumerate(
                zip(self.original_scenario.train_stream, self.original_scenario.val_stream, self.original_scenario.test_stream), start=1):
            with tqdm(total=len(self.original_scenario.train_stream), desc="Learning Episodes", unit="episode") as pbar:
                self.start_episode(train_task, val_task, test_task, episode_index)
                self.learn_episode(train_task, val_task, test_task, episode_index)
                self.end_episode(train_task, val_task, test_task, episode_index)
                pbar.update(1)
        log_end_of_sequence(self.args, self.network, self.context_detector,
                            self.original_scenario, self.log_dirpath)

        # 全ての学習が終わった後に、蓄積したメトリクスを pkl ファイルとして保存
        metrics_save_path = os.path.join(self.log_dirpath, "metrics.pkl")
        with open(metrics_save_path, 'wb') as f:
            pickle.dump(self.all_episode_metrics, f)
        print(f"\nAll episode metrics saved to: {metrics_save_path}")

    def save_model(self):
        model_save_path = os.path.join(self.log_dirpath, "model.pth")
        torch.save(self.network.state_dict(), model_save_path)
        print("Model saved to {}".format(model_save_path))