from argparse import Namespace
from avalanche.benchmarks import GenericCLScenario
from typing import Any, List, Tuple # NameErrorを修正
import os
import csv
import torch
from Source.train_eval import test
from Source.helper import get_data_loaders
from Source.resnet18 import ResNet18
from torchsummary import summary
from tqdm import tqdm

def calculate_accuracy(preds, gts):
    return np.mean(np.array(preds) == np.array(gts))

def acc_prev_tasks(args: Namespace, network: Any, task_index: int, scenario: GenericCLScenario) -> List[Tuple[str, float]]:
    all_accuracies = []
    # 過去タスクの評価ループにtqdmを適用
    for test_task in tqdm(scenario.test_stream[:task_index], desc="Evaluating Past Tasks", leave=False):
        task_classes = str(test_task.classes_in_this_experience)
        _, _, test_loader = get_data_loaders(args, test_task, test_task, test_task)
        
        test_acc = test(network, test_loader)
        all_accuracies.append((task_classes, test_acc))
    return all_accuracies

def log_end_of_episode(args: Namespace, network: Any, scenario: GenericCLScenario, episode_index: int, dirpath: str):
    """ SYNAPSE用に修正されたエピソード終了時のログ記録 """
    dirpath_episode = os.path.join(dirpath, f"Episode_{episode_index}")
    os.makedirs(dirpath_episode, exist_ok=True)
    
    csv_path = os.path.join(dirpath_episode, f"Episode_{episode_index}_summary.csv")
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Task", "Test Accuracy"])
        
        # 過去のタスクに対する精度を評価
        prev_task_accs = acc_prev_tasks(args, network, episode_index, scenario)
        for task_classes, test_acc in prev_task_accs:
            writer.writerow([task_classes, f"{test_acc:.4f}"])

def log_end_of_phase(args: Namespace, network: Any, episode_index: int, phase_index: int,
                     test_loader: Any, dirpath: str):
    """ SYNAPSE用に、引数をシンプルにしたログ関数 """
    dirpath_phase = os.path.join(dirpath, f"Episode_{episode_index}", f"Phase_{phase_index}")
    os.makedirs(dirpath_phase, exist_ok=True)
    
    csv_path = os.path.join(dirpath_phase, f"Phase_{phase_index}.csv")
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer = write_units(writer, network)
        test_accuracy = test(network, test_loader)
        writer.writerow(["Test Accuracy", f"{test_accuracy:.4f}"])
        print(f"Episode-{episode_index}, Phase-{phase_index} Test Accuracy: {test_accuracy:.4f}")


def log_end_of_sequence(args: Namespace, network: Any, scenario: GenericCLScenario, dirpath: str):
    print("Logging end of sequence summary...")
    try:
        summary_str = str(summary(network, dataset2input(args.dataset)))
        with open(os.path.join(dirpath, "model_summary.txt"), 'w') as f:
            f.write(summary_str)
    except Exception as e:
        print(f"Could not generate model summary: {e}")

def write_units(writer, network: Any):
    all_units_counts = [len(ranks) for ranks, _ in network.unit_ranks[1:]]
    writer.writerow(["All Units"] + all_units_counts)
    writer.writerow(["Immature Neurons"] + [len(u) for u in network.immature_neurons[1:]])
    writer.writerow(["Transitional Neurons"] + [len(u) for u in network.transitional_neurons[1:]])
    writer.writerow(["Mature Neurons"] + [len(u) for u in network.mature_neurons[1:]])
    return writer