# SYNAPSE/visualize_results.py

import argparse
import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE


def plot_metrics(df, save_dir):
    """
    DataFrameから主要なメトリクスの推移をプロットする。
    """
    # グラフ1: 平均CIL精度とタスク1の精度（忘却の代理指標）
    plt.figure(figsize=(12, 7))
    plt.plot(df['episode'], df['acc/cil_avg_test'], marker='o', linestyle='-', label='Average CIL Accuracy')
    plt.plot(df['episode'], df['acc/cil_test_task_1'], marker='s', linestyle='--', label='Task 1 Accuracy (Forgetting)')
    plt.title('Continual Learning Performance')
    plt.xlabel('Episode')
    plt.ylabel('Accuracy')
    plt.grid(True, which='both', linestyle='--')
    plt.legend()
    plt.savefig(save_dir / 'accuracy_and_forgetting.png')
    plt.close()

    # グラフ2: 未熟ニューロンの割合
    plt.figure(figsize=(12, 7))
    plt.plot(df['episode'], df['neurons/immature_ratio'], marker='o', linestyle='-', color='g')
    plt.title('Immature Neuron Ratio')
    plt.xlabel('Episode')
    plt.ylabel('Ratio')
    plt.grid(True, which='both', linestyle='--')
    plt.savefig(save_dir / 'immature_ratio.png')
    plt.close()

    # グラフ3: SYNAPSEの活動
    plt.figure(figsize=(12, 7))
    plt.bar(df['episode'], df['synapse/pruned_blocks'], label='Pruned Blocks')
    plt.title('SYNAPSE Activity')
    plt.xlabel('Episode')
    plt.ylabel('Count')
    plt.gca().yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.legend()
    plt.savefig(save_dir / 'synapse_activity.png')
    plt.close()
    
    print(f"Metrics plots saved in: {save_dir}")

def plot_tsne(log_dir, save_dir):
    """
    各エピソードの活性化pklファイルを読み込み、t-SNEプロットを生成する。
    """
    try:
        # 最初の2つのタスク（エピソード）のデータのみを対象とする
        with open(log_dir / 'Episode_1/test_dataset_activations.pkl', 'rb') as f:
            activations1 = torch.cat(pickle.load(f)).numpy()
        with open(log_dir / 'Episode_1/test_dataset_labels.pkl', 'rb') as f:
            labels1 = torch.cat(pickle.load(f)).numpy()

        with open(log_dir / 'Episode_2/test_dataset_activations.pkl', 'rb') as f:
            activations2 = torch.cat(pickle.load(f)).numpy()
        with open(log_dir / 'Episode_2/test_dataset_labels.pkl', 'rb') as f:
            labels2 = torch.cat(pickle.load(f)).numpy()
        
        # データを結合
        all_activations = np.concatenate((activations1, activations2), axis=0)
        all_labels = np.concatenate((labels1, labels2), axis=0)
        
        print("Starting t-SNE calculation... (This may take a while)")
        tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
        tsne_results = tsne.fit_transform(all_activations)
        
        plt.figure(figsize=(12, 10))
        # タスク1のクラスをプロット
        task1_indices = np.isin(all_labels, np.unique(labels1))
        plt.scatter(tsne_results[task1_indices, 0], tsne_results[task1_indices, 1], c=all_labels[task1_indices], cmap='tab10', alpha=0.6, label='Task 1 Classes')
        # タスク2のクラスをプロット
        task2_indices = np.isin(all_labels, np.unique(labels2))
        plt.scatter(tsne_results[task2_indices, 0], tsne_results[task2_indices, 1], c=all_labels[task2_indices], cmap='viridis', alpha=0.6, marker='x', label='Task 2 Classes')

        plt.title('t-SNE of Feature Space (Task 1 vs Task 2)')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.legend()
        plt.savefig(save_dir / 'tsne_features.png')
        plt.close()
        print(f"t-SNE plot saved in: {save_dir}")
        
    except FileNotFoundError as e:
        print(f"Could not generate t-SNE plot: {e}")
    except Exception as e:
        print(f"An error occurred during t-SNE plotting: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize SYNAPSE experiment results from pkl files.")
    parser.add_argument("log_dir", type=str, help="Path to the experiment log directory (e.g., 'Logs/20250818_123456/TinyImagenet_MEMO1_SEED0')")
    args = parser.parse_args()

    log_path = Path(args.log_dir)
    
    # 1. メトリクスpklファイルの可視化
    metrics_file = log_path / 'metrics.pkl'
    if metrics_file.exists():
        with open(metrics_file, 'rb') as f:
            all_metrics_data = pickle.load(f)
        df_metrics = pd.DataFrame(all_metrics_data)
        df_metrics['episode'] = range(1, len(df_metrics) + 1)
        plot_metrics(df_metrics, log_path)
    else:
        print(f"Error: metrics.pkl not found in {log_path}")

    # 2. 活性化pklファイルの可視化 (t-SNE)
    plot_tsne(log_path, log_path)