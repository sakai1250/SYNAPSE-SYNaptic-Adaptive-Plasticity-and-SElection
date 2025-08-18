# SYNAPSE/visualize_results.py

import argparse
import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch  # torchをインポート
from sklearn.manifold import TSNE

"""
python3 visualize_results.py "/media/blackie/8000GB_blackie/SYNAPSE-SYNaptic-Adaptive-Plasticity-and-SElection/NICE/Logs/20250818_214636/CIFAR100_MEMO1_SEED0"
"""

def plot_metrics(df, save_dir):
    """
    DataFrameから主要なメトリクスの推移をプロットする。
    """
    # グラフ1: 平均CIL精度とタスク1の精度（忘却の代理指標）
    plt.figure(figsize=(12, 7))
    plt.plot(df['episode'], df['acc/cil_avg_test'], marker='o', linestyle='-', label='Average CIL Accuracy')
    # 'acc/cil_test_task_1' 列が存在する場合のみプロット
    if 'acc/cil_test_task_1' in df.columns:
        plt.plot(df['episode'], df['acc/cil_test_task_1'], marker='s', linestyle='--', label='Task 1 Accuracy (Forgetting)')
    plt.title('Continual Learning Performance')
    plt.xlabel('Episode')
    plt.ylabel('Accuracy')
    plt.grid(True, which='both', linestyle='--')
    plt.legend()
    plt.savefig(save_dir / 'accuracy_and_forgetting.png')
    plt.close()
    print(f"Accuracy plot saved in: {save_dir / 'accuracy_and_forgetting.png'}")

    # グラフ2: 未熟ニューロンの割合
    if 'neurons/immature_ratio' in df.columns:
        plt.figure(figsize=(12, 7))
        plt.plot(df['episode'], df['neurons/immature_ratio'], marker='o', linestyle='-', color='g')
        plt.title('Immature Neuron Ratio')
        plt.xlabel('Episode')
        plt.ylabel('Ratio')
        plt.grid(True, which='both', linestyle='--')
        plt.savefig(save_dir / 'immature_ratio.png')
        plt.close()
        print(f"Immature neuron ratio plot saved in: {save_dir / 'immature_ratio.png'}")
    else:
        print("Immature neuron ratio plot could not be created.")

    # グラフ3: SYNAPSEの活動
    if 'synapse/pruned_blocks' in df.columns:
        plt.figure(figsize=(12, 7))
        plt.bar(df['episode'], df['synapse/pruned_blocks'], label='Pruned Blocks')
        # 'synapse/shared_blocks' 列が存在する場合のみプロット
        if 'synapse/shared_blocks' in df.columns:
            plt.bar(df['episode'], df['synapse/shared_blocks'], bottom=df['synapse/pruned_blocks'], label='Shared Blocks')
        plt.title('SYNAPSE Activity')
        plt.xlabel('Episode')
        plt.ylabel('Count')
        plt.gca().yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        plt.legend()
        plt.savefig(save_dir / 'synapse_activity.png')
        plt.close()
        print(f"SYNAPSE activity plot saved in: {save_dir / 'synapse_activity.png'}")
    else:
        print("SYNAPSE activity plot could not be created.")

    print(f"Metrics plots saved in: {save_dir}")

def plot_tsne(log_dir, save_dir):
    """
    各エピソードの活性化pklファイルを読み込み、t-SNEプロットを生成する。
    """
    try:
        # 最初の2つのタスク（エピソード）のデータのみを対象とする
        with open(log_dir / 'Episode_1/test_dataset_activations.pkl', 'rb') as f:
            # torch.catを使用してテンソルリストを結合
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
        # ユニークなラベルごとにプロットして凡例を作成
        unique_labels = np.unique(all_labels)
        colors = plt.cm.get_cmap('tab20', len(unique_labels))

        for i, label in enumerate(unique_labels):
            indices = all_labels == label
            plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1], color=colors(i), label=f'Class {label}', alpha=0.7)

        plt.title('t-SNE of Feature Space (Tasks 1 & 2)')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.legend(loc='best', bbox_to_anchor=(1.05, 1))
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.savefig(save_dir / 'tsne_features.png')
        plt.close()
        print(f"t-SNE plot saved in: {save_dir}")
        
    except FileNotFoundError as e:
        print(f"Could not generate t-SNE plot (file not found): {e}")
    except Exception as e:
        print(f"An error occurred during t-SNE plotting: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize SYNAPSE experiment results and convert metrics to CSV.")
    parser.add_argument("log_dir", type=str, help="Path to the experiment log directory (e.g., 'Logs/20250818_123456/TinyImagenet_MEMO1_SEED0')")
    args = parser.parse_args()

    log_path = Path(args.log_dir)
    
    # --- metrics.pklを読み込み、CSVに変換し、可視化 ---
    metrics_file = log_path / 'metrics.pkl'
    if metrics_file.exists():
        try:
            with open(metrics_file, 'rb') as f:
                all_metrics_data = pickle.load(f)
            
            # データフレームに変換
            df_metrics = pd.DataFrame(all_metrics_data)
            df_metrics['episode'] = range(1, len(df_metrics) + 1)
            
            # CSVファイルとして保存
            csv_save_path = log_path / 'metrics.csv'
            df_metrics.to_csv(csv_save_path, index=False)
            print(f"Successfully converted metrics and saved to: {csv_save_path}")

            # グラフをプロット
            plot_metrics(df_metrics, log_path)

        except Exception as e:
            print(f"An error occurred while processing {metrics_file}: {e}")
    else:
        print(f"Error: metrics.pkl not found in {log_path}")

    # --- 活性化データ（activation.pkl）をt-SNEで可視化 ---
    plot_tsne(log_path, log_path)