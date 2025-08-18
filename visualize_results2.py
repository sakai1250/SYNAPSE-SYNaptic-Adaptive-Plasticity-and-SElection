# SYNAPSE/visualize_results.py
import argparse
from pathlib import Path
import pickle
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.manifold import TSNE
from typing import Sequence


# /media/blackie/8000GB_blackie/SYNAPSE-SYNaptic-Adaptive-Plasticity-and-SElection
"""
python3 visualize_results2.py \
  "NICE/Logs/20250818_214636/CIFAR100_MEMO1_SEED0" \
  "SYNAPSE/Logs/CIFAR100/20250818_003646_CIFAR100_MEMO1_SEED1"
"""

def to_path(x):
    return x if isinstance(x, Path) else Path(x)

def safe_name(p) -> str:
    p = to_path(p)
    parts = [q for q in p.parts if q not in ("/", "")]
    return "_".join(parts[-2:]) if len(parts) >= 2 else (parts[-1] if parts else str(p))

def make_abs_paths(root: Path, rel_or_abs: Sequence[str]) -> list[Path]:
    root = to_path(root)
    out = []
    for s in rel_or_abs:
        p = to_path(s)
        out.append(p if p.is_absolute() else (root / p))
    return out

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p


def safe_name(p: Path):
    # 末尾2階層くらいで識別しやすく
    parts = [q for q in p.parts if q not in ("/", "")]
    return "_".join(parts[-2:]) if len(parts) >= 2 else parts[-1]


def load_metrics_df(log_path: Path):
    metrics_file = log_path / "metrics.pkl"
    if not metrics_file.exists():
        print(f"[skip] metrics.pkl not found: {log_path}")
        return None
    try:
        with open(metrics_file, "rb") as f:
            data = pickle.load(f)
        df = pd.DataFrame(data)
        if "episode" not in df.columns:
            df["episode"] = range(1, len(df) + 1)
        # CSV保存
        csv_path = log_path / "metrics.csv"
        df.to_csv(csv_path, index=False)
        print(f"[ok] metrics.csv saved: {csv_path}")
        return df
    except Exception as e:
        print(f"[error] reading metrics.pkl in {log_path}: {e}")
        return None


def plot_metrics_single(df: pd.DataFrame, save_dir: Path):
    # 個別図: 平均CIL、タスク1精度
    fig = plt.figure(figsize=(12, 7))
    plt.plot(df["episode"], df["acc/cil_avg_test"], marker="o", linestyle="-", label="Average CIL Accuracy")
    if "acc/cil_test_task_1" in df.columns:
        plt.plot(df["episode"], df["acc/cil_test_task_1"], marker="s", linestyle="--", label="Task 1 Accuracy (Forgetting)")
    plt.title("Continual Learning Performance")
    plt.xlabel("Episode")
    plt.ylabel("Accuracy")
    plt.grid(True, which="both", linestyle="--")
    plt.legend()
    fig.savefig(save_dir / "accuracy_and_forgetting.png")
    plt.close()
    print(f"[ok] saved: {save_dir / 'accuracy_and_forgetting.png'}")

    # 個別図: 未熟ニューロン割合
    if "neurons/immature_ratio" in df.columns:
        fig = plt.figure(figsize=(12, 7))
        plt.plot(df["episode"], df["neurons/immature_ratio"], marker="o", linestyle="-")
        plt.title("Immature Neuron Ratio")
        plt.xlabel("Episode")
        plt.ylabel("Ratio")
        plt.grid(True, which="both", linestyle="--")
        fig.savefig(save_dir / "immature_ratio.png")
        plt.close()
        print(f"[ok] saved: {save_dir / 'immature_ratio.png'}")
    else:
        print("[info] immature ratio not found, skip plot.")

    # 個別図: SYNAPSE活動
    if "synapse/pruned_blocks" in df.columns:
        fig = plt.figure(figsize=(12, 7))
        plt.bar(df["episode"], df["synapse/pruned_blocks"], label="Pruned Blocks")
        if "synapse/shared_blocks" in df.columns:
            bottom = df["synapse/pruned_blocks"].values
            plt.bar(df["episode"], df["synapse/shared_blocks"], bottom=bottom, label="Shared Blocks")
        plt.title("SYNAPSE Activity")
        plt.xlabel("Episode")
        plt.ylabel("Count")
        plt.gca().yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        plt.legend()
        fig.savefig(save_dir / "synapse_activity.png")
        plt.close()
        print(f"[ok] saved: {save_dir / 'synapse_activity.png'}")
    else:
        print("[info] synapse activity not found, skip plot.")


def _collect_task_series(df_row: pd.Series, prefix: str):
    cols = [c for c in df_row.index if c.startswith(prefix)]
    pairs = []
    for c in cols:
        try:
            tid = int(c.split(prefix)[1])
        except Exception:
            continue
        val = df_row[c]
        if pd.notna(val):
            pairs.append((tid, float(val)))
    pairs.sort(key=lambda x: x[0])
    ids = [p[0] for p in pairs]
    vals = [p[1] for p in pairs]
    return ids, vals


def _plot_bar(task_ids, scores, title, save_path: Path):
    fig = plt.figure(figsize=(12, 6))
    x = np.arange(len(task_ids))
    plt.bar(x, scores)
    plt.xticks(x, [f"T{t}" for t in task_ids], rotation=0)
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.ylim(0, 1.0)
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close()
    print(f"[ok] saved: {save_path}")


def export_final_task_bars_and_csv(df: pd.DataFrame, save_dir: Path, label_suffix: str):
    if df is None or "episode" not in df.columns or len(df) == 0:
        print("[skip] no metrics for final task bars.")
        return
    max_ep = int(df["episode"].max())
    final_row = df.loc[df["episode"] == max_ep].iloc[0]

    # CIL
    cil_tasks, cil_scores = _collect_task_series(final_row, prefix="acc/cil_test_task_")
    if len(cil_tasks) > 0:
        _plot_bar(
            cil_tasks, cil_scores,
            title=f"CIL Task-wise Accuracy (Episode {max_ep}) [{label_suffix}]",
            save_path=save_dir / f"cil_task_accuracy_final__{label_suffix}.png",
        )
        cil_df = pd.DataFrame({"task": cil_tasks, "accuracy": cil_scores})
        cil_csv = save_dir / f"cil_task_accuracy_final__{label_suffix}.csv"
        cil_df.to_csv(cil_csv, index=False)
        print(f"[ok] saved: {cil_csv}")
    else:
        print("[info] no CIL per-task columns.")

    # TIL
    til_tasks, til_scores = _collect_task_series(final_row, prefix="acc/til_test_task_")
    if len(til_tasks) > 0:
        _plot_bar(
            til_tasks, til_scores,
            title=f"TIL Task-wise Accuracy (Episode {max_ep}) [{label_suffix}]",
            save_path=save_dir / f"til_task_accuracy_final__{label_suffix}.png",
        )
        til_df = pd.DataFrame({"task": til_tasks, "accuracy": til_scores})
        til_csv = save_dir / f"til_task_accuracy_final__{label_suffix}.csv"
        til_df.to_csv(til_csv, index=False)
        print(f"[ok] saved: {til_csv}")
    else:
        print("[info] no TIL per-task columns.")


def plot_tsne(log_dir: Path, save_dir: Path):
    try:
        with open(log_dir / "Episode_1/test_dataset_activations.pkl", "rb") as f:
            act1 = torch.cat(pickle.load(f)).numpy()
        with open(log_dir / "Episode_1/test_dataset_labels.pkl", "rb") as f:
            lab1 = torch.cat(pickle.load(f)).numpy()
        with open(log_dir / "Episode_2/test_dataset_activations.pkl", "rb") as f:
            act2 = torch.cat(pickle.load(f)).numpy()
        with open(log_dir / "Episode_2/test_dataset_labels.pkl", "rb") as f:
            lab2 = torch.cat(pickle.load(f)).numpy()

        X = np.concatenate((act1, act2), axis=0)
        y = np.concatenate((lab1, lab2), axis=0)

        print("[tsne] start...")
        tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
        Z = tsne.fit_transform(X)

        fig = plt.figure(figsize=(12, 10))
        uniq = np.unique(y)
        cmap = plt.cm.get_cmap("tab20", len(uniq))
        for i, c in enumerate(uniq):
            idx = y == c
            plt.scatter(Z[idx, 0], Z[idx, 1], color=cmap(i), label=f"Class {c}", alpha=0.7)
        plt.title("t-SNE of Feature Space (Tasks 1 & 2)")
        plt.xlabel("t-SNE Dimension 1")
        plt.ylabel("t-SNE Dimension 2")
        plt.legend(loc="best", bbox_to_anchor=(1.05, 1))
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        out = save_dir / "tsne_features.png"
        fig.savefig(out)
        plt.close()
        print(f"[ok] saved: {out}")
    except FileNotFoundError as e:
        print(f"[tsne] missing file: {e}")
    except Exception as e:
        print(f"[tsne] error: {e}")


def compare_plots(run_dfs, run_labels, out_dir: Path):
    # 比較図: 平均CIL
    fig = plt.figure(figsize=(12, 7))
    for df, lb in zip(run_dfs, run_labels):
        if df is None:
            continue
        if "acc/cil_avg_test" not in df.columns:
            print(f"[info] {lb}: no 'acc/cil_avg_test', skip in compare.")
            continue
        plt.plot(df["episode"], df["acc/cil_avg_test"], marker="o", linestyle="-", label=f"{lb} / CIL-Avg")
    plt.title("Average CIL Accuracy (comparison)")
    plt.xlabel("Episode")
    plt.ylabel("Accuracy")
    plt.grid(True, which="both", linestyle="--")
    plt.legend()
    out = out_dir / "compare_cil_avg_accuracy.png"
    fig.savefig(out)
    plt.close()
    print(f"[ok] saved: {out}")

    # 比較図: タスク1精度
    fig = plt.figure(figsize=(12, 7))
    any_line = False
    for df, lb in zip(run_dfs, run_labels):
        if df is None:
            continue
        if "acc/cil_test_task_1" in df.columns:
            plt.plot(df["episode"], df["acc/cil_test_task_1"], marker="s", linestyle="--", label=f"{lb} / T1")
            any_line = True
        else:
            print(f"[info] {lb}: no 'acc/cil_test_task_1', skip in compare.")
    if any_line:
        plt.title("Task 1 Accuracy (comparison)")
        plt.xlabel("Episode")
        plt.ylabel("Accuracy")
        plt.grid(True, which="both", linestyle="--")
        plt.legend()
        out = out_dir / "compare_task1_accuracy.png"
        fig.savefig(out)
        plt.close()
        print(f"[ok] saved: {out}")
    else:
        plt.close()
        print("[info] no run had 'acc/cil_test_task_1'; skip comparison plot.")


def main():
    parser = argparse.ArgumentParser(description="Visualize and compare SYNAPSE experiment results.")
    parser.add_argument(
        "log_dirs",
        type=str,
        nargs="+",
        help="One or more experiment log directories. Example: Logs/20250818_xxx/ExpA Logs/20250818_yyy/ExpB",
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs="*",
        help="Optional labels for runs (same length as log_dirs). Default: derived from path.",
    )
    parser.add_argument(
        "--tsne",
        action="store_true",
        help="Enable t-SNE for each run (Episode_1 and Episode_2 activations required).",
    )
    # main() 内のこのブロックを置き換え
    args = parser.parse_args()

    root = Path("/media/blackie/8000GB_blackie/SYNAPSE-SYNaptic-Adaptive-Plasticity-and-SElection")

    # 引数は相対でも絶対でもOK。相対なら root を前置する。常に Path を返す。
    log_paths = make_abs_paths(root, args.log_dirs)

    labels = args.labels if args.labels and len(args.labels) == len(log_paths) else [safe_name(p) for p in log_paths]

    # 共通親ディレクトリが取れないケースに備えて try
    try:
        common = Path(os.path.commonpath([str(p.resolve()) for p in log_paths]))
    except Exception:
        # 共通親が見つからない場合は root 直下に比較出力を置く
        common = root

    out_dir = ensure_dir(common / "_compare_outputs")
    print(f"[info] compare outputs dir: {out_dir}")

    run_dfs = []
    for p, lb in zip(log_paths, labels):
        print(f"\n=== Run: {lb} ===")
        df = load_metrics_df(p)
        run_dfs.append(df)
        if df is not None:
            plot_metrics_single(df, p)
            export_final_task_bars_and_csv(df, p, label_suffix=lb)
            if args.tsne:
                plot_tsne(p, p)

    # 複数比較図
    compare_plots(run_dfs, labels, out_dir)


if __name__ == "__main__":
    main()
