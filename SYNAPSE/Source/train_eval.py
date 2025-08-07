from argparse import Namespace
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Any
import numpy as np
from tqdm import tqdm   
from pytorch_metric_learning import losses

from Source.helper import get_device


def test(network: Any, context_detector: Any, data_loader: DataLoader, episode_id=None, return_preds=False) -> float:
    network.eval()
    predictions = []
    ground_truths = []
    episode_preds_all = []
    with torch.no_grad():
        for data, target, _ in tqdm(data_loader, desc="Testing", unit="batch", leave=False):
            data = data.to(get_device())
            target = target.to(get_device())
            output, activations = network.get_activations(data, return_output=True)
            class_preds, episode_preds = context_detector.predict_context(activations, episode_id)

            for index, episode_pred in tqdm(enumerate(class_preds), desc="Episode Predictions", unit="prediction", leave=False):
                output[index, episode_pred] = output[index, episode_pred] + 99999
            preds = output.argmax(dim=1, keepdim=True)
            predictions.extend(preds)
            ground_truths.extend(target)
            episode_preds_all.extend(episode_preds)

    predictions = np.array([int(p) for p in predictions])
    ground_truths = np.array([int(gt) for gt in ground_truths])
    network.train()
    if return_preds:
        return sum(predictions == ground_truths) / len(predictions), predictions, ground_truths, episode_preds_all # type: ignore
    else:
        return sum(predictions == ground_truths) / len(predictions)


def phase_training_ce(network: Any, phase_epochs: int,
                      loss: nn.Module, optimizer: ..., train_loader: DataLoader, args: Namespace) -> Any:
    for epoch in tqdm(range(phase_epochs), desc="Phase Training", unit="epoch", leave=False):
        network.train()
        epoch_l2_loss = []
        epoch_ce_loss = []

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{phase_epochs}", unit="batch", leave=False)
        for data, target, _ in tqdm(train_loader, desc="Training", unit="batch", leave=False):
            data = data.to(get_device())
            target = target.to(get_device())
            optimizer.zero_grad()
            stream_output = network.forward_output(data)
            ce_loss = loss(stream_output, target.long())
            reg_loss = (args.weight_decay * network.l2_loss())
            epoch_ce_loss.append(ce_loss)
            epoch_l2_loss.append(reg_loss)
            batch_loss = reg_loss + ce_loss
            batch_loss.backward()
            if network.freeze_masks:
                network.reset_frozen_gradients()
            optimizer.step()
        print("Average training loss input: {}".format(
            sum(epoch_ce_loss) / len(epoch_ce_loss)))
        print("Average l2 loss: {}".format(
            sum(epoch_l2_loss) / len(epoch_l2_loss)))

    return network


def phase_training_scl(network: Any, phase_epochs: int, optimizer: Any, train_loader: DataLoader, args: Namespace) -> Any:
    """
    Supervised Contrastive Loss を使ってネットワークを訓練します。
    """
    print("Supervised Contrastive Loss を用いた訓練を開始します。")

    # SCL損失関数を初期化します
    # temperatureは、クラス間の距離感を調整する重要なパラメータです
    loss_func = losses.SupConLoss(temperature=0.07)

    for epoch in range(phase_epochs):
        network.train()
        total_loss = 0
        for data, target, _ in train_loader:
            data = data.to(get_device())
            target = target.to(get_device())

            optimizer.zero_grad()

            # SYNAPSEでは、最終出力層の一つ手前の「特徴量(embeddings)」が必要です
            # forward_outputは最終出力なので、モデルに新しいメソッドが必要かもしれません
            # ここでは、network.forward(data)が特徴量を返すと仮定します。
            # (この部分はモデル側の実装と合わせる必要があります)
            embeddings = network.forward(data) # .forward()が出力層手前の特徴量を返すように修正要

            # 損失を計算
            scl_loss = loss_func(embeddings, target)

            # L2正則化項（重み減衰）
            reg_loss = args.weight_decay * network.l2_loss()

            batch_loss = scl_loss + reg_loss
            batch_loss.backward()

            # (NICEの既存ロジック)
            if hasattr(network, 'freeze_masks') and network.freeze_masks:
                network.reset_frozen_gradients()

            optimizer.step()
            total_loss += batch_loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{phase_epochs}, Average SCL Loss: {avg_loss:.4f}")

    return network