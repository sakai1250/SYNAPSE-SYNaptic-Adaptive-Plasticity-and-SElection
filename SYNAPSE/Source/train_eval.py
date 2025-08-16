from argparse import Namespace
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Any
import numpy as np
from tqdm import tqdm   

from Source.helper import get_device
import wandb


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
    accuracy = sum(predictions == ground_truths) / len(predictions)
    
    # =================================================================
    # Wandb: テスト精度を記録
    # =================================================================
    # wandbが初期化されているかチェック
    if wandb.run is not None:
        # TIL評価かCIL評価かを判断してログを分ける
        log_key = f"Test/acc_til_episode_{episode_id}" if episode_id is not None else "Test/acc_cil"
        wandb.log({log_key: accuracy})
    # =================================================================
    
    network.train()
    if return_preds:
        return accuracy, predictions, ground_truths, episode_preds_all
    else:
        return accuracy

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

        avg_ce_loss = sum(epoch_ce_loss) / len(epoch_ce_loss)
        avg_l2_loss = sum(epoch_l2_loss) / len(epoch_l2_loss)
        
        # =================================================================
        # Wandb: 学習損失をエポックごとに記録
        # =================================================================
        if wandb.run is not None:
            wandb.log({
                "Train/epoch_ce_loss": avg_ce_loss,
                "Train/epoch_l2_loss": avg_l2_loss,
                "epoch": epoch
            })
        # =================================================================
        
        print("Average training loss input: {}".format(avg_ce_loss))
        print("Average l2 loss: {}".format(avg_l2_loss))
    return network
