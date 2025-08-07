from argparse import Namespace
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Any
import numpy as np
from tqdm import tqdm
from pytorch_metric_learning import losses
from Source.helper import get_device

def test(network: Any, data_loader: DataLoader, return_preds=False) -> Any:
    """
    SYNAPSE用の新しいtest関数。ネットワークの純粋な分類性能を評価します。
    """
    network.eval()
    predictions = []
    ground_truths = []
    
    with torch.no_grad():
        for data, target, _ in tqdm(data_loader, desc="Testing", leave=False):
            data = data.to(get_device())
            target = target.to(get_device())
            
            output = network.forward_output(data)
            preds = output.argmax(dim=1)
            
            predictions.extend(preds.cpu().numpy())
            ground_truths.extend(target.cpu().numpy())

    predictions = np.array(predictions).flatten()
    ground_truths = np.array(ground_truths).flatten()
    accuracy = np.mean(predictions == ground_truths)
    
    network.train()
    
    if return_preds:
        return accuracy, predictions, ground_truths
    else:
        return accuracy

def phase_training_scl(network: Any, phase_epochs: int, optimizer: Any, train_loader: DataLoader, args: Namespace) -> Any:
    """
    Supervised Contrastive Loss を使ってネットワークを訓練します。
    """
    print("Supervised Contrastive Loss を用いた訓練を開始します。")
    loss_func = losses.SupConLoss(temperature=0.07)

    for epoch in range(phase_epochs):
        network.train()
        total_loss = 0
        for data, target, _ in train_loader:
            data, target = data.to(get_device()), target.to(get_device())
            optimizer.zero_grad()

            embeddings = network.forward(data)
            scl_loss = loss_func(embeddings, target)
            reg_loss = args.weight_decay * network.l2_loss()
            batch_loss = scl_loss + reg_loss
            
            batch_loss.backward()

            if hasattr(network, 'freeze_masks') and network.freeze_masks:
                network.reset_frozen_gradients()

            optimizer.step()
            total_loss += batch_loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{phase_epochs}, Average SCL Loss: {avg_loss:.4f}")

    return network