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
    network.eval()
    predictions, ground_truths = [], []
    with torch.no_grad():
        for data, target, _ in tqdm(data_loader, desc="Testing", leave=False):
            data, target = data.to(get_device()), target.to(get_device())
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
    return accuracy

def phase_training_scl(network: Any, phase_epochs: int, optimizer: Any, train_loader: DataLoader, args: Namespace) -> Any:
    print("Training with Combined Loss (SCL + CrossEntropy)...")
    scl_loss_func = losses.SupConLoss(temperature=0.07)
    ce_loss_func = nn.CrossEntropyLoss()
    
    # 2つの損失のバランスを調整
    scl_weight = 0.5
    ce_weight = 0.5

    for epoch in tqdm(range(phase_epochs), desc=f"Training Epochs", leave=True):
        network.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{phase_epochs}", leave=False)
        for data, target, _ in progress_bar:
            data, target = data.to(get_device()), target.to(get_device())
            optimizer.zero_grad()

            embeddings = network.forward(data)
            output = network.output_layer(embeddings)

            scl_loss = scl_loss_func(embeddings, target)
            ce_loss = ce_loss_func(output, target)
            reg_loss = args.weight_decay * network.l2_loss()

            batch_loss = (scl_weight * scl_loss) + (ce_weight * ce_loss) + reg_loss
            
            batch_loss.backward()

            if hasattr(network, 'freeze_masks') and network.freeze_masks and hasattr(network, 'reset_frozen_gradients'):
                network.reset_frozen_gradients()
            optimizer.step()
            
            total_loss += batch_loss.item()
            progress_bar.set_postfix({'loss': f'{batch_loss.item():.4f}', 'scl': f'{scl_loss.item():.2f}', 'ce': f'{ce_loss.item():.2f}'})

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{phase_epochs} Completed, Average Combined Loss: {avg_loss:.4f}")

    return network