from argparse import Namespace
import torch
import torch.nn as nn
import numpy as np
from typing import List, Any
from Source.helper import SparseConv2d, SparseLinear, SparseOutput
from Source.resnet18 import ResNet18
from .synapse_operations import IMMATURE, MATURE_BASE_RANK

def get_backbone(args: Namespace, input_size: int, output_size: int) -> nn.Module:
    if args.model == "VGG11_SLIM":
        return VGG11_SLIM(args, input_size, output_size)
    elif args.model == "ResNet18":
        # ResNet18も同様にクリーンアップが必要です
        return ResNet18(args, input_size, output_size)
    else:
        raise Exception(f"Model {args.model} is not defined!")

class VGG11_SLIM(nn.Module):
    def __init__(self, args: Namespace, input_size: int, output_size: int) -> None:
        super(VGG11_SLIM, self).__init__()
        self.args = args
        self.input_size = input_size
        self.output_size = output_size
        
        if args.dataset in ["CIFAR10", "CIFAR100"]:
            self.conv2lin_size = 256 * 1 * 1 # 32x32画像は5回のMaxPoolで1x1になる
        elif args.dataset == "TinyImagenet":
            self.conv2lin_size = 256 * 2 * 2 # 64x64画像は5回のMaxPoolで2x2になる
        else:
            self.conv2lin_size = 256 * 1 * 1 # 仮

        # 層のリストとunit_ranksリストを並行して作成し、ズレをなくす
        self.features_list = nn.ModuleList()
        self.classifier_list = nn.ModuleList()
        self.ranked_modules = []
        self.unit_ranks = [(np.array([999]*self.input_size, dtype=int), "input")]

        # --- 層定義とランク作成を同時に行う ---
        in_channels = 3
        cfg = [32, 64, 'M', 128, 128, 'M', 256, 256, 'M', 256, 256, 'M', 256, 256, 'M']
        for i, v in enumerate(cfg):
            if v == 'M':
                self.features_list.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                conv2d = SparseConv2d(in_channels, v, kernel_size=3, padding=1, layer_name=f"conv_{i+1}")
                self.features_list.append(conv2d)
                self.features_list.append(nn.ReLU(inplace=True))
                self.ranked_modules.append(conv2d)
                self.unit_ranks.append((np.full(v, IMMATURE, dtype=int), f"conv_{i+1}"))
                in_channels = v
        
        linear1 = SparseLinear(self.conv2lin_size, 1024, layer_name="linear_1")
        self.classifier_list.extend([linear1, nn.ReLU(inplace=True)])
        self.ranked_modules.append(linear1)
        self.unit_ranks.append((np.full(1024, IMMATURE, dtype=int), "linear_1"))

        linear2 = SparseLinear(1024, 1024, layer_name="linear_2")
        self.classifier_list.append(linear2)
        self.ranked_modules.append(linear2)
        self.unit_ranks.append((np.full(1024, IMMATURE, dtype=int), "linear_2"))
        
        self.output_layer = SparseOutput(1024, output_size, layer_name="output")
        self.ranked_modules.append(self.output_layer)
        self.unit_ranks.append((np.full(output_size, IMMATURE, dtype=int), "output"))

        self.immature_pool_ratio = 0.1
        self.update_neuron_state_lists()
        self.freeze_masks = []
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.constant_(m.bias, 0)

    def update_neuron_state_lists(self):
        self.immature_neurons = [list(np.where(ranks == IMMATURE)[0]) for ranks, _ in self.unit_ranks]
        self.transitional_neurons = [list(np.where(ranks == 0)[0]) for ranks, _ in self.unit_ranks]
        self.mature_neurons = [list(np.where(ranks >= MATURE_BASE_RANK)[0]) for ranks, _ in self.unit_ranks]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.features_list: x = layer(x)
        x = x.view(x.size(0), -1)
        for layer in self.classifier_list: x = layer(x)
        return x

    def forward_output(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward(x)
        x = self.output_layer(x)
        return x
    
    def get_activations(self, x: torch.Tensor, return_output=False) -> List[Any]:
        activations = [x.detach().cpu()]
        current_x = x
        for module in self.features_list:
            current_x = module(current_x)
            activations.append(current_x.detach().cpu())
        current_x = current_x.view(current_x.size(0), -1)
        for module in self.classifier_list:
            current_x = module(current_x)
            activations.append(current_x.detach().cpu())
        output = self.output_layer(current_x)
        activations.append(output.detach().cpu())
        if return_output: return output, activations
        return activations
        
    def l2_loss(self):
        reg_loss = 0.0
        for module in self.modules():
            if isinstance(module, (SparseLinear, SparseConv2d, SparseOutput)):
                if hasattr(module, 'weight_mask'):
                    reg_loss += torch.sum((module.weight * module.weight_mask)**2)
        return reg_loss