import torch.nn as nn
from Source.helper import SparseConv2d, SparseLinear, BatchNorm2Custom, get_device, SparseOutput
from argparse import Namespace
from typing import List, Any
import torch
import numpy as np
from .synapse_operations import IMMATURE, MATURE_BASE_RANK

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, prev_layer_count=0):
        super(BasicBlock, self).__init__()
        self.conv1 = SparseConv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False, layer_name=f"conv_early_{prev_layer_count + 1}")
        self.bn1 = BatchNorm2Custom(out_channels, layer_name=f"conv_early_{prev_layer_count + 1}")
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = SparseConv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False, layer_name=f"conv_early_{prev_layer_count + 2}")
        self.bn2 = BatchNorm2Custom(out_channels, layer_name=f"conv_early_{prev_layer_count + 2}")
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                SparseConv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False, layer_name=f"conv_early_{prev_layer_count + 0}->conv_early_{prev_layer_count + 2}"),
                BatchNorm2Custom(out_channels, layer_name=f"conv_early_{prev_layer_count + 2}")
            )
    def forward(self, x):
        out1 = self.relu(self.bn1(self.conv1(x)))
        out2 = self.bn2(self.conv2(out1))
        out = out2 + self.shortcut(x)
        out = self.relu(out)
        return out, torch.relu(out2), torch.relu(out1)

class ResNet18(nn.Module):
    def __init__(self, args: Namespace, input_size: int, output_size: int):
        super(ResNet18, self).__init__()
        self.args = args
        self.c = args.resnet_multiplier
        self.penultimate_layer_size = int(512 * self.c)
        self.input_size = input_size
        self.output_size = output_size
        
        self.layers_config = [(3, "conv_early_0"), (int(64 * self.c), "conv_early_1"), (int(64 * self.c), "conv_early_2"), (int(64 * self.c), "conv_early_3"), (int(64 * self.c), "conv_early_4"), (int(64 * self.c), "conv_early_5"), (int(128 * self.c), "conv_early_6"), (int(128 * self.c), "conv_early_7"), (int(128 * self.c), "conv_early_8"), (int(128 * self.c), "conv_early_9"), (int(256 * self.c), "conv_early_10"), (int(256 * self.c), "conv_early_11"), (int(256 * self.c), "conv_early_12"), (int(256 * self.c), "conv_early_13"), (int(512 * self.c), "conv_early_14"), (int(512 * self.c), "conv_early_15"), (int(512 * self.c), "conv_early_16"), (int(512 * self.c), "conv_early_17")]
        
        self.conv1 = SparseConv2d(input_size, int(64 * self.c), kernel_size=3, stride=1, padding=1, bias=False, layer_name="conv_early_1")
        self.bn1 = BatchNorm2Custom(int(64 * self.c), layer_name="conv_early_1")
        self.relu = nn.ReLU(inplace=True)
        self.block1_1 = BasicBlock(int(64 * self.c), int(64 * self.c), 1, 1)
        self.block1_2 = BasicBlock(int(64 * self.c), int(64 * self.c), 1, 3)
        self.block2_1 = BasicBlock(int(64 * self.c), int(128 * self.c), 2, 5)
        self.block2_2 = BasicBlock(int(128 * self.c), int(128 * self.c), 1, 7)
        self.block3_1 = BasicBlock(int(128 * self.c), int(256 * self.c), 2, 9)
        self.block3_2 = BasicBlock(int(256 * self.c), int(256 * self.c), 1, 11)
        self.block4_1 = BasicBlock(int(256 * self.c), int(512 * self.c), 2, 13)
        self.block4_2 = BasicBlock(int(512 * self.c), int(512 * self.c), 1, 15)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.output_layer = SparseOutput(int(512 * self.c), output_size, layer_name="output")

        self.immature_pool_ratio = 0.1
        self.unit_ranks = [(np.array([999]*self.input_size, dtype=int), "conv_early_0")]
        
        self.ranked_modules = []
        for m in self.modules():
            if hasattr(m, 'layer_name') and isinstance(m, (SparseConv2d, SparseLinear, SparseOutput)):
                self.ranked_modules.append(m)
        
        # ranked_modulesを使ってunit_ranksを生成
        for module in self.ranked_modules:
            num_units = module.weight.shape[0]
            self.unit_ranks.append((np.full(num_units, IMMATURE, dtype=int), module.layer_name))

        self.update_neuron_state_lists()
        self.freeze_masks = []
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.constant_(m.bias, 0)

    def update_neuron_state_lists(self):
        self.immature_neurons = [list(np.where(ranks == IMMATURE)[0]) for ranks, _ in self.unit_ranks]
        self.transitional_neurons = [list(np.where(ranks == 0)[0]) for ranks, _ in self.unit_ranks]
        self.mature_neurons = [list(np.where(ranks >= MATURE_BASE_RANK)[0]) for ranks, _ in self.unit_ranks]

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x, _, _ = self.block1_1(x); x, _, _ = self.block1_2(x)
        x, _, _ = self.block2_1(x); x, _, _ = self.block2_2(x)
        x, _, _ = self.block3_1(x); x, _, _ = self.block3_2(x)
        x, _, _ = self.block4_1(x); x, _, _ = self.block4_2(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    def forward_output(self, x):
        x = self.forward(x)
        x = self.output_layer(x)
        return x

    def get_activations(self, x: torch.Tensor, return_output=False) -> List[Any]:
        activations = [x.detach().cpu()]
        x = self.conv1(x); activations.append(x.detach().cpu())
        x = self.bn1(x); activations.append(x.detach().cpu())
        x = self.relu(x); activations.append(x.detach().cpu())
        x, x2, x1 = self.block1_1(x); activations.extend([x1, x2, x])
        x, x2, x1 = self.block1_2(x); activations.extend([x1, x2, x])
        x, x2, x1 = self.block2_1(x); activations.extend([x1, x2, x])
        x, x2, x1 = self.block2_2(x); activations.extend([x1, x2, x])
        x, x2, x1 = self.block3_1(x); activations.extend([x1, x2, x])
        x, x2, x1 = self.block3_2(x); activations.extend([x1, x2, x])
        x, x2, x1 = self.block4_1(x); activations.extend([x1, x2, x])
        x, x2, x1 = self.block4_2(x); activations.extend([x1, x2, x])
        x = self.avgpool(x); activations.append(x.detach().cpu())
        x = x.view(x.size(0), -1); activations.append(x.detach().cpu())
        output = self.output_layer(x); activations.append(output.detach().cpu())
        if return_output: return output, activations
        return activations

    def l2_loss(self):
        reg_loss = 0.0
        for module in self.modules():
            if isinstance(module, (SparseLinear, SparseConv2d, SparseOutput)):
                if hasattr(module, 'weight_mask'):
                    reg_loss += torch.sum((module.weight * module.weight_mask)**2)
        return reg_loss