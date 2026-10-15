from typing import Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

def conv3x3(in_ch, out_ch, stride=1):
    return nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)

def conv1x1(in_ch, out_ch, stride=1):
    return nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_ch, out_ch, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch, stride)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_ch, out_ch)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)

class HookedResNet18(nn.Module):
    HOOK_DIMS = {
        'layer1': 64,
        'layer2': 128,
        'layer3': 256,
        'layer4': 512,
    }

    def __init__(self, num_classes: int = 100, variant: str = "cifar",
                 detach_hooks: bool = True):
        super().__init__()
        self.variant = variant
        self.detach_hooks = detach_hooks
        self.hooks: Dict[str, torch.Tensor] = {}

        if variant == "cifar":
            self.stem = nn.Sequential(
                nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            )
        else:
            self.stem = nn.Sequential(
                nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(3, stride=2, padding=1),
            )

        self.layer1 = self._make_layer(64,  64,  2, stride=1)
        self.layer2 = self._make_layer(64,  128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        self._init_weights()

    @staticmethod
    def _make_layer(in_ch, out_ch, blocks, stride):
        layers = []
        downsample = None
        if stride != 1 or in_ch != out_ch:
            downsample = nn.Sequential(
                conv1x1(in_ch, out_ch, stride),
                nn.BatchNorm2d(out_ch),
            )
        layers.append(BasicBlock(in_ch, out_ch, stride, downsample))
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_ch, out_ch))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _gap(self, x: torch.Tensor, name: str) -> torch.Tensor:
        v = self.avgpool(x).flatten(1)
        self.hooks[name] = v.detach() if self.detach_hooks else v
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x); self._gap(x, 'layer1')
        x = self.layer2(x); self._gap(x, 'layer2')
        x = self.layer3(x); self._gap(x, 'layer3')
        x = self.layer4(x); self._gap(x, 'layer4')
        x = self.avgpool(x).flatten(1)
        return self.fc(x)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        self(x)
        return self.hooks['layer4']

    def get_all_hooks(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        self(x)
        return dict(self.hooks)

    def expand_classifier(self, new_total_classes: int):
        old = self.fc
        new_fc = nn.Linear(512, new_total_classes)
        new_fc.weight.data[:old.out_features] = old.weight.data
        new_fc.bias.data[:old.out_features]   = old.bias.data
        self.fc = new_fc

    def freeze_backbone(self):
        for name, param in self.named_parameters():
            if 'fc' not in name:
                param.requires_grad_(False)

    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad_(True)

def build_resnet18(dataset: str, num_classes: int,
                   detach_hooks: bool = True) -> HookedResNet18:
    variant = "cifar" if dataset == "cifar100" else "imagenet"
    return HookedResNet18(num_classes=num_classes, variant=variant,
                          detach_hooks=detach_hooks)
