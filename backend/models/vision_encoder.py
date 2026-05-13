import torch
import torch.nn as nn
from torchvision import models

class VisionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = nn.Identity()

    def forward(self, x):
        return self.backbone(x)
