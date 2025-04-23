import torch
import torch.nn as nn
from torchvision import models

class ResNet50(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet50, self).__init__()
        resnet = models.resnet50(pretrained=pretrained)
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])  # Remove the final fully connected layer
        self.fc = nn.Linear(resnet.fc.in_features, 1)  # Add a new fully connected layer for age prediction

    def forward(self, x):
        x = self.resnet(x)
        x = torch.flatten(x, 1)  # Flatten the output
        x = self.fc(x)  # Predict age
        return x
