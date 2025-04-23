import torch.nn as nn
from torchvision import models

class AgeEstimationModel(nn.Module):
    def __init__(self):
        super(AgeEstimationModel, self).__init__()
        self.base_model = models.resnet50(pretrained=True)
        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.base_model(x)
