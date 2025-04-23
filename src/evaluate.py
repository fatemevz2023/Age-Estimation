import torch
import pandas as pd
from models.resnet import ResNet50
from torch.utils.data import DataLoader
from dataset import AgeDataset
import torchvision.transforms as transforms
from utils import Metric

def evaluate(model, test_loader, device):
    model.eval()

    metric = Metric()
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            metric.update(outputs, targets)

    return metric.compute()

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load dataset
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = AgeDataset(csv_file='data/test_set.csv', transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Initialize model
    model = ResNet50(pretrained=True).to(device)
    model.load_state_dict(torch.load('model_epoch_20.pth'))  # Load the model you want to evaluate

    # Evaluate the model
    mae = evaluate(model, test_loader, device)
    print(f'Model MAE: {mae:.4f}')

if __name__ == '__main__':
    main()
