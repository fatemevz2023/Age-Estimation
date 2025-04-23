import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from models.resnet import ResNet50
from utils import AverageMeter, Metric
from dataset import AgeDataset

def train(model, train_loader, criterion, optimizer, epoch, device):
    model.train()

    losses = AverageMeter()
    metric = Metric()

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)

        # Compute loss
        loss = criterion(outputs, targets)
        losses.update(loss.item(), inputs.size(0))

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Update metrics
        metric.update(outputs, targets)

        if batch_idx % 100 == 0:
            print(f'Epoch [{epoch}]: Loss {losses.avg:.4f}, MAE {metric.compute():.4f}')

    return losses.avg, metric.compute()

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load dataset
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = AgeDataset(csv_file='data/train_set.csv', transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

    # Initialize model, criterion, and optimizer
    model = ResNet50(pretrained=True).to(device)
    criterion = torch.nn.L1Loss().to(device)  # MAE loss
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Training loop
    num_epochs = 20
    for epoch in range(1, num_epochs + 1):
        train_loss, train_mae = train(model, train_loader, criterion, optimizer, epoch, device)
        
        # Save model
        if epoch % 10 == 0:
            torch.save(model.state_dict(), f'model_epoch_{epoch}.pth')
            print(f'Model saved at epoch {epoch}')

if __name__ == '__main__':
    main()
