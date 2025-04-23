import torch

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Metric:
    """Computes the Mean Absolute Error (MAE)"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.outputs = []
        self.targets = []

    def update(self, outputs, targets):
        self.outputs.append(outputs.cpu().detach().numpy())
        self.targets.append(targets.cpu().detach().numpy())

    def compute(self):
        outputs = np.concatenate(self.outputs, axis=0)
        targets = np.concatenate(self.targets, axis=0)
        return np.mean(np.abs(outputs - targets))
