import torch

def RMSE(predicts, labels):
    return torch.sqrt(torch.mean((predicts - labels)**2))

def MAE(predicts, labels):
    return torch.mean(torch.abs(predicts - labels))
