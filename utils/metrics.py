import torch
import numpy as np

def RMSE(predicts, labels, scaler_label):
    preds = predicts.cpu().detach().numpy()
    label = labels.cpu().detach().numpy()
    preds = scaler_label.inverse_transform(preds)
    label = scaler_label.inverse_transform(label)

    return np.sqrt(np.mean((preds - label)**2))

def MAE(predicts, labels, scaler_label):
    preds = predicts.cpu().detach().numpy()
    label = labels.cpu().detach().numpy()
    preds = scaler_label.inverse_transform(preds)
    label = scaler_label.inverse_transform(label)

    return np.mean(np.abs(preds - label))

def MAPE(predicts,labels, scaler_label):
    preds = predicts.cpu().detach().numpy()
    label = labels.cpu().detach().numpy()
    preds = scaler_label.inverse_transform(preds)
    label = scaler_label.inverse_transform(label)

    mape = np.mean(np.abs((label - preds)/label))*100

    return mape

def RMSE_1(predicts, labels, scaler_label):
    preds = predicts.cpu().detach().numpy()
    label = labels.cpu().detach().numpy()
    
    preds = scaler_label.inverse_transform(preds.reshape(-1, 1))
    label = scaler_label.inverse_transform(label.reshape(-1, 1))

    return np.sqrt(np.mean((preds - label)**2))

def MAE_1(predicts, labels, scaler_label):
    preds = predicts.cpu().detach().numpy()
    label = labels.cpu().detach().numpy()
    
    preds = scaler_label.inverse_transform(preds.reshape(-1, 1))
    label = scaler_label.inverse_transform(label.reshape(-1, 1))
    return np.mean(np.abs(preds - label))

def MAPE_1(predicts,labels, scaler_label):
    preds = predicts.cpu().detach().numpy()
    label = labels.cpu().detach().numpy()
    
    preds = scaler_label.inverse_transform(preds.reshape(-1, 1))
    label = scaler_label.inverse_transform(label.reshape(-1, 1))
    mape = np.mean(np.abs((label - preds)/label))*100

    return mape