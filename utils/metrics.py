import torch
import numpy as np

def RMSE(predicts, labels, list_label_scaler_station):
    preds = predicts.cpu().detach().numpy()
    label = labels.cpu().detach().numpy()
    for station in range(len(list_label_scaler_station)):
        preds[:, station] = np.squeeze(list_label_scaler_station[station][0].inverse_transform(preds[:, station].reshape(-1, 1)), axis=1)
        label[:, station] = np.squeeze(list_label_scaler_station[station][0].inverse_transform(label[:, station].reshape(-1, 1)), axis=1)

    return np.sqrt(np.mean((preds - label)**2))

def MAE(predicts, labels, list_label_scaler_station):
    preds = predicts.cpu().detach().numpy()
    label = labels.cpu().detach().numpy()
    for station in range(len(list_label_scaler_station)):
        preds[:, station] = np.squeeze(list_label_scaler_station[station][0].inverse_transform(preds[:, station].reshape(-1, 1)), axis=1)
        label[:, station] = np.squeeze(list_label_scaler_station[station][0].inverse_transform(label[:, station].reshape(-1, 1)), axis=1)

    return np.mean(np.abs(preds - label))

def MAPE(predicts,labels, list_label_scaler_station):
    preds = predicts.cpu().detach().numpy()
    label = labels.cpu().detach().numpy()
    for station in range(len(list_label_scaler_station)):
        preds[:, station] = np.squeeze(list_label_scaler_station[station][0].inverse_transform(preds[:, station].reshape(-1, 1)), axis=1)
        label[:, station] = np.squeeze(list_label_scaler_station[station][0].inverse_transform(label[:, station].reshape(-1, 1)), axis=1)

    mape = np.mean(np.abs((label - preds)/label))*100

    return mape

def RMSE_1(predicts, labels, list_label_scaler_station):
    preds = predicts.cpu().detach().numpy()
    label = labels.cpu().detach().numpy()
    
    preds = list_label_scaler_station[0].inverse_transform(preds.reshape(-1, 1))
    label = list_label_scaler_station[0].inverse_transform(label.reshape(-1, 1))

    return np.sqrt(np.mean((preds - label)**2))

def MAE_1(predicts, labels, list_label_scaler_station):
    preds = predicts.cpu().detach().numpy()
    label = labels.cpu().detach().numpy()
    
    preds = list_label_scaler_station[0].inverse_transform(preds.reshape(-1, 1))
    label = list_label_scaler_station[0].inverse_transform(label.reshape(-1, 1))
    return np.mean(np.abs(preds - label))

def MAPE_1(predicts,labels, list_label_scaler_station):
    preds = predicts.cpu().detach().numpy()
    label = labels.cpu().detach().numpy()
    
    preds = list_label_scaler_station[0].inverse_transform(preds.reshape(-1, 1))
    label = list_label_scaler_station[0].inverse_transform(label.reshape(-1, 1))
    mape = np.mean(np.abs((label - preds)/label))*100

    return mape