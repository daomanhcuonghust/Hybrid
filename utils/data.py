import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader
import os

def get_data_array(file_path, config):
    file_gauge = file_path + 'gauges_processed/'
    file_location = file_path + 'location.csv' 
    nan_station = config['data']['nan_station']

    # list_station = list(set([ stat.split('.csv')[0] for stat in os.listdir(file_gauge)]) - set(nan_station))
    list_station = [ stat.split('.csv')[0] for stat in os.listdir(file_gauge) if stat.split('.csv')[0] not in nan_station]
    list_input_ft = config['data']['input_features']
    target_output_ft = config['data']['target_features'] 

    scaler_data = MinMaxScaler()
    scaler_label = MinMaxScaler()
    list_data = []
    list_label = []
    # print(list_station)
    for stat in list_station:
      # list station data
      df = pd.read_csv(file_gauge  + f"{stat}.csv")
      data = df[list_input_ft]
      label = df[target_output_ft]

      data = data.astype(np.float32).values
      label = label.astype(np.float32).values
      # arr = np.expand_dims(arr,axis=1) # date, station, feat
      list_data.append(data)
      list_label.append(label)

    # list_arr = np.concatenate(list_arr,axis=1)   
    num_ft = list_data[0].shape[-1]  #14

    data = np.concatenate(list_data, axis=0) # 8642 * 20, 14
    label = np.concatenate(list_label, axis=0)
    scaled_data = scaler_data.fit_transform(data)
    scaled_label = scaler_label.fit_transform(label)
    # transformed = scaler.transform(list_arr)
    transformed = scaled_data.copy()
    transformed_data = transformed.reshape(len(list_station), -1, num_ft) #  20, 8642, 14
    transformed_data = np.swapaxes(transformed_data, 0,1) # 8642, 20, 14
    
    transformed = scaled_label.copy()
    transformed_label = transformed.reshape(len(list_station), -1) #  20, 8642, 14
    transformed_label = np.swapaxes(transformed_label, 0,1)

    # return transformed_data, location_, list_station
    return transformed_data, transformed_label, scaler_label, list_station
    




def make_dataset(data_path, n_in, n_out, n_timestep, batch_size, config):
    list_data_dataset, list_label_dataset, scaler_label, list_station = get_data_array(data_path, config)
    
    data = []
    for i in range (n_in,0,-1):
        if i == n_in:
            datai = list_data_dataset[i-1:, :, :]
        else: 
            datai = list_data_dataset[i-1: i-n_in, :, :]
        data.append(np.expand_dims(datai, axis=3))
    
    data = np.concatenate(data, axis=3)
    label =list_label_dataset[n_in-1:, :]
    

    data_train, data_valid_test, label_train, label_valid_test = train_test_split(data, label, test_size=0.2, random_state=42)
    data_valid, data_test, label_valid, label_test = train_test_split(data_valid_test, label_valid_test, test_size=0.5, random_state=42)
    # data_train = np.expand_dims(data_train, axis=1)
    # data_valid = np.expand_dims(data_valid, axis=1)
    
    #create tensor datasets
    train_data = TensorDataset(torch.from_numpy(data_train), torch.from_numpy(label_train))
    valid_data = TensorDataset(torch.from_numpy(data_valid), torch.from_numpy(label_valid))

    #dataloaders
    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)
    valid_dataloader = DataLoader(valid_data, shuffle=True, batch_size=batch_size, drop_last=True)

    return train_dataloader, valid_dataloader, (data_test, label_test), scaler_label, list_station

