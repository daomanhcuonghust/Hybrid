import numpy as np
import pandas as pd
import main
from sklearn.preprocessing import MinMaxScaler

data_path = main.args.data_path

def get_data_array(file_path):
    location_df = pd.read_csv(file_path + "location.csv")
    stations = location_df['location'].values
    # location = location_df.values[:,1:]
    # location_ = location[:,[1,0]]
    
    list_dataset = []
    for i in stations:
        df = pd.read_csv(file_path  + f"{i}.csv")
        # df = df.fillna(method='ffill')
        # df = df.fillna(10)
        data = df.iloc[:,1:].astype(float).values
        label = df.iloc[:, 3].astype(float).values
        dataset = np.concatenate((data, label), axis=1)
        dataset = np.expand_dims(dataset,axis=1)
        list_dataset.append(dataset)
    list_dataset = np.concatenate(list_dataset,axis=1)
    return list_dataset,stations

def min_max_scaler(list_dataset, stations):
    list_scaler = []
    for i in range(stations.size):
        dataset = list_dataset[:, i, :]


