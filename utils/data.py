import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader


def get_data_array(file_path):
    location_df = pd.read_csv(file_path + "location.csv")
    stations = location_df['location'].values
    # location = location_df.values[:,1:]
    # location_ = location[:,[1,0]]
    
    list_data = []
    list_label = []
    for i in stations:
        df = pd.read_csv(file_path  + f"{i}.csv")
        # df = df.fillna(method='ffill')
        # df = df.fillna(10)

        data = df.iloc[:,6:].astype(float).values
        label = df.iloc[:, 8].astype(float).values
        label = np.expand_dims(label, axis=1)
        # dataset = np.concatenate((data, label), axis=1)
        data_dataset = np.expand_dims(data, axis=0)
        label_dataset = np.expand_dims(label, axis=0)
        list_data.append(data_dataset)
        list_label.append(label_dataset)
    list_data_dataset = np.concatenate(list_data,axis=0)
    list_label_dataset = np.concatenate(list_label, axis=0)
    return list_data_dataset, list_label_dataset, stations


def min_max_scaler(list_dataset, stations):
    list_scaler = []
    for i in range(stations.size):
        scaleri = []
        for j in range(list_dataset.shape[2]):
            scalerj = MinMaxScaler()
            list_dataset[i, :, j:j+1] = scalerj.fit_transform(list_dataset[i, :, j:j+1])
            scaleri.append(scalerj)
        list_scaler.append(scaleri)

    
    return list_dataset, list_scaler


def my_series_to_supervised(data, n_in=1, n_out=1, n_timestep=1, dropnan=True):
	"""
	Frame a time series as a supervised learning dataset.
	Arguments:
		data: Sequence of observations as a list or NumPy array.
		n_in: Number of lag observations as input (X).
		n_out: Number of observations as output (y).
		dropnan: Boolean whether or not to drop rows with NaN values.
	Returns:
		Pandas DataFrame of series framed for supervised learning.
	"""
	n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in*n_timestep, 0, -n_timestep):
		cols.append(df.shift(i))
		names += [('Time%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('Time%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('Time%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = pd.concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg


# def make_list_dataframe(list_dataset, list_station,  n_in, n_out, n_timestep):
#     list_dataframe = []
#     for i in list_station:
#         dataset_i = list_dataset[i, :, :]
#         dataframe_i = my_series_to_supervised(dataset_i, n_in=n_in, n_out=n_out, n_timestep=n_timestep)
#         list_dataframe.append(dataframe_i)
    
#     return list_dataframe


def make_dataset(data_path, list_station, n_in, n_out, n_timestep, batch_size):
    list_data_dataset, list_label_dataset, stations = get_data_array(data_path)
    list_data_dataset, list_data_scaler = min_max_scaler(list_data_dataset, stations)
    list_label_dataset, list_label_scaler = min_max_scaler(list_label_dataset, stations)

    list_data_dataframe = []
    list_label_dataframe = []
    list_label_scaler_station = []
    for i in list_station:
        list_data_dataframe.append(list_data_dataset[i, :, :].reshape(list_data_dataset.shape[1], -1))
        list_label_dataframe.append(list_label_dataset[i, :, :].reshape(list_label_dataset.shape[1], -1))
        list_label_scaler_station.append(list_label_scaler[i])
    data_frame = np.concatenate(list_data_dataframe, axis=1)
    label_frame = np.concatenate(list_label_dataframe, axis=1)
    import pdb;pdb.set_trace()

    # list_dataframe = make_list_dataframe(list_dataset, list_station, n_in, n_out, n_timestep)
    data_dataframe = my_series_to_supervised(data_frame, n_in, 1, n_timestep)
    label_dataframe = my_series_to_supervised(label_frame, n_in, n_out, n_timestep)
    data = data_dataframe.iloc[:, :-1*len(list_station)*14].astype('float32').values
    label = label_dataframe.iloc[:, -len(list_station):].astype('float32').values
    # list_data = []
    # list_label = []
    # for dataframe in list_dataframe:
    #     list_data.append(dataframe.iloc[:, :-1])
    #     list_label.append(dataframe.iloc[:, -1])
    
    # data = pd.concat(list_data, axis=1).astype('float32').values
    # label = pd.concat(list_label, axis=1).astype('float32').values

    data_train, data_valid_test, label_train, label_valid_test = train_test_split(data, label, test_size=0.4, random_state=42)
    data_valid, data_test, label_valid, label_test = train_test_split(data_valid_test, label_valid_test, test_size=0.5, random_state=42)
    # data_train = np.expand_dims(data_train, axis=1)
    # data_valid = np.expand_dims(data_valid, axis=1)
    
    #create tensor datasets
    train_data = TensorDataset(torch.from_numpy(data_train), torch.from_numpy(label_train))
    valid_data = TensorDataset(torch.from_numpy(data_valid), torch.from_numpy(label_valid))

    #dataloaders
    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)
    valid_dataloader = DataLoader(valid_data, shuffle=True, batch_size=batch_size, drop_last=True)

    return train_dataloader, valid_dataloader, (data_test, label_test), list_label_scaler_station

