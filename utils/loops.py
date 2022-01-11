import torch
from torch.optim import optimizer
# from torch.utils import tensorboard
from models.Hybrid import Hybrid
from utils.data import make_dataset
from utils.metrics import MAE, MAE_1, MAPE, MAPE_1, RMSE, RMSE_1
from tqdm import tqdm
import os
from torch.utils.tensorboard import SummaryWriter
# from torchsummary import summary
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd


def train(config):
    writer = SummaryWriter()
    device =torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = Hybrid(config=config)
    model.to(device)
    # summary(model)
    
    train_dataloader, valid_dataloader,(data_test, label_test), list_label_scaler_station = make_dataset(config.get('data_path'), config.get('list_station'), config.get('n_in'), config.get('n_out'), config.get('n_timestep'), config.get('batch_size'))

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.get('lr'))
    
    best_rmse = 1e10
    best_mae = 1e10
    best_mape = 1e10
    best_running_loss = 1e10
    for epoch in tqdm(range(1, config.get('epochs') + 1)):
        running_loss = 0.0
        rmse = 0.0
        mae = 0.0
        mape = 0.0
        iter_cnt = 0
        model.train()

        for (data, label) in train_dataloader:
            iter_cnt += 1
            optimizer.zero_grad()

            data = data.to(device)
            label = label.to(device)

            predict = model(data)
            loss = loss_fn(predict, label)

            loss.backward()
            optimizer.step()

            running_loss += loss
            rmse += RMSE(predict, label, list_label_scaler_station)
            mae += MAE(predict, label, list_label_scaler_station)
            mape += MAPE(predict, label, list_label_scaler_station)
        
        running_loss /= iter_cnt
        rmse /= iter_cnt
        mae /= iter_cnt
        mape /= iter_cnt
        tqdm.write(f'[Epoch {epoch} Training] MSELoss: {running_loss}. RMSE: {rmse}. MAE: {mae}. MAPE: {mape}')
        writer.add_scalar("Train_Loss", running_loss, epoch)
        writer.add_scalar("Train_RMSE", rmse, epoch)
        writer.add_scalar("Train_MAE", mae, epoch)
        writer.add_scalar("Train_MAPE", mape, epoch)


        with torch.no_grad():
            running_loss = 0.0
            iter_cnt = 0
            rmse = 0.0
            mae = 0.0
            mape = 0.0
            model.eval()
            for (data, label) in valid_dataloader:
                iter_cnt += 1 
                data = data.to(device)
                label = label.to(device)

                predict = model(data)
                loss = loss_fn(predict, label)

                running_loss += loss
                rmse += RMSE(predict, label, list_label_scaler_station)
                mae += MAE(predict, label, list_label_scaler_station)
                mape += MAPE(predict, label, list_label_scaler_station)
            running_loss /= iter_cnt
            rmse /= iter_cnt
            mae /= iter_cnt
            mape /= iter_cnt
            tqdm.write(f'[Epoch {epoch} Validation] MSELoss: {running_loss}. RMSE: {rmse}. MAE: {mae}. MAPE: {mape}')
            
            writer.add_scalar("Valid_Loss", running_loss, epoch)
            writer.add_scalar("Valid_RMSE", rmse, epoch)
            writer.add_scalar("Valid_MAE", mae, epoch)
            writer.add_scalar("Valid_MAPE", mape, epoch)

            best_rmse = min(best_rmse, rmse)
            best_mae = min(best_mae, mae)
            best_mape = min(best_mape, mape)
            best_running_loss = min(best_running_loss, running_loss)
            count = 0
            if running_loss > best_running_loss:
                count += 1
                if(count == 20):
                    torch.save({'iter': epoch,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),},
                                os.path.join('checkpoints', "epoch"+str(epoch)+"_rmse"+str(best_rmse)+"_mae"+str(best_mae)+"_mape"+str(best_mape)+".pth"))
                    tqdm.write('Model saved.')
                    tqdm.write('Early stopping')
                    writer.close()
                    return data_test, label_test,list_label_scaler_station, os.path.join('checkpoints', "epoch"+str(epoch)+"_rmse"+str(best_rmse)+"_mae"+str(best_mae)+"_mape"+str(best_mape)+".pth")
    

    torch.save({'iter': config.get('epochs'),
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),},
                 os.path.join('checkpoints', "epoch"+str(config.get('epochs'))+"_rmse"+str(best_rmse)+"_mae"+str(best_mae)+"_mape"+str(best_mape)+".pth"))
    tqdm.write('Model saved.')
    writer.close()
    return data_test, label_test,list_label_scaler_station, os.path.join('checkpoints', "epoch"+str(epoch)+"_rmse"+str(best_rmse)+"_mae"+str(best_mae)+"_mape"+str(best_mape)+".pth")
                  
def test(data_test, label_test,list_label_scaler_station, path, config):
    device =torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Hybrid(config=config)
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    loss_fn = torch.nn.MSELoss()

    location_df = pd.read_csv(config.get('data_path') + "location.csv")
    stations = location_df['location'].values

    test_data = TensorDataset(torch.from_numpy(data_test), torch.from_numpy(label_test))
    test_dataloader = DataLoader(test_data, shuffle=True, batch_size=config.get('batch_size'), drop_last=True)
    with torch.no_grad():
            iter_cnt = 0
            running_loss = 0
            rmse = np.zeros(config.get('num_station'))
            mae = np.zeros(config.get('num_station'))
            mape = np.zeros(config.get('num_station'))
            model.eval()
            for (data, label) in test_dataloader:
                iter_cnt += 1 
                data = data.to(device)
                label = label.to(device)

                predict = model(data)
                loss = loss_fn(predict, label)

                running_loss += loss
                for station in range(config.get('num_station')):
                    rmse[station] += RMSE_1(predict[:, station], label[:, station], list_label_scaler_station[station])
                    mae[station] += MAE_1(predict[:, station], label[:, station], list_label_scaler_station[station])
                    mape[station] += MAPE_1(predict[:, station], label[:, station], list_label_scaler_station[station])
            running_loss = running_loss/iter_cnt
            rmse = rmse/iter_cnt
            mae = mae/iter_cnt
            mape = mape/iter_cnt

            with open('log/result.csv','w') as f:
                f.write('Test result')
                for i in range(config.get('num_station')):
                    name = stations[config.get('list_station')[i]]
                    f.write(f'Station: {name}, RMSE: {rmse[i]}, MAE: {mae[i]}, MAPE: {mape[i]}' )
           

    data_test = torch.from_numpy(data_test[:200, :]).to(device)
    label_test = torch.from_numpy(label_test[:200, :]).to(device)


    predicts = model(data_test)
    

    predict = predicts.cpu().detach().numpy()
    label = label_test.cpu().detach().numpy()
    for station in range(len(list_label_scaler_station)):
        predict[:, station] = np.squeeze(list_label_scaler_station[station][0].inverse_transform(predict[:, station].reshape(-1, 1)), axis=1)
        label[:, station] = np.squeeze(list_label_scaler_station[station][0].inverse_transform(label[:, station].reshape(-1, 1)), axis=1)


    
    for i in range(config.get('num_station')):
        name = stations[config.get('list_station')[i]]
        plt.plot(label[:, i], label='label')
        plt.plot(predict[:, i], label='predict')
        plt.title(f'Tram {name}')
        plt.legend()
        plt.savefig(f'log/image_result/{name}.png')
        plt.clf()





    