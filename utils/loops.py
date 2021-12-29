import torch
from models.Hybrid import Hybrid
from utils.data import make_dataset
from utils.metrics import MAE, RMSE
from tqdm import tqdm
import os


def train(config, ):
    device =torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = Hybrid(config=config)
    model.to(device)

    train_dataloader, valid_dataloader = make_dataset(config.data_path, config.list_station, config.n_in, config.n_out, config.n_timestep, config.batch_size)

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters, lr=config.lr)
    
    best_rmse = 0
    bese_mae = 0
    for epoch in tqdm(range(1, config.epochs + 1)):
        running_loss = 0.0
        rmse = 0.0
        mae = 0.0
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
            rmse += RMSE(predict, label)
            mae += MAE(predict, label)
        
        running_loss /= iter_cnt
        rmse /= iter_cnt
        mae /= iter_cnt
        tqdm.write(f'[Epoch {epoch} Training] MSELoss: {running_loss}. RMSE: {rmse}. MAE: {mae}')


        with torch.no_grad():
            running_loss = 0.0
            iter_cnt = 0
            rmse = 0.0
            mae = 0.0
            model.eval()
            for (data, label) in valid_dataloader:
                iter_cnt += 1 
                data.to(device)
                label.to(device)

                predict = model(data)
                loss = loss_fn(predict, label)

                running_loss += loss
                rmse += RMSE(predict, label)
                mae += MAE(predict, label)
            running_loss /= iter_cnt
            rmse /= iter_cnt
            mae /= iter_cnt
            tqdm.write(f'[Epoch {epoch} Validation] MSELoss: {running_loss}. RMSE: {rmse}. MAE: {mae}')
            
            best_rmse = min(best_rmse, rmse)
            best_mae = min(best_mae, mae)

            count = 0
            if rmse > best_rmse and mae > best_mae:
                count += 1
                if(count == 15):
                    torch.save({'iter': epoch,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),},
                                os.path.join('checkpoints', "epoch"+str(epoch)+"_acc"+str(acc)+"_bacc"+str(bacc)+".pth"))
                    tqdm.write('Model saved.')
                    tqdm.write('Early stopping')
                    break
             

def test():
    pass