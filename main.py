import argparse
from utils.loops import train
import yaml

def seed():
    pass

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='dataset/', help='PM2.5 dataset path.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--num_station', type=int, default=3, help='Num station.')
    parser.add_argument('--epochs', type=int, default=100, help='Epochs.')
    
    return parser.parse_args()

if __name__ == '__main__':
    seed()
    args = parse_args()

    with open("config/hyperparameter.yaml") as f:
        config = yaml.safe_load(f)
    for arg in args:
        config.arg = args.arg
    with open('config/hyperparameter.yaml', 'w') as f:
        yaml.dump(config, f)
    train(config=config)
    