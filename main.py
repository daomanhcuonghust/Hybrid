import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='dataset/', help='PM2.5 dataset path.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--num_station', type=int, default=3, help='Num station.')
    parser.add_argument('--epochs', type=int, default=100, help='Epochs.')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    