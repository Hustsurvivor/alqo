from create_training_file import create_training_file
from lero_train import training_pairwise
from lero_test import get_lero_dict
import argparse, os, json

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='stats')
args = parser.parse_args()


if __name__ == '__main__':
    latency_dir = f'results/{args.dataset}'
    os.makedirs(latency_dir, exist_ok=True)
    train_path = os.path.join('data/labeled_train_data', f'{args.dataset}_labeled.txt')
    train_file = f'data/lero_train_files/{args.dataset}_labeled_train_file'
    save_path = f"saved_models/{args.dataset}/stable_model"
    test_file = f"data/test/{args.dataset}_test.txt"

    os.makedirs(os.path.dirname(train_file), exist_ok=True)
    create_training_file(train_file, train_path)
    training_pairwise(None, save_path, train_file)
    get_lero_dict(save_path, test_file, latency_dir)
    with open(os.path.join(latency_dir, 'lero_dict.json'), 'r') as f:
        latency = json.load(f)['sum'] / 1000
        print(f'test latency:{latency}')
        
    with open(os.path.join(latency_dir, 'lero_latency'), 'w') as f:
        json.dump(latency, f, indent=4)


