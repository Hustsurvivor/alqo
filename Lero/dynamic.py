from create_training_file import create_training_file
from lero_train import training_pairwise
from lero_test import get_lero_dict
import argparse, os, json

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='stats_simplified')
parser.add_argument('--num_batch', type=int, default=5)
args = parser.parse_args()

res_dir = '../../results_dynamic_new'
with open(f'{res_dir}/{args.dataset}/tmp_1/train_1', 'r') as f:
    line = f.readline()
with open(f'{res_dir}/{args.dataset}/tmp_1/train_0', 'w') as f:
    f.write(line)

with open(f'{res_dir}/{args.dataset}/train_paths', 'r') as f:
    train_paths = json.load(f)
train_paths = [os.path.join('../..', p)for p in train_paths]
with open(f'{res_dir}/{args.dataset}/test_paths', 'r') as f:
    test_paths = json.load(f)
test_paths = [os.path.join('../..', p)for p in test_paths]
if __name__ == '__main__':
    lero_latency_list = []
    latency_dir = f'dynamic_results/{args.dataset}'
    os.makedirs(latency_dir, exist_ok=True)
    for i in range(1, args.num_batch+1):
        if i == 1:
            train_path = f'{res_dir}/{args.dataset}/tmp_1/train_0'
            train_file = f'dynamic_data/{args.dataset}/train_0'
            model_name = f"saved_models/dynamic/{args.dataset}/model_0"
            test_file = test_paths[i-1]
            os.makedirs(os.path.dirname(train_file), exist_ok=True)
            create_training_file(train_file, train_path)
            training_pairwise(None, model_name, train_file)
            get_lero_dict(model_name, test_file, latency_dir)
            with open(os.path.join(latency_dir, 'lero_dict.json'), 'r') as f:
                lero_latency_list.append(json.load(f)['sum'])

        if i!= args.num_batch:
            train_path = train_paths[i-1]
            train_file = f'dynamic_data/{args.dataset}/train_{i}'
            model_name = f"saved_models/dynamic/{args.dataset}/model_{i}"
            test_file = test_paths[i]
            os.makedirs(os.path.dirname(train_file), exist_ok=True)
            create_training_file(train_file, train_path)
            training_pairwise(None, model_name, train_file)
            get_lero_dict(model_name, test_file, latency_dir)
            with open(os.path.join(latency_dir, 'lero_dict.json'), 'r') as f:
                lero_latency_list.append(json.load(f)['sum'])
        
    with open(os.path.join(latency_dir, 'lero_dynamic_latency_list'), 'w') as f:
        json.dump(lero_latency_list, f, indent=4)


