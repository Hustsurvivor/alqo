from model import LeroModel
import json, os
import numpy as np
import argparse
import logging
logger = logging.getLogger('my_logger')

def load_model(model_path=None):
    lero_model = LeroModel(None)
    if model_path is not None:
        lero_model.load(model_path)
    
    return lero_model

def load_plans(plan_path):
    plans_list = []
    with open(plan_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            plans = line.strip().split('#####')[1:]
            plans_list.append(plans)
    
    return plans_list

def get_lero_dict(model_path, plan_path, dir):
    # model_path = 'lero_model_tpch_10g_25%'
    # plan_path = '/home/lgn/source/LearnedQO/data/tpch-10g/plan/combined_tpch_10g_train_processed_25%.txt'
    
    lero_model = load_model(model_path)
    plans_list = load_plans(plan_path)
    
    y_list = []
    for plans in plans_list:
        # for plan in plans:
        local_features, _ = lero_model._feature_generator.transform(plans)
        y = lero_model.predict(local_features)
        y_list.append(y)
    
    
    choice = [np.argmin(row) for row in y_list]
    
    lero_dict = {}
    sum = 0.0
    id = 0
    for y in choice:
        plan = json.loads(plans_list[id][y])
        lero_dict['q'+str(id)] = plan[0]['Execution Time']/1000 
        id += 1
        sum += plan[0]['Execution Time']/1000 
    
    lero_dict['sum'] = sum
    
    with open(os.path.join(dir,'lero_dict.json'), 'w') as f:
        json.dump(lero_dict, f, indent=4)

def get_pg_dict(plan_path, dir):
    # plan_path = '/home/lgn/source/LearnedQO/data/tpch-10g/plan/combined_tpch_10g_train_processed_25%.txt'
    plans_list = load_plans(plan_path)
    
    # choice = [ 0 for i in range(30) ]
    pg_dict = {}
    sum = 0.0
    
    for id in range(len(plans_list)):
        plan = json.loads(plans_list[id][0])
        pg_dict['q'+str(id)] = plan[0]['Execution Time']/1000  
        sum += plan[0]['Execution Time']/1000
        
    # for y in choice:
    #     plan = json.loads(plans_list[id][0])
    #     pg_dict['q'+str(id)] = plan[0]['Execution Time']/1000  
    #     id += 1
    #     sum += plan[0]['Execution Time']/1000 

    pg_dict['sum'] = sum
    
    with open(dir+'pg_dict.txt', 'w') as f:
        f.write(json.dumps(pg_dict))


####################################### new
def test(model_path, plan_path, dir):
    lero_model = load_model(model_path)
    plans_list = load_plans(plan_path)
    
    y_list, true_latencys_list = [], []
    for plans in plans_list:
        true_latencys = np.array([json.loads(plan)[0]['Execution Time']/1000 for plan in plans])
        true_latencys_list.append(np.array(true_latencys))
        local_features, _ = lero_model._feature_generator.transform(plans)
        y = lero_model.predict(local_features)
        y_list.append(y.squeeze())
    
    ranking_loss = compute_ranking_loss(y_list, true_latencys_list)
    logger.info(f'ranking loss: {ranking_loss}')

    choice = [np.argmin(row) for row in y_list]
    
    lero_dict = {}
    sum = 0.0
    id = 0
    for y in choice:
        plan = json.loads(plans_list[id][y])
        lero_dict['q'+str(id)] = plan[0]['Execution Time']/1000 
        id += 1
        sum += plan[0]['Execution Time']/1000 
    
    lero_dict['sum'] = sum
    logger.info(f'total latency:{sum}s')
    
    with open(os.path.join(dir,'lero_dict.json'), 'w') as f:
        json.dump(lero_dict, f, indent=4)

########################################### new
def compute_ranking_loss(y_list, true_latencys_list):
    from scipy.stats import spearmanr
    import warnings
    from scipy.stats import ConstantInputWarning
    warnings.filterwarnings("error", category=ConstantInputWarning)
    """
    计算预测的排序损失（Spearman 相关系数的均值）。

    参数：
    - y_list: list of numpy arrays，每个元素是一个一维 numpy 数组，表示一个 SQL 的候选 plan 的预测延迟。
    - true_latencys_list: list of numpy arrays，结构与 y_list 相同，表示真实延迟。

    返回：
    - avg_spearman_corr: 所有 SQL 的 Spearman 相关系数的均值，作为整体的排序损失指标。
    """
    spearman_corrs = []

    for y_pred, y_true in zip(y_list, true_latencys_list):
        # 检查长度是否一致
        assert len(y_pred) == len(y_true), "预测值和真实值的长度不一致。"
        try:
            # 计算 Spearman 相关系数
            coef, _ = spearmanr(y_pred, y_true)
        except ConstantInputWarning as e:
            logger.warning(f'捕获到警告：{e}')
        if np.isnan(coef):
            # 如果相关系数为 NaN（可能因为常数数组），则跳过
            continue
        spearman_corrs.append(coef)

    if len(spearman_corrs) == 0:
        logger.info("没有有效的 Spearman 相关系数计算结果。")
        return None

    # 计算平均 Spearman 相关系数
    avg_spearman_corr = np.mean(spearman_corrs)

    # 可以选择返回 1 - 平均相关系数，作为损失（相关系数越高，损失越低）
    ranking_loss = 1 - avg_spearman_corr

    return ranking_loss
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser("test")
    parser.add_argument("--plan_path",
                        metavar="PATH",
                        help="Load the plans")
    parser.add_argument("--model_path",
                        metavar="PATH",
                        help="model path")
    parser.add_argument("--dict_dir",
                        metavar="PATH",
                        help="dict dir")
    args = parser.parse_args()
    
    print("do test:")
    print('plan path: ' + args.plan_path)
    print('model path: ' + args.model_path)
    print('dict dir: ' + args.dict_dir)
    get_lero_dict(args.model_path, args.plan_path, args.dict_dir)
    # get_pg_dict(args.plan_path, args.dict_dir)              
    