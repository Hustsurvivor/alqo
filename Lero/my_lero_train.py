import json
import torch, logging
import random
from feature import *
from model import LeroModel, LeroModelPairWise, BayesianModelPairWise
from coreset import *
import os
import time
import pickle
from sklearn.cluster import KMeans

np.random.seed(42)
#################################### utils  ###################################
def split_train_and_test_data(path, rate, train_path, test_path):
    count = 0
        
    with open(path, 'r') as f:
        lines = f.readlines()
        count = int(len(lines) * rate)
    
    random.shuffle(lines)
    
    with open(test_path, 'w') as f:
        f.write(''.join(lines[:count]))
    
    with open(train_path, 'w') as f:
        f.write(''.join(lines[count:]))

    print('------------------- finish split train and test data -------------------------')
    print(f'count of training sql: {len(lines)-count}')
    print(f'count of test sql: {count}')
    print('-------------------------------------------------------------------------')

def _load_pairwise_plans(path, count=-1):
    X1, X2 = [], []
    with open(path, 'r') as f:
        for line in f.readlines()[:count]:
            arr = line.split("#####")
            x1, x2 = get_training_pair(arr[1:])
            X1 += x1
            X2 += x2
    return X1, X2

def get_training_pair(candidates):
    assert len(candidates) >= 2
    X1, X2 = [], []

    i = 0
    while i < len(candidates) - 1:
        s1 = candidates[i]
        j = i + 1
        while j < len(candidates):
            s2 = candidates[j]
            X1.append(s1)
            X2.append(s2)
            j += 1
        i += 1
    return X1, X2

# 绝对误差
def are_close_absolute(a, b, tol):
    return abs(a - b) < tol

# 相对误差
def are_close_relative(a, b, tol):
    return abs(a - b) < tol * max(abs(a), abs(b))

def filter_close_latencies(plans1, plans2, use_relative=True, threshold=0.01):
    assert(len(plans1) == len(plans2))
    X1, X2 = [], []

    for plan1, plan2 in zip(plans1, plans2):
        json_plan1 = json.loads(plan1)
        json_plan2 = json.loads(plan2)
        
        # 使用相对误差
        if use_relative:
            if not are_close_relative(json_plan1[0]["Execution Time"], json_plan2[0]["Execution Time"], threshold):
                X1.append(plan1)
                X2.append(plan2)
        
        # 使用绝对误差
        else:
            if not are_close_absolute(json_plan1[0]["Execution Time"], json_plan2[0]["Execution Time"], threshold):
                X1.append(plan1)
                X2.append(plan2)
                
    print('------------------- finish filter close latency -------------------------')
    print(f'count of origin plan pairs: {len(plans1)}')
    print(f'count of filtered plan pairs: {len(X1)}')
    print('-------------------------------------------------------------------------')
    
    return X1, X2 

def training_pairwise(X1, X2, tuning_model_path, model_name, pretrain=False):
    tuning_model = tuning_model_path is not None
    lero_model = None
    if tuning_model:
        lero_model = LeroModelPairWise(None)
        lero_model.load(tuning_model_path)
        feature_generator = lero_model._feature_generator
    else:
        feature_generator = FeatureGenerator()
        feature_generator.fit(X1 + X2)

    Y1, Y2 = None, None
    if pretrain:
        Y1 = [json.loads(c)[0]['Plan']['Total Cost'] for c in X1]
        Y2 = [json.loads(c)[0]['Plan']['Total Cost'] for c in X2]
        X1, _ = feature_generator.transform(X1)
        X2, _ = feature_generator.transform(X2)
    else:
        X1, Y1 = feature_generator.transform(X1)
        X2, Y2 = feature_generator.transform(X2)
    logger.info("Training data set size = " + str(len(X1)))

    if not tuning_model:
        assert lero_model == None
        lero_model = LeroModelPairWise(feature_generator)
    lero_model.fit(X1, X2, Y1, Y2, tuning_model)

    logger.info(f"saving model to {model_name}")
    lero_model.save(model_name)
    
def training_Bayesian(X1, X2, tuning_model_path, model_name, pretrain=False):
    tuning_model = tuning_model_path is not None
    bayesian_model = None
    if tuning_model:
        bayesian_model = BayesianModelPairWise(None)
        bayesian_model.load(tuning_model_path)
        feature_generator = bayesian_model._feature_generator
    else:
        feature_generator = FeatureGenerator()
        feature_generator.fit(X1 + X2)

    Y1, Y2 = None, None
    if pretrain:
        Y1 = [json.loads(c)[0]['Plan']['Total Cost'] for c in X1]
        Y2 = [json.loads(c)[0]['Plan']['Total Cost'] for c in X2]
        X1, _ = feature_generator.transform(X1)
        X2, _ = feature_generator.transform(X2)
    else:
        X1, Y1 = feature_generator.transform(X1)
        X2, Y2 = feature_generator.transform(X2)
    logger.info("Training data set size = " + str(len(X1)))
    
    if not tuning_model:
        assert bayesian_model == None
        bayesian_model = BayesianModelPairWise(feature_generator)
    bayesian_model.fit(X1, X2, Y1, Y2, tuning_model)

    logger.info(f"saving model to {model_name}")
    bayesian_model.save(model_name)
    
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

def get_feature(model_path, data_path, batch_size=64, test=False):
    X1, X2 = _load_pairwise_plans(data_path) # 构建n(n-1)/2对比较数据
    lero_model = LeroModelPairWise(None)
    lero_model.load(model_path)
    feature_generator = lero_model._feature_generator

    ### 
    if test:
        count = int(len(X1) * 0.25)
        X1 = X1[:count]
        X2 = X2[:count]
    ###

    X1, _ = feature_generator.transform(X1)
    X2, _ = feature_generator.transform(X2)
    
    features1, features2 = [], []
    for i in range(0, len(X1), batch_size):
        x1, x2 = X1[i:i+batch_size], X2[i:i+batch_size]
        fea1, fea2 = lero_model.get_inter_fea(x1, x2)
        features1.extend(fea1)
        features2.extend(fea2)
    features1 = torch.stack(features1) #  list 48882 * 64 -> shape [3128448]
    features2 = torch.stack(features2) #  
    return features1, features2

def test_latency_version(model_path, plan_path, output_path, use_relative=True, threshold=0.01):
    """
    测试模型效果，剔除latency相近的plan_pair,计算排序准确率
    """
    lero_model = load_model(model_path)
    X1_original, X2_original = _load_pairwise_plans(plan_path)
    
    # 过滤相近的latency
    X1_filtered, X2_filtered = filter_close_latencies(X1_original, X2_original, use_relative, threshold)

    correct_count = 0
    total_count = len(X1_filtered)
     
    local_features1, _ = lero_model._feature_generator.transform(X1_filtered)
    local_features2, _ = lero_model._feature_generator.transform(X2_filtered)
    
    y_pred_1 = lero_model.predict(local_features1)
    y_pred_2 = lero_model.predict(local_features2) 
    
    pred_diff_list = [ y1 - y2 for y1,y2 in zip(y_pred_1, y_pred_2)]
     
    for id, (x1, x2) in enumerate(zip(X1_filtered, X2_filtered)):
        pred = pred_diff_list[id]  
        label = json.loads(x1)[0]['Execution Time'] - json.loads(x2)[0]['Execution Time']

        if ( pred >= 0 ) == (label >= 0):
            correct_count += 1
    
    lero_dict = { 'filter_type' : 'relative' if use_relative else 'absolute', 'threshold' : threshold ,'accuracy' : 1.0 * correct_count / total_count}
    
    with open(output_path, 'w') as f:
        json.dump(lero_dict, f, indent=4)

def test_BayesianModel(model_path, plan_path, output_path):
    """
    测试贝叶斯模型，计算分类准确率和排序loss
    """
    bayesian_model = BayesianModelPairWise(None)
    bayesian_model.load(model_path)
    plans_list = load_plans(plan_path)
    
    correct_count = 0
    total_count = 0
    sum = 0.0
    for plans in plans_list:
        X1, X2 = get_training_pair(plans)
        
        n = len(plans)
        score_matrix = np.zeros(n)
        
        i, j = 0, 0
        while i < n - 1:
            s1 = plans[i]
            j = i + 1
            while j < n:
                s2 = plans[j]
                x1, y1 = bayesian_model._feature_generator.transform([s1])
                x2, y2 = bayesian_model._feature_generator.transform([s2])
                prob, _ = bayesian_model.predict(x1, x2)

                true_label = 1 if y1[0] >= y2[0] else 0
                prob_label = 1 if prob[0] >= 0.5 else 0                
                
                correct_count += 1 if true_label == prob_label or y1[0] == y2[0] else 0
                total_count += 1
                
                if prob_label == 1:
                    score_matrix[j] += 1
                else:
                    score_matrix[i] += 1
                
                j += 1
                i += 1
        
        selected_plan_index = np.argmax(score_matrix)
        sum += json.loads(plans[selected_plan_index])[0]['Execution Time']/1000
        
    bayesian_dict = {}
    bayesian_dict['sum'] = sum / 60.0 / 60.0
    bayesian_dict['accuracy'] = 1.0*correct_count/total_count    
    
    with open(output_path, 'w') as f:
        json.dump(bayesian_dict, f, indent=4)    
    
def test(model_path, plan_path, output_path):
    """
    测试模型效果，计算排序loss
    """
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
    
    lero_dict['sum'] = sum / 60.0 /60.0
    lero_dict['ranking loss'] = ranking_loss
    logger.info(f'total latency:{sum}s')
    
    with open(output_path, 'w') as f:
        json.dump(lero_dict, f, indent=4)

def test_LeroModel(model_path, plan_path, output_path):
    lero_model = load_model(model_path)
    plans_list = load_plans(plan_path)
    
    correct_count = 0
    total_count = 0
    sum = 0.0
    for plans in plans_list:
               
        n = len(plans)
        score_matrix = np.zeros(n)
        
        i, j = 0, 0
        while i < n - 1:
            s1 = plans[i]
            j = i + 1
            while j < n:
                s2 = plans[j]
                x1, y1 = lero_model._feature_generator.transform([s1])
                x2, y2 = lero_model._feature_generator.transform([s2])
                prob1 = lero_model.predict(x1)
                prob2 = lero_model.predict(x2)

                true_label = 1 if y1[0] >= y2[0] else 0
                prob_label = 1 if prob1[0] >= prob2[0] else 0                
                
                correct_count += 1 if true_label == prob_label or y1[0] == y2[0] else 0
                total_count += 1
                
                if prob_label == 1:
                    score_matrix[j] += 1
                else:
                    score_matrix[i] += 1
                
                j += 1
                i += 1
        
        selected_plan_index = np.argmax(score_matrix)
        sum += json.loads(plans[selected_plan_index])[0]['Execution Time']/1000
        
    bayesian_dict = {}
    bayesian_dict['sum'] = sum / 60.0 / 60.0
    bayesian_dict['accuracy'] = 1.0*correct_count/total_count    
    
    with open(output_path, 'w') as f:
        json.dump(bayesian_dict, f, indent=4) 

# def test_LeroModel(model_path, plans_list, output_path):
#     lero_model = load_model(model_path)
    
#     correct_count = 0
#     total_count = 0
#     sum = 0.0
#     for plans in plans_list:
               
#         n = len(plans)
#         score_matrix = np.zeros(n)
        
#         i, j = 0, 0
#         while i < n - 1:
#             s1 = plans[i]
#             j = i + 1
#             while j < n:
#                 s2 = plans[j]
#                 x1, y1 = lero_model._feature_generator.transform([s1])
#                 x2, y2 = lero_model._feature_generator.transform([s2])
#                 prob1 = lero_model.predict(x1)
#                 prob2 = lero_model.predict(x2)

#                 true_label = 1 if y1[0] >= y2[0] else 0
#                 prob_label = 1 if prob1[0] >= prob2[0] else 0                
                
#                 correct_count += 1 if true_label == prob_label or y1[0] == y2[0] else 0
#                 total_count += 1
                
#                 if prob_label == 1:
#                     score_matrix[j] += 1
#                 else:
#                     score_matrix[i] += 1
                
#                 j += 1
#                 i += 1
        
#         selected_plan_index = np.argmax(score_matrix)
#         sum += json.loads(plans[selected_plan_index])[0]['Execution Time']/1000
        
#     bayesian_dict = {}
#     bayesian_dict['sum'] = sum / 60.0 / 60.0
#     bayesian_dict['accuracy'] = 1.0*correct_count/total_count    
    
#     with open(output_path, 'w') as f:
#         json.dump(bayesian_dict, f, indent=4)

def standardize_and_normalize(U, L):
    """
    对特征矩阵进行标准化和归一化。

    参数：
    - X: torch.Tensor，形状为 (样本数量, 特征数量)

    返回：
    - X_normalized: 标准化并归一化后的特征矩阵，形状与 X 相同
    """
    # 标准化（对每个特征减去均值，除以标准差）
    # 计算每个特征的均值和标准差
    N_U = len(U)
    X = torch.cat((U, L), dim=0)
    means = X.mean(dim=0, keepdim=True)        # Shape: (1, 特征数量)
    stds = X.std(dim=0, unbiased=False, keepdim=True)  # Shape: (1, 特征数量)
    
    # 避免除以零，对于标准差为零的特征，设置为1
    stds[stds == 0] = 1.0

    X_standardized = (X - means) / stds

    # 归一化（对每个样本除以其范数）
    # 计算每个样本的范数（L2 范数）
    norms = X_standardized.norm(p=2, dim=1, keepdim=True)  # Shape: (样本数量, 1)

    # 避免除以零，对于范数为零的样本，设置为1
    norms[norms == 0] = 1.0

    X_normalized = X_standardized / norms

    return X_normalized[:N_U], X_normalized[N_U:]

def coreset(L, U, D, num_groups=32, n_selections_per_group=100):
    """
    L: labeled data
    U: unlabeled data
    D: dim of data
    """
    # L = L / L.norm(dim=1, keepdim=True)

    # U = U / U.norm(dim=1, keepdim=True)

    U, L = standardize_and_normalize(U, L)
    # 使用 LSH 将未标记样本分组
    # num_groups = 32
    t1 = time.time()
    U_groups_indices = lsh_partition(U, num_groups)
    t2 = time.time()
    print(f'lsh time:{t2-t1}')
    # 将组索引转换为特征矩阵列表
    U_groups = [U[indices] if len(indices) > 0 else torch.empty(0, D) for indices in U_groups_indices]

    # 对每个组应用核心集选择算法
    # n_selections_per_group = 100
    selected_indices_per_group = greedy_core_set_selection(L, U_groups, n_selections_per_group)
    t3 = time.time()
    print(f'coreset time:{t3-t2}')
    # 输出每个组中选出的样本索引（在未标记样本集 U 中的全局索引）
    for group_idx, (group_indices, selected_indices) in enumerate(zip(U_groups_indices, selected_indices_per_group)):
        global_selected_indices = [group_indices[idx] for idx in selected_indices]
        print(f"组 {group_idx} 中选出的样本全局索引：", global_selected_indices)
    
    with open('../result/exp1/u_groups_indices.pkl','wb') as f:
        pickle.dump(U_groups_indices, f)
    
    with open('../result/exp1/selected_indices_per_group.pkl', 'wb') as f:
        pickle.dump(selected_indices_per_group, f)
    
    
def kmeans_partition(U, num_groups):
    """
    使用 K-means 将数据分组
    :param U: 输入数据矩阵，形状为 (N, D)，N 为样本数量，D 为特征维度
    :param num_groups: 需要划分的组数 (即 K-means 的 K 值)
    :return: 分组后的结果，列表形式
    """
    if isinstance(U, torch.Tensor):
        U = U.detach().numpy()
    
    # 使用 K-means 聚类
    kmeans = KMeans(n_clusters=num_groups, random_state=42)
    labels = kmeans.fit_predict(U)  # 获取每个样本的簇标签

    # 根据标签分组
    groups = [[] for _ in range(num_groups)]
    for idx, label in enumerate(labels):
        groups[label].append(idx)  # 将样本添加到对应簇的组中

    return groups

def coreset_kmeans_version(L, U, D, num_groups=32, n_selections_per_group=100):
    U, L = standardize_and_normalize(U, L)
    t1 = time.time()
    U_groups_indices = kmeans_partition(U, num_groups)
    t2 = time.time()
    print(f'kmeans time:{t2-t1}')
    
    U_groups = [U[indices] if len(indices) > 0 else torch.empty(0, D) for indices in U_groups_indices]
    selected_indices_per_group = greedy_core_set_selection(L, U_groups, n_selections_per_group)
    t3 = time.time()
    print(f'coreset time:{t3-t2}')
    
    selected_samples = []
    for group_idx, (group_indices, selected_indices) in enumerate(zip(U_groups_indices, selected_indices_per_group)):
        global_selected_indices = [group_indices[idx] for idx in selected_indices]
        print(f"组 {group_idx} 中选出的样本全局索引：", global_selected_indices)
        selected_samples.extend(global_selected_indices)
    
    return selected_samples, U_groups_indices
        
#####################################################################################


############################### experiment ##########################################

"""
random方法与核心集方法对比
思考：
1. feature1和feature2怎么处理来用到核心集里?
    a. 直接cat连接
"""
def compare_random_and_coreset_ferformance():
    print(f'do expriment 1:')
    # 参数设置
    labeled_sql_path = '../data/labeled_sql.txt' 
    unlabeled_sql_path = '../data/unlabeled_sql.txt'
    test_sql_path = '../data/test.txt'
    U_path = '../data/exp1/U.pt'
    L_path = '../data/exp1/L.pt'
    
    siginal = 'without_labeled_sql/32_100'
    feature_model_path = '../model/exp1/feature_stats_lero_model'
    target_model_path = f'../model/exp1/kmeans/{siginal}/target_stats_lero_model'
    random_model_path_list = [ f'../model/exp1/kmeans/{siginal}/random_stats_lero_model_{i}' for i in range(5)]
    
    tartget_model_test_output_path = f'../result/exp1/kmeans/{siginal}/tartget_model_test_output.txt'
    random_model_test_output_path_list = [f'../result/exp1/kmeans/{siginal}/random_model_{i}_test_output.txt' for i in range(5)]
    
    feature_model = None 
    target_model = None 
    random_model = None 
    
    #### 用labeled sql进行训练
    
    # X1_labeled, X2_labeled = _load_pairwise_plans(labeled_sql_path)
    # training_pairwise(X1_labeled, X2_labeled, None, feature_model_path)
    
    # ## 计算核心集
    # feature1, feature2 = get_feature(feature_model_path, labeled_sql_path, test=False)
    # L = torch.concatenate((feature1, feature2), dim=1)
    # print(f'finish preparing L: {L.shape}')
    
    # feature1, feature2 = get_feature(feature_model_path, unlabeled_sql_path, test=False)
    # U = torch.concatenate((feature1, feature2), dim=1)
    # print(f'finish preparing U: {U.shape}')
    
    # torch.save(L, L_path)
    # torch.save(U, U_path)
    L = torch.load(L_path)
    U = torch.load(U_path)
    
    
    assert(L.shape[1] == U.shape[1])
    D = L.shape[1] 

    selected_indices, _ = coreset_kmeans_version(L, U, D, num_groups=64, n_selections_per_group=25)
    print(len(selected_indices))
    
    X1_unlabeled, X2_unlabeled = _load_pairwise_plans(unlabeled_sql_path)
    X1_coreset = [X1_unlabeled[indice] for indice in selected_indices]
    X2_coreset = [X2_unlabeled[indice] for indice in selected_indices]
    # X1_coreset.extend(X1_labeled)
    # X2_coreset.extend(X2_labeled)
    
    # # #### random方法筛选核心集
    print(len(X1_coreset))
    
    # # #### 训练模型
    
    training_pairwise(X1_coreset, X2_coreset, None, target_model_path)
    
    for i in range(5):
        random_indices = np.random.choice(len(X1_unlabeled), len(selected_indices), replace=False)
        X1_random = [X1_unlabeled[indice] for indice in random_indices]
        X2_random = [X2_unlabeled[indice] for indice in random_indices]
        # X1_random.extend(X1_labeled)
        # X2_random.extend(X2_labeled)
        training_pairwise(X1_random, X2_random, None, random_model_path_list[i])
    
    #### 测试模型
    # test_LeroModel(feature_model_path, test_sql_path, f'../result/exp1/kmeans/{siginal}/original_model_test_output.txt')
    
    # test_LeroModel(target_model_path, test_sql_path, tartget_model_test_output_path)
    
    # for i in range(5):
    #     test_LeroModel(random_model_path_list[i], test_sql_path, random_model_test_output_path_list[i])
    

""" 
剔除latency相近的plan,测试剔除latency相近的plan和未剔除后模型的性能比较,计算排序loss
思考：
1. 如何衡量两个latency是否相近:  
    a. 设定threshold
    b. todo: 相对值  (此时loss用分类loss，分类对的除以总共的对数) 
"""
def compare_latency_filtering_performance():
    print(f'do expriment2:')
    # 参数设置
    threshold = 0.1 
    training_data_file = '../data/train.txt'
    test_data_file = '../data/test.txt'
    original_model_path = f'../model/exp2/relative/{str(threshold)}/relative_original_stats_lero_model'
    filtered_model_path = f'../model/exp2/relative/{str(threshold)}/relative_filtered_stats_lero_model'
    
    original_model_test_output_path = f'../result/exp2/relative/{str(threshold)}/original_classification_loss_dict.json'
    filtered_model_test_output_path = f'../result/exp2/relative/{str(threshold)}/filtered_classification_loss_dict.json'
    
    #### train部分
    
    # 构建n(n-1)/2对比较数据
    # X1_original, X2_original = _load_pairwise_plans(training_data_file)
    
    # # 过滤相近的latency
    # X1_filtered, X2_filtered = filter_close_latencies(X1_original, X2_original, True, threshold)
    
    # training_pairwise(X1_original, X2_original, None, original_model_path)
    # training_pairwise(X1_filtered, X2_filtered, None, filtered_model_path)
    
    # #### test部分
    
    test_latency_version(original_model_path, test_data_file, original_model_test_output_path, True, threshold)
    test_latency_version(filtered_model_path, test_data_file, filtered_model_test_output_path, True, threshold)

"""
跑贝叶斯模型，测试uncertainty：将uncertainty高的训练数据提取出来作为新的核心集进行训练，观察uncertainty的效果 
"""
def compare_Bayesian_performance():
    print(f'do expriment3:')
    # 参数设置
    labeled_sql_path = '../data/labeled_sql.txt' 
    unlabeled_sql_path = '../data/unlabeled_sql.txt'
    training_data_file = '../data/train.txt'
    test_data_file = '../data/test.txt'
    siginal = 'prob'
    
    bayesian_model_path = '../model/exp3/bayesian_model'
    coreset_model_path = f'../model/exp3/{siginal}/coreset_model'
    
    random_model_path_list = [ f'../model/exp3/{siginal}/random_model_{i}' for i in range(1,5)]
    # random_model_path = '../model/exp3/random_model'
    
    bayesian_model_test_output_path = f'../result/exp3/original_loss_dict.json'
    coreset_model_test_output_path = f'../result/exp3/{siginal}/coreset_loss_dict.json'
    # random_model_test_output_path = '../result/exp3/random_loss_dict.json'
    random_model_test_output_path_list = [f'../result/exp3/{siginal}/random_loss_dict_{i}.json' for i in range(1,5)]
    
    #### 训练贝叶斯网络   
    # X1_labeled, X2_labeled = _load_pairwise_plans(labeled_sql_path)
    
    # # training_Bayesian(X1_labeled, X2_labeled, None, bayesian_model_path)
    
    # #### 提取训练数据中uncertainty高的训练数据作为核心集
    # X1_unlabeled, X2_unlabeled = _load_pairwise_plans(unlabeled_sql_path)
    
    # # rate = 0.15 # 筛选核心集的比例
    # # count = int(rate * len(X1_unlabeled))
    # count = 100

    # bayesian_model = BayesianModelPairWise(None)
    # bayesian_model.load(bayesian_model_path)
    # X1, Y1 = bayesian_model._feature_generator.transform(X1_unlabeled)
    # X2, Y2 = bayesian_model._feature_generator.transform(X2_unlabeled)
    
    # probs, log_variances = bayesian_model.predict(X1, X2)
    
    ## 1.挑选log_variance大的数据作为核心集
    # 对log_variance进行从小到大排序
    # sorted_indices = np.argsort(log_variances)
    # sorted_log_variance = log_variance[sorted_indices]
    # threshold = sorted_log_variance[-count]
    
    # coreset_indices = np.where(log_variances > threshold)[0]
    # coreset_X1 = [X1_unlabeled[indice] for indice in coreset_indices]
    # coreset_X2 = [X2_unlabeled[indice] for indice in coreset_indices]
    # coreset_X1.extend(X1_labeled)
    # coreset_X2.extend(X2_labeled)
    
    ## 2. 挑选loss大的数据作为核心集
    # labels =  [1.0 if Y1[i] >= Y2[i] else 0.0 for i in range(len(X1))]
    # loss_list = []
    # criterion = torch.nn.BCELoss()
    # with torch.no_grad():
    #     for idx, (prob, label) in enumerate(zip(probs, torch.tensor(labels))):
    #         loss = criterion(prob, label)
    #         loss_list.append((idx,loss.item()))
            
    # # 降序排列
    # loss_samples_sorted = sorted(loss_list, key=lambda x: x[1], reverse=True)
    # coreset_indices = [sample[0] for sample in loss_samples_sorted[:count]]
    # coreset_X1 = [X1_unlabeled[indice] for indice in coreset_indices]
    # coreset_X2 = [X2_unlabeled[indice] for indice in coreset_indices]
    # coreset_X1.extend(X1_labeled)
    # coreset_X2.extend(X2_labeled)
    
    ## 3. 挑选prob在0.5左右的数据组为核心集
    # v = [ (idx,abs(prob.item() - 0.5)) for idx,prob in enumerate(probs) ]
    # v_sorted = sorted(v, key=lambda x: x[1]) # 从小到大排序
    # coreset_indices = [sample[0] for sample in v_sorted[:count]]
    # coreset_X1 = [X1_unlabeled[indice] for indice in coreset_indices]
    # coreset_X2 = [X2_unlabeled[indice] for indice in coreset_indices]
    # coreset_X1.extend(X1_labeled)
    # coreset_X2.extend(X2_labeled)
    
    ## 使用random方法选择count个核心集作为对比
    # random_indices = np.random.choice(len(X1_unlabeled), count, replace=False)
    # random_X1 = [X1_unlabeled[indice] for indice in random_indices]
    # random_X2 = [X2_unlabeled[indice] for indice in random_indices]
    # random_X1.extend(X1_labeled)
    # random_X2.extend(X2_labeled)
    
    #### 训练两种模型
    # print(f'coreset data count: {len(coreset_X1)}')
    # # print(f'random data count: {len(random_X1)}')
    
    # training_Bayesian(coreset_X1, coreset_X2, None, coreset_model_path)
    # # training_Bayesian(random_X1, random_X2, None, random_model_path)
    
    # for random_model_path in random_model_path_list:
    #     random_indices = np.random.choice(len(X1_unlabeled), count, replace=False)
    #     random_X1 = [X1_unlabeled[indice] for indice in random_indices]
    #     random_X2 = [X2_unlabeled[indice] for indice in random_indices]
    #     random_X1.extend(X1_labeled)
    #     random_X2.extend(X2_labeled)
    #     training_Bayesian(random_X1, random_X2, None, random_model_path)

    #### 测试结果
    # test_BayesianModel(coreset_model_path, test_data_file, coreset_model_test_output_path)
    
    # for random_model_path, random_model_test_output_path in zip(random_model_path_list, random_model_test_output_path_list):
    #     test_BayesianModel(random_model_path, test_data_file, random_model_test_output_path)
        
    test_BayesianModel(bayesian_model_path, test_data_file, bayesian_model_test_output_path)
    

# 数据集测试
def test_dataset():
    dataset_path='../data/my_merged_stats_train_pool_plan.txt'
    model_path_list= [ f'../model/exp4/model_{i}'for i in range(5)]
    model_test_output_path_list = [ f'../result/exp4/model_{i}_test_output.json' for i in range(5)]
    
    
    plans = load_plans(dataset_path)    
    random.shuffle(plans)
    count = len(plans)
        
    train_plans_list = [ plans[:int(count*0.1)] ,
                        plans[:int(count*0.3)] ,
                        plans[:int(count*0.5)] ,
                        plans[:int(count*0.7)] ,
                        plans[:int(count*0.9)] ]
    test_plans = plans[int(count * 0.9):]
    for plans in train_plans_list:
        print(len(plans))
    
    print(len(test_plans)) 
    
    # for i in range(5):
    #     X1, X2 = [], []
    #     for plan in train_plans_list[i]:
    #         x1, x2 = get_training_pair(plan)
    #         X1.extend(x1)
    #         X2.extend(x2)
            
    #     training_pairwise(X1, X2, None, model_path_list[i])

    # for i in range(5):
    #     test_LeroModel(model_path_list[i], plans_list=test_plans, output_path=model_test_output_path_list[i])

if __name__ == '__main__':
    logger = logging.getLogger('my_logger')
    file_handler = logging.FileHandler('logfile.log')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    compare_random_and_coreset_ferformance()
    # compare_latency_filtering_performance()
    # compare_Bayesian_performance()
    # test_dataset()
    
    file_handler.close()
        