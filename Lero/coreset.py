import numpy as np
import torch, time
from datasketch import MinHashLSH, MinHash

def lsh_partition(U, num_groups, num_perm=1024):
    N_U, D = U.shape
    lsh = MinHashLSH(threshold=0.1, num_perm=num_perm)
    U_np = U.detach().numpy()

    # 将样本索引和对应的 MinHash 存入 LSH
    for idx in range(N_U):
        vec = U_np[idx]
        quantized_vec = set(np.where(vec > 0)[0])
        m = MinHash(num_perm=num_perm)
        for d in quantized_vec:
            m.update(str(d).encode('utf8'))
        lsh.insert(idx, m)

    # 获取分组
    buckets = {}
    for idx in range(N_U):
        vec = U_np[idx]
        quantized_vec = set(np.where(vec > 0)[0])
        m = MinHash(num_perm=num_perm)
        for d in quantized_vec:
            m.update(str(d).encode('utf8'))
        similar_indices = lsh.query(m)
        bucket_id = tuple(sorted(similar_indices))
        if bucket_id not in buckets:
            buckets[bucket_id] = []
        buckets[bucket_id].append(idx)

    # 将桶转换为列表
    U_groups = list(buckets.values())

    # 如果组数量超过期望的 num_groups，可以进行合并或截断
    if len(U_groups) > num_groups:
        # 合并小的组
        U_groups = sorted(U_groups, key=len, reverse=True)
        U_groups = U_groups[:num_groups]
    elif len(U_groups) < num_groups:
        # 随机填充空的组
        while len(U_groups) < num_groups:
            U_groups.append([])

    return U_groups

def greedy_core_set_selection(L, U_groups, n_selections_per_group):
    """
    1. 将输入矩阵 `L` 转置得到 `L_T`。
    2. 对于每个组 `U`：
    - 计算 `U` 中每个向量与 `L` 中所有向量的相似性，得到 `sim_with_L`。
    - 计算每个向量与 `L` 中向量的最大相似性，存储在 `max_sim_L` 中。
    3. 对于每个组 `U`：
    - 选择一定数量的向量作为代表：
        - 初始化空的 `selected_indices` 和 `selected_vectors` 列表。
        - 在剩余候选向量中，选择与 `L` 中最不相似的向量，并将其加入代表集。
        - 更新剩余候选向量和最大相似性。
        - 重复以上步骤，直到选取所需数量的代表向量。
    - 将选取的向量索引存储在 `selected_indices_per_group` 中。

    最终，函数返回一个列表 `selected_indices_per_group`，其中包含每个组选取的代表向量的索引。
    """
    selected_indices_per_group = []
    L_T = L.t()

    sim_with_L_list = []
    max_sim_L_list = []
    for U in U_groups:
        if len(U) == 0:
            sim_with_L_list.append(None)
            max_sim_L_list.append(None)
            continue
        sim_with_L = torch.matmul(U, L_T)
        sim_with_L_list.append(sim_with_L)
        max_sim_L, _ = sim_with_L.max(dim=1)
        max_sim_L_list.append(max_sim_L)

    for group_idx, U in enumerate(U_groups):
        if len(U) == 0:
            selected_indices_per_group.append([])
            continue
        selected_indices = []
        selected_vectors = []

        candidate_indices = torch.arange(U.size(0))
        max_sim = max_sim_L_list[group_idx].clone()

        for _ in range(min(n_selections_per_group, U.size(0))):
            min_max_sim, min_idx = torch.min(max_sim[candidate_indices], dim=0)
            selected_idx = candidate_indices[min_idx].item()

            selected_indices.append(selected_idx)
            selected_vectors.append(U[selected_idx])

            candidate_indices = candidate_indices[candidate_indices != selected_idx]

            if len(selected_vectors) < n_selections_per_group and len(candidate_indices) > 0:
                new_sim = torch.matmul(U[candidate_indices], U[selected_idx].unsqueeze(1)).squeeze(1)
                max_sim[candidate_indices] = torch.max(max_sim[candidate_indices], new_sim)

        selected_indices_per_group.append(selected_indices)

    return selected_indices_per_group

def greedy_core_set_selection_without_L(U_groups, n_selections_per_group):
    selected_indices_per_group = []

    for group_idx, U in enumerate(U_groups):
        if len(U) == 0:
            selected_indices_per_group.append([])
            continue
        selected_indices = []
        selected_vectors = []

        candidate_indices = torch.arange(U.size(0))
        # max_sim = max_sim_L_list[group_idx].clone()

        max_sim = torch.ones(U.size(0),1).squeeze(1)
        for _ in range(min(n_selections_per_group, U.size(0))):
            min_max_sim, min_idx = torch.min(max_sim[candidate_indices], dim=0)
            selected_idx = candidate_indices[min_idx].item()

            selected_indices.append(selected_idx)
            selected_vectors.append(U[selected_idx])

            candidate_indices = candidate_indices[candidate_indices != selected_idx]
        
            if len(selected_vectors) < n_selections_per_group and len(candidate_indices) > 0:
                new_sim = torch.matmul(U[candidate_indices], U[selected_idx].unsqueeze(1)).squeeze(1)
                max_sim[candidate_indices] = torch.max(max_sim[candidate_indices], new_sim)

        selected_indices_per_group.append(selected_indices)

    return selected_indices_per_group
    
def coreset(U, L, num_groups, n_selections_per_group):
    U_groups_indices = lsh_partition(U, num_groups)
    U_groups = [U[indices] if len(indices) > 0 else torch.empty(0, D) for indices in U_groups_indices]
    selected_indices_per_group = greedy_core_set_selection(L, U_groups, n_selections_per_group)
    coreset_indices = []
    for group_idx, (group_indices, selected_indices) in enumerate(zip(U_groups_indices, selected_indices_per_group)):
        coreset_indices.extend([group_indices[idx] for idx in selected_indices])
    return U[coreset_indices], coreset_indices

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

if __name__ == '__main__':
    # 示例数据
    N_L = 2000
    N_U = 10000
    D = 128

    torch.manual_seed(0)

    L = torch.randn(N_L, D)
    # L = L / L.norm(dim=1, keepdim=True)

    U = torch.randn(N_U, D)
    # U = U / U.norm(dim=1, keepdim=True)
    U, L = standardize_and_normalize(U, L)
    

    # 使用 LSH 将未标记样本分组
    num_groups = 64
    t1 = time.time()
    U_groups_indices = lsh_partition(U, num_groups)
    t2 = time.time()
    print(f'lsh time:{t2-t1}')
    # 将组索引转换为特征矩阵列表
    U_groups = [U[indices] if len(indices) > 0 else torch.empty(0, D) for indices in U_groups_indices]

    # 对每个组应用核心集选择算法
    n_selections_per_group = 40
    selected_indices_per_group = greedy_core_set_selection(L, U_groups, n_selections_per_group)
    t3 = time.time()
    print(f'coreset time:{t3-t2}')
    # 输出每个组中选出的样本索引（在未标记样本集 U 中的全局索引）
    for group_idx, (group_indices, selected_indices) in enumerate(zip(U_groups_indices, selected_indices_per_group)):
        global_selected_indices = [group_indices[idx] for idx in selected_indices]
        print(f"组 {group_idx} 中选出的样本全局索引：", global_selected_indices)
