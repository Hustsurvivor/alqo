import time
import numpy as np

def compute_distance_matrix_vectorized(encodings, join_list_length, filter_list_length, d_max=2, w_join=0.5, w_filter=0.5):
    """
    Compute the distance matrix between all SQL query encodings using vectorized operations.

    :param encodings: List of encodings, shape (N, D)
    :return: Distance matrix, shape (N, N)
    """
    N = encodings.shape[0]
    join_encodings = encodings[:, :join_list_length]  # Shape: (N, join_list_length)
    filter_encodings = encodings[:, join_list_length:]  # Shape: (N, filter_list_length * 3)
    filter_encodings = filter_encodings.reshape(N, filter_list_length, 3)  # Shape: (N, filter_list_length, 3)

    # Compute the normalized join distances
    # Calculate the Hamming distances between all pairs
    join_diff = join_encodings[:, np.newaxis, :] != join_encodings[np.newaxis, :, :]  # Shape: (N, N, join_list_length)
    D_join = np.sum(join_diff, axis=2)  # Shape: (N, N)
    D_join_norm = D_join / join_list_length if join_list_length > 0 else 0

    # Compute the normalized filter distances
    total_filter_distance_matrix = np.zeros((N, N))

    # Extract filter components
    filter_present = filter_encodings[:, :, 0] == 1  # Shape: (N, filter_list_length)
    lb = filter_encodings[:, :, 1]  # Shape: (N, filter_list_length)
    ub = filter_encodings[:, :, 2]  # Shape: (N, filter_list_length)

    max_filter_distance = filter_list_length * d_max

    for i in range(filter_list_length):
        f1_present = filter_present[:, i][:, np.newaxis]  # Shape: (N, 1)
        f2_present = filter_present[:, i][np.newaxis, :]  # Shape: (1, N)

        # Boolean matrices indicating presence conditions
        f_present_both = f1_present & f2_present  # Both present
        f_present_diff = f1_present != f2_present  # One present, one absent

        # Compute differences where both filters are present
        lb_diff = np.abs(lb[:, i][:, np.newaxis] - lb[:, i][np.newaxis, :])
        ub_diff = np.abs(ub[:, i][:, np.newaxis] - ub[:, i][np.newaxis, :])
        d_i_both_present = lb_diff + ub_diff

        # Initialize distance matrix for this filter
        d_i = np.zeros((N, N))

        # Assign distances based on presence conditions
        d_i[f_present_both] = d_i_both_present[f_present_both]
        d_i[f_present_diff] = d_max
        # d_i is already zero where both filters are absent

        total_filter_distance_matrix += d_i

    D_filter_norm = total_filter_distance_matrix / max_filter_distance if max_filter_distance > 0 else 0

    # Combine distances
    D_total = w_join * D_join_norm + w_filter * D_filter_norm
    return D_total


import cupy as cp  # 使用 CuPy 代替 NumPy

def compute_distance_matrix_vectorized_gpu(encodings, join_list_length, filter_list_length, d_max=2, w_join=0.5, w_filter=0.5):
    """
    使用 GPU 加速计算距离矩阵。
    
    :param encodings: 编码列表，形状为 (N, D)
    :return: 距离矩阵，形状为 (N, N)
    """
    # 将数据从 CPU 内存传输到 GPU 内存
    encodings_gpu = cp.asarray(encodings)
    
    N = encodings_gpu.shape[0]
    join_encodings = encodings_gpu[:, :join_list_length]  # 形状: (N, join_list_length)
    filter_encodings = encodings_gpu[:, join_list_length:]  # 形状: (N, filter_list_length * 3)
    filter_encodings = filter_encodings.reshape(N, filter_list_length, 3)  # 形状: (N, filter_list_length, 3)
    
    # 计算归一化的连接距离（Join Distance）
    join_diff = join_encodings[:, cp.newaxis, :] != join_encodings[cp.newaxis, :, :]  # 形状: (N, N, join_list_length)
    D_join = cp.sum(join_diff, axis=2)  # 形状: (N, N)
    D_join_norm = D_join / join_list_length if join_list_length > 0 else 0
    
    # 初始化总的过滤器距离矩阵
    total_filter_distance_matrix = cp.zeros((N, N))
    max_filter_distance = filter_list_length * d_max
    
    # 提取过滤器的各个部分
    filter_present = filter_encodings[:, :, 0] == 1  # 形状: (N, filter_list_length)
    lb = filter_encodings[:, :, 1]  # 形状: (N, filter_list_length)
    ub = filter_encodings[:, :, 2]  # 形状: (N, filter_list_length)
    
    for i in range(filter_list_length):
        f1_present = filter_present[:, i][:, cp.newaxis]  # 形状: (N, 1)
        f2_present = filter_present[:, i][cp.newaxis, :]  # 形状: (1, N)
        
        # 两个过滤器都存在的情况
        f_present_both = f1_present & f2_present
        # 只有一个过滤器存在的情况
        f_present_diff = f1_present != f2_present
        
        # 计算下界和上界的差异
        lb_diff = cp.abs(lb[:, i][:, cp.newaxis] - lb[:, i][cp.newaxis, :])
        ub_diff = cp.abs(ub[:, i][:, cp.newaxis] - ub[:, i][cp.newaxis, :])
        d_i_both_present = lb_diff + ub_diff
        
        # 初始化当前过滤器的距离矩阵
        d_i = cp.zeros((N, N))
        d_i[f_present_both] = d_i_both_present[f_present_both]
        d_i[f_present_diff] = d_max
        # 当两个过滤器都不存在时，距离为零，已默认设置
        
        total_filter_distance_matrix += d_i
    
    D_filter_norm = total_filter_distance_matrix / max_filter_distance if max_filter_distance > 0 else 0
    
    # 组合距离
    D_total = w_join * D_join_norm + w_filter * D_filter_norm
    
    # 将结果从 GPU 内存传回 CPU 内存（如果需要在 CPU 上进一步处理）
    D_total_cpu = cp.asnumpy(D_total)
    return D_total_cpu

from tqdm import tqdm

def generate_test_data(N, join_length, filter_list_length, seed=42):
    """
    生成测试数据。

    :param N: 向量数量。
    :param join_length: join 编码长度。
    :param filter_list_length: filter 列表长度（每个 filter 有 3 个元素）。
    :param seed: 随机种子。
    :return: encodings 数组，形状 (N, D)。
    """
    np.random.seed(seed)
    
    # 生成 join 编码：0 或 1
    join_encodings = np.random.randint(0, 2, size=(N, join_length)).astype(np.float32)  # 转换为浮点数
    
    # 生成 filter 编码
    # 每个 filter 有 3 个元素：存在标志（0.0 或 1.0），lb，ub
    # lb 和 ub 是 [0.0, 1.0] 之间的浮点数，且 ub >= lb
    filter_present = np.random.randint(0, 2, size=(N, filter_list_length)).astype(np.float32)  # 0.0 或 1.0
    filter_lb = np.random.rand(N, filter_list_length).astype(np.float32)  # [0.0, 1.0)
    filter_ub = filter_lb + np.random.rand(N, filter_list_length).astype(np.float32) * (1.0 - filter_lb)  # 确保 ub >= lb

    # 组合 filter 编码
    filter_encodings = np.stack((filter_present, filter_lb, filter_ub), axis=2).reshape(N, filter_list_length * 3)
    
    # 合并 join 和 filter 编码
    encodings = np.hstack((join_encodings, filter_encodings)).astype(np.float32)
    
    return encodings
if __name__ == "__main__":
     # 参数设置
    N = 6000
    join_length = 20
    filter_list_length = 20  # 每个 filter 有 3 个元素，60 / 3 = 20
    d_max = 2
    w_join = 0.5
    w_filter = 0.5

    print("生成测试数据...")
    encodings = generate_test_data(N, join_length, filter_list_length)
    print(f"生成的数据形状: {encodings.shape}")

    # 预热 CPU 和 GPU
    print("预热 CPU 和 GPU...")
    _ = compute_distance_matrix_vectorized(encodings, join_length, filter_list_length, d_max, w_join, w_filter)
    _ = compute_distance_matrix_vectorized_gpu(encodings, join_length, filter_list_length, d_max, w_join, w_filter)
    print("预热完成。\n")

    # 测量 CPU 计算时间
    print("开始 CPU 计算距离矩阵...")
    start_time = time.perf_counter()
    D_cpu = compute_distance_matrix_vectorized(encodings, join_length, filter_list_length, d_max, w_join, w_filter)
    end_time = time.perf_counter()
    cpu_time = end_time - start_time
    print(f"CPU 计算完成，耗时: {cpu_time:.2f} 秒\n")

    # 测量 GPU 计算时间
    print("开始 GPU 计算距离矩阵...")
    start_time = time.perf_counter()
    D_gpu = compute_distance_matrix_vectorized_gpu(encodings, join_length, filter_list_length, d_max, w_join, w_filter)
    # 等待 GPU 计算完成
    cp.cuda.Stream.null.synchronize()
    end_time = time.perf_counter()
    gpu_time = end_time - start_time
    print(f"GPU 计算完成，耗时: {gpu_time:.2f} 秒\n")

    # 验证 CPU 和 GPU 结果是否一致
    print("验证 CPU 和 GPU 计算结果是否一致...")
    difference = np.max(np.abs(D_cpu - D_gpu))
    if difference < 1e-6:
        print("验证通过：CPU 和 GPU 计算结果一致。\n")
    else:
        print(f"验证失败：CPU 和 GPU 计算结果存在差异，最大差异为 {difference}。\n")

    # 打印性能对比
    print("性能对比：")
    print(f"CPU 耗时: {cpu_time:.2f} 秒")
    print(f"GPU 耗时: {gpu_time:.2f} 秒")
    speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')
    print(f"加速比 (CPU / GPU): {speedup:.2f}x")