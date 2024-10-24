
from extract_join_and_filter_lists import extract_join_and_filter_lists, parse_conditions
import numpy as np
from collections import defaultdict
import sqlparse, json
from sqlparse.sql import Where, Comparison, Identifier, IdentifierList, Token
from sqlparse.tokens import Keyword, DML, Whitespace
from sklearn_extra.cluster import KMedoids
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor


def normalize_value(value, min_val, max_val):
    """将值归一化到0到1之间"""
    if max_val > min_val:
        return (value - min_val) / (max_val - min_val)
    else:
        return 0.0


def encode_sql_query(sql_query, alias_map, range_dict, join_list, filter_list):
    """
    编码 SQL 查询中的条件为向量表示，包括连接条件和过滤条件。
    :param sql_query: SQL 查询字符串
    :param alias_map: 别名映射字典，形如 {'a_alias': 'a', ...}
    :param range_dict: 每列的最小值和最大值，格式{'a': {'a1': [min, max], 'a2': [min, max]}, ...}
    :param join_list: 所有可能的连接条件列表，元素为集合，形如 [{a.a1, b.b1}, {b.b2, c.c2}, ...]
    :param filter_list: 需要考虑的过滤列列表，形如 ['a.a1', 'b.b2', ...]
    :return: 编码后的向量，包含连接条件编码和过滤条件编码
    """
    # 1. 解析 SQL 查询
    parsed = sqlparse.parse(sql_query)[0]
    from_seen = False
    actual_joins = []
    actual_filters = []
    local_alias_map = alias_map.copy()  # 本地的别名映射

    # 2. 提取表和别名
    for token in parsed.tokens:
        if token.ttype is Whitespace:
            continue
        if token.ttype is DML and token.value.upper() == 'SELECT':
            continue
        if token.ttype is Keyword and token.value.upper() == 'FROM':
            from_seen = True
            continue
        if from_seen and (isinstance(token, IdentifierList) or isinstance(token, Identifier)):
            identifiers = [token] if isinstance(token, Identifier) else token.get_identifiers()
            for idf in identifiers:
                # 提取表名和别名，转换为小写
                table_name = idf.get_real_name().lower()
                alias = (idf.get_alias() or table_name).lower()
                local_alias_map[alias] = table_name
            from_seen = False  # 只处理一次FROM子句
        if isinstance(token, Where):
            # 解析 WHERE 条件
            conditions = parse_conditions(token)
            for left, op, right in conditions:
                left = left.strip().lower()
                op = op.strip()
                right = right.strip().lower()
                # 检查是否为连接条件
                if op == '=' and '.' in left and '.' in right:
                    left_table_alias = left.split('.')[0]
                    right_table_alias = right.split('.')[0]
                    left_col = local_alias_map.get(left_table_alias, left_table_alias) + '.' + left.split('.')[1]
                    right_col = local_alias_map.get(right_table_alias, right_table_alias) + '.' + right.split('.')[1]
                    join_condition = frozenset([left_col.lower(), right_col.lower()])
                    actual_joins.append(join_condition)
                else:
                    # 处理过滤条件
                    if '.' in left:
                        table_alias = left.split('.')[0]
                        column_name = left.split('.')[1]
                        col_full_name = local_alias_map.get(table_alias, table_alias) + '.' + column_name
                        actual_filters.append((col_full_name.lower(), op, right))
            break  # 假设只有一个 WHERE 子句

    # 3. 收集所有的列并建立索引映射
    # 修改此部分，仅对 filter_list 中的列建立索引映射
    # 将 filter_list 中的列名全部转换为小写
    filter_list_lower = [col.lower() for col in filter_list]
    column_index_map = {}
    for idx, col_name in enumerate(filter_list_lower):
        column_index_map[col_name] = idx
    total_filter_columns = len(filter_list_lower)

    # 4. 初始化编码向量
    join_encoding = np.zeros(len(join_list), dtype=int)
    filter_encoding = np.zeros(3 * total_filter_columns)

    # 5. 构建连接图并计算传递闭包
    parent = dict()
    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]  # 路径压缩
            u = parent[u]
        return u
    def union(u, v):
        pu, pv = find(u), find(v)
        if pu != pv:
            parent[pv] = pu
    # 初始化父指针
    columns_in_joins = set()
    for join in actual_joins:
        columns_in_joins.update(join)
    for col in columns_in_joins:
        parent[col] = col
    # 联合连接的列
    for join in actual_joins:
        cols = list(join)
        if len(cols) == 2:
            u, v = cols
            if u in parent and v in parent:
                union(u, v)
    # 生成所有隐式连接
    connected_components = defaultdict(set)
    for col in columns_in_joins:
        root = find(col)
        connected_components[root].add(col)
    implied_joins = set()
    for group in connected_components.values():
        cols = list(group)
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                implied_joins.add(frozenset([cols[i], cols[j]]))
    all_joins = set(actual_joins) | implied_joins

    # 将 join_list 中的连接条件和 all_joins 中的连接条件都转换为小写
    join_list_lower = [frozenset([col.lower() for col in join]) for join in join_list]
    all_joins_lower = set(frozenset([col.lower() for col in join]) for join in all_joins)

    # 6. 编码连接条件
    for i, possible_join in enumerate(join_list_lower):
        if possible_join in all_joins_lower:
            join_encoding[i] = 1

    # 7. 编码过滤条件
    for col_full_name, op, right in actual_filters:
        if col_full_name in filter_list_lower:
            col_index = column_index_map[col_full_name]
            table_name, column_name = col_full_name.split('.')
            # 将表名和列名转换为小写
            table_name = table_name.lower()
            column_name = column_name.lower()
            min_val, max_val = range_dict[table_name][column_name]
            try:
                value = float(right)
                normalized_value = normalize_value(value, min_val, max_val)
                filter_encoding[3 * col_index] = 1  # 有过滤条件
                if op in ['>=', '>']:
                    filter_encoding[3 * col_index + 1] = normalized_value
                elif op in ['<=', '<']:
                    filter_encoding[3 * col_index + 2] = normalized_value
                elif op == '=':
                    # 等于视为同时存在下界和上界
                    filter_encoding[3 * col_index + 1] = normalized_value
                    filter_encoding[3 * col_index + 2] = normalized_value
            except ValueError:
                continue  # 无法将右侧转换为浮点数，跳过
        else:
            continue  # 列不在 filter_list 中，跳过

    # 8. 合并编码向量
    encoding_vector = np.concatenate([join_encoding, filter_encoding])

    return encoding_vector

def compute_sql_distance(encoding1, encoding2, join_list_length, filter_list_length, d_max=2, w_join=0.5, w_filter=0.5):
    """
    Compute the distance between two SQL query encodings.

    :param encoding1: Encoding vector for query 1.
    :param encoding2: Encoding vector for query 2.
    :param join_list_length: Length of the join encoding.
    :param filter_list_length: Number of filter columns.
    :param d_max: Maximum penalty for missing filters.
    :param w_join: Weight for the join distance.
    :param w_filter: Weight for the filter distance.
    :return: Total distance between the two queries.
    """
    # Split the encodings into join and filter parts
    join1 = encoding1[:join_list_length]
    join2 = encoding2[:join_list_length]
    filter1 = encoding1[join_list_length:]
    filter2 = encoding2[join_list_length:]
    
    # Compute join distance (Hamming distance)
    D_join = np.sum(join1 != join2)
    D_join_norm = D_join / join_list_length if join_list_length > 0 else 0
    
    # Compute filter distance
    total_filter_distance = 0
    max_filter_distance = filter_list_length * d_max
    for i in range(filter_list_length):
        idx = 3 * i
        f1_present = filter1[idx] == 1
        f2_present = filter2[idx] == 1
        if f1_present and f2_present:
            lb_diff = abs(filter1[idx + 1] - filter2[idx + 1])
            ub_diff = abs(filter1[idx + 2] - filter2[idx + 2])
            d_i = lb_diff + ub_diff
        elif f1_present != f2_present:
            # Assign maximum penalty
            d_i = d_max
        else:
            d_i = 0  # Filter absent in both queries
        total_filter_distance += d_i
    D_filter_norm = total_filter_distance / max_filter_distance if max_filter_distance > 0 else 0
    
    # Combine distances
    D_total = w_join * D_join_norm + w_filter * D_filter_norm
    return D_total

def compute_distance_matrix(encodings, join_list_length, filter_list_length, d_max=2, w_join=0.5, w_filter=0.5):
    """
    计算所有 SQL 查询编码之间的距离矩阵。

    :param encodings: 编码列表，形状为 (N, D)
    :return: 距离矩阵，形状为 (N, N)
    """
    def compute_distance_batch(encodings, index_pairs, join_list_length, filter_list_length, d_max, w_join, w_filter):
        """计算一批编码之间的距离"""
        distances = {}
        for i, j in index_pairs:
            dist = compute_sql_distance(
                encodings[i], encodings[j],
                join_list_length, filter_list_length,
                d_max, w_join, w_filter
            )
            distances[(i, j)] = dist
        return distances  # 返回字典，包含距离和对应的索引
    
    N = len(encodings)
    distance_matrix = np.zeros((N, N))
    batch_size=100
    index_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]  # 创建所有需要计算的 (i, j) 对

    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = {}
        for start in range(0, len(index_pairs), batch_size):
            end = min(start + batch_size, len(index_pairs))
            batch = index_pairs[start:end]
            futures[executor.submit(compute_distance_batch, encodings, batch, join_list_length, filter_list_length, d_max, w_join, w_filter)] = batch

        # 处理所有结果
        for future in as_completed(futures):
            try:
                distances = future.result()
                # 处理结果
                for (i, j), dist in distances.items():
                    distance_matrix[i, j] = dist
                    distance_matrix[j, i] = dist  # 对称性
            except Exception as e:
                print(f"Task failed: {e}")

    return distance_matrix

def sql_coreset(subset_sql_queries, range_dict_path):

    with open("/home/lgn/source/updatedLearnedQO/active_query_optimizer/data/test/stats_test_sql.txt", 'r') as f:
        lines = f.readlines()
    all_sql_queries = [line.split('#####')[1].strip() for line in lines]
    join_list, filter_list, alias_map = extract_join_and_filter_lists(all_sql_queries)

    # with open(range_dict_path, 'r') as f:
    #     range_dict = json.load(f)
    
    # new_range_dict = {alias_map[k]:v for k, v in range_dict.items()}

    # encodings = []
    # for sql_query in subset_sql_queries:
    #     encoding = encode_sql_query(sql_query, alias_map, new_range_dict, join_list, filter_list)
    #     encodings.append(encoding)
    #     # print(sql_query)
    #     # print("编码结果：", encoding[:len(join_list)])
    #     # print("编码结果：", encoding[len(join_list):])
    # encodings = np.stack(encodings)
    # np.save('../data/tmp/stats_test_sql_encodings.npy', encodings)

    encodings = np.load('../data/tmp/stats_test_sql_encodings.npy')

    distance_matrix = compute_distance_matrix(
        encodings, len(join_list), len(filter_list)
    )
    
    num_groups = 5
    # 使用 KMedoids 聚类
    kmedoids = KMedoids(n_clusters=num_groups, metric='precomputed', random_state=42)
    labels = kmedoids.fit_predict(distance_matrix)

    # 根据标签分组
    groups = [[] for _ in range(num_groups)]
    for idx, label in enumerate(labels):
        groups[label].append(idx)  # 将样本添加到对应簇的组中
    for group in groups:
        print(group)

# 示例使用
if __name__ == "__main__":
    with open("data/test/stats_test_sql.txt", 'r') as f:
        lines = f.readlines()
    sql_queries = [line.split('#####')[1].strip() for line in lines]

    join_list, filter_list, alias_map = extract_join_and_filter_lists(sql_queries)

    with open('infos/stats/range_dict', 'r') as f:
        range_dict = json.load(f)
    
    new_range_dict = {alias_map[k]:v for k, v in range_dict.items()}
    with open('data/unlabeled_train_data/stats_train_pool.txt', 'r') as f:
        lines = f.readlines()
    sql_queries = [line.split('#####')[1].strip() for line in lines]

    encodings = []
    for sql_query in sql_queries:
        encoding = encode_sql_query(sql_query, alias_map, new_range_dict, join_list, filter_list)
        encodings.append(encoding)
        # print(sql_query)
        # print("编码结果：", encoding[:len(join_list)])
        # print("编码结果：", encoding[len(join_list):])
    encodings = np.stack(encodings)
    np.save('data/tmp/stats_test_sql_encodings.npy', encodings)

    encodings = np.load('data/tmp/stats_test_sql_encodings.npy')

    distance_matrix = compute_distance_matrix(
        encodings, len(join_list), len(filter_list)
    )
    num_groups = 5
    # 使用 KMedoids 聚类
    kmedoids = KMedoids(n_clusters=num_groups, metric='precomputed', random_state=42)
    labels = kmedoids.fit_predict(distance_matrix)

    # 根据标签分组
    groups = [[] for _ in range(num_groups)]
    for idx, label in enumerate(labels):
        groups[label].append(idx)  # 将样本添加到对应簇的组中
    for group in groups:
        print(group)
    




