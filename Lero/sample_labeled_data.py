import sqlparse
from sqlparse.sql import Where, Identifier, IdentifierList, Token, Function, Parenthesis, Comparison
from sqlparse.tokens import Keyword, DML, Whitespace, Operator, Comparison as ComparisonToken
from collections import defaultdict
from typing import List, Dict, Tuple
import numpy as np
import random
class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, item):
        """查找 item 的根节点"""
        if item not in self.parent:
            self.parent[item] = item
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, item1, item2):
        """合并两个元素"""
        root1 = self.find(item1)
        root2 = self.find(item2)
        if root1 != root2:
            self.parent[root2] = root1

def parse_conditions(token_list):
    """递归解析 WHERE 子句，提取条件列表"""
    conditions = []
    tokens = list(token_list.flatten())
    idx = 0

    while idx < len(tokens):
        token = tokens[idx]

        if token.ttype is sqlparse.tokens.Whitespace:
            idx += 1
            continue

        if token.ttype == sqlparse.tokens.Operator.Comparison:
            # 提取比较条件
            if hasattr(token.parent, 'left'):
                left = str(token.parent.left).strip().lower()
            else:
                left = ''
            op = str(token).strip()
            if hasattr(token.parent, 'right'):
                right = str(token.parent.right).strip().lower()
            else:
                right = ''
            conditions.append((left, op, right))
            idx += 1  # 继续下一个 token
        elif token.is_group:
            # 递归解析嵌套条件
            conditions.extend(parse_conditions(token))
            idx += 1
        else:
            idx += 1

    return conditions

def extract_join_and_filter_lists(sql_query):
    """
    提取 SQL 查询中的连接条件和过滤条件。
    
    参数:
    - sql_query: SQL 查询字符串
    
    返回:
    - join_conditions (List[frozenset]): 连接条件列表，形如 [frozenset([a.a1, b.b1]), frozenset([a.a2, c.c2])]
    - filter_conditions (List[str]): 过滤条件列表，形如 [a, b, c]
    """
    uf = UnionFind()  # 使用并查集来处理连接条件
    filter_set = set()  # 用于存储过滤条件的表
    
    parsed = sqlparse.parse(sql_query)[0]
    alias_map = {}
    from_seen = False

    tokens = parsed.tokens
    idx = 0
    while idx < len(tokens):
        token = tokens[idx]
        if token.ttype is sqlparse.tokens.Whitespace:
            idx += 1
            continue
        if token.ttype is sqlparse.tokens.DML and token.value.upper() == 'SELECT':
            idx += 1
            continue
        if token.ttype is sqlparse.tokens.Keyword and token.value.upper() == 'FROM':
            from_seen = True
            idx += 1
            continue
        if from_seen:
            if isinstance(token, sqlparse.sql.IdentifierList):
                identifiers = token.get_identifiers()
            elif isinstance(token, sqlparse.sql.Identifier):
                identifiers = [token]
            else:
                identifiers = []
            for idf in identifiers:
                table_name = idf.get_real_name().lower()
                alias = (idf.get_alias() or table_name).lower()
                alias_map[alias] = table_name
            from_seen = False
        if isinstance(token, sqlparse.sql.Where):
            # 解析 WHERE 子句中的条件
            conditions = parse_conditions(token)
            for left, op, right in conditions:
                op = op.strip()

                # 检查是否为连接条件 (形式如 a.a1 = b.b1)
                if op == '=' and '.' in left and '.' in right:
                    left_table_alias = left.split('.')[0]
                    right_table_alias = right.split('.')[0]
                    # 使用并查集将相连的列合并在一起
                    uf.union(left.lower(), right.lower())
                else:
                    # 将存在过滤条件的表加入 filter_set
                    if '.' in left:
                        table_alias = left.split('.')[0]
                        filter_set.add(alias_map.get(table_alias, table_alias))
            break  # 假设只有一个 WHERE 子句
        idx += 1

    # 处理连接条件，将并查集中的集合以 frozenset 形式返回
    group_map = defaultdict(set)
    for item in uf.parent:
        root = uf.find(item)
        group_map[root].add(item)
    
    join_conditions = [frozenset(group) for group in group_map.values()]
    
    # 过滤条件：包含过滤条件的表
    filter_conditions = list(filter_set)

    return join_conditions, filter_conditions

def generate_template_id(join_list: List[frozenset], filter_list: List[str]) -> str:
    """
    生成模板的唯一标识符。
    
    参数：
    - join_list (List[frozenset]): 连接条件列表，每个连接条件是一个 frozenset。
    - filter_list (List[str]): 过滤列列表。
    
    返回：
    - template_id (str): 模板的唯一标识符。
    """
    # 对连接条件排序并转换为字符串
    sorted_joins = sorted(['_'.join(sorted(j)) for j in join_list])
    join_str = '|'.join(sorted_joins)
    
    # 对过滤条件排序并转换为字符串
    sorted_filters = sorted(filter_list)
    filter_str = '|'.join(sorted_filters)
    
    # 组合连接条件和过滤条件作为模板标识符
    template_id = f"Joins:{join_str}|Filters:{filter_str}"
    return template_id

def generate_template_id_from_joins(join_list):
    """
    根据 join_list 生成模板的唯一标识符。
    
    参数:
    - join_list (List[frozenset]): 连接条件列表
    
    返回:
    - template_id (str): 根据 join_list 生成的模板标识符
    """
    # 对连接条件列表进行排序，并将每个 frozenset 转换为字符串
    sorted_joins = sorted(['_'.join(sorted(join)) for join in join_list])
    # 将排序后的连接条件组合成一个字符串，作为模板 ID
    template_id = '|'.join(sorted_joins)
    return template_id
def group_sql_by_template(sql_queries: List[str]) -> Tuple[Dict[str, List[int]], List[str]]:
    """
    将 SQL 查询按模板分组，并统计每个模板的 SQL 累积索引。
    
    参数：
    - sql_queries (List[str]): SQL 查询字符串的列表。
    
    返回：
    - template_to_sql_indices (Dict[str, List[int]]): 模板到 SQL 累积索引的映射。
    - templates (List[str]): 所有模板的列表。
    """
    template_to_sql_indices = defaultdict(list)
    
    for idx, sql_query in enumerate(sql_queries):
        # 提取连接条件和过滤条件
        join_list, filter_list = extract_join_and_filter_lists(sql_query)
        
        # 生成模板标识符
        # template_id = generate_template_id(join_list, filter_list)
        template_id = generate_template_id_from_joins(join_list)
        
        # 分组
        template_to_sql_indices[template_id].append(idx)
    
    templates = list(template_to_sql_indices.keys())
    return template_to_sql_indices, templates
def sample_sql_queries(template_to_sql_indices: Dict[str, List[int]], 
                       proportions: np.ndarray, 
                       total_samples: int) -> List[int]:
    """
    根据模板采样比例，从每个模板中采样 SQL 查询。

    参数：
    - template_to_sql_indices: 模板到 SQL 索引列表的映射。
    - proportions: 每个模板的采样比例数组。
    - total_samples: 需要采样的 SQL 总数。

    返回：
    - sampled_sql_indices: 采样得到的 SQL 查询索引列表。
    """
    templates = list(template_to_sql_indices.keys())
    num_templates = len(templates)
    
    # 计算每个模板需要采样的 SQL 数量
    samples_per_template = (proportions * total_samples).astype(int)
    
    # 确保总采样数量等于 total_samples
    remainder = total_samples - np.sum(samples_per_template)
    if remainder > 0:
        # 随机分配剩余的样本数
        for _ in range(remainder):
            idx = np.random.choice(num_templates)
            samples_per_template[idx] += 1
    elif remainder < 0:
        # 减少多余的样本数
        for _ in range(-remainder):
            idx = np.random.choice(num_templates)
            if samples_per_template[idx] > 0:
                samples_per_template[idx] -= 1

    # 根据每个模板的样本数量进行采样
    sampled_sql_indices = []
    for i, template in enumerate(templates):
        sql_indices = template_to_sql_indices[template]
        num_samples = samples_per_template[i]
        if num_samples > len(sql_indices):
            # 如果需要的样本数超过了可用的 SQL 数量，则全部使用
            sampled_sql_indices.extend(sql_indices)
            # 进行重复采样
            sampled_sql_indices.extend(random.choices(sql_indices, k=num_samples-len(sql_indices)))
        else:
            sampled_sql_indices.extend(random.sample(sql_indices, num_samples))

    return sampled_sql_indices
def generate_dirichlet_proportions(alpha, num_templates):
    """
    使用 Dirichlet 分布生成模板的采样比例。

    参数：
    - alpha: Dirichlet 分布的超参数，控制不平衡度。
    - num_templates: 模板数量。

    返回：
    - proportions: 每个模板的采样比例数组。
    """
    return np.random.dirichlet([alpha] * num_templates)
def create_imbalanced_sql_subset(sql_queries: List[str], 
                                 alpha: float, 
                                 total_samples: int) -> Tuple[List[str], List[int]]:
    """
    使用 Dirichlet 分布采样不均衡的 SQL 子集。

    参数：
    - sql_queries: 所有 SQL 查询的列表。
    - alpha: Dirichlet 分布的超参数，控制不平衡度。
    - total_samples: 需要采样的 SQL 总数。

    返回：
    - subset_sql_queries: 采样得到的 SQL 查询子集。
    """
    # 分组 SQL 查询并统计模板频率
    template_to_sql_indices, templates = group_sql_by_template(sql_queries)
    num_templates = len(templates)
    
    # 使用 Dirichlet 分布生成模板采样比例
    proportions = generate_dirichlet_proportions(alpha, num_templates)
    
    # 根据模板采样比例采样 SQL 查询
    sampled_sql_indices = sample_sql_queries(template_to_sql_indices, proportions, total_samples)
    
    # 构建 SQL 子集
    subset_sql_queries = [sql_queries[idx] for idx in sampled_sql_indices]
    
    return subset_sql_queries, sampled_sql_indices

if __name__ == '__main__':
    with open("data/unlabeled_train_data/stats_train_pool.txt", 'r') as f:
        lines = f.readlines()
    sql_queries = [line.split('#####')[1].strip() for line in lines]


    total_samples = 1000
    alpha=0.2
    imbalanced_sql_subset, _ = create_imbalanced_sql_subset(sql_queries, alpha, total_samples)
