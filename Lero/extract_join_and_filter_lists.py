import sqlparse
from sqlparse.sql import Where, Identifier, IdentifierList, Token, Function, Parenthesis, Comparison
from sqlparse.tokens import Keyword, DML, Whitespace, Operator, Comparison as ComparisonToken
from typing import List, Tuple, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

def parse_conditions(token_list):
    """递归解析 WHERE 子句，提取条件列表"""
    conditions = []
    tokens = list(token_list.flatten())
    idx = 0

    while idx < len(tokens):
        token = tokens[idx]

        if token.ttype is Whitespace:
            idx += 1
            continue

        # 检查是否为比较操作符
        if token.ttype == Operator.Comparison:
            # 提取比较条件
            # 左侧操作数在 token.parent 的 left 属性
            if hasattr(token.parent, 'left'):
                left = str(token.parent.left).strip().lower()
            else:
                left = ''
            op = str(token).strip()
            # 右侧操作数在 token.parent 的 right 属性
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

def process_query(sql_query: str, global_alias_map: Dict[str, str]) -> Tuple[set, set]:
    """
    提取所有 SQL 查询中的连接条件、过滤列和别名映射，并将它们转换为小写。
    :param sql_queries: SQL 查询字符串的列表
    :return: join_list（连接条件列表），filter_list（过滤列列表），alias_map（别名映射字典）
    """
    join_set = set()
    filter_set = set()
    alias_map = {}
    from_seen = False

    parsed = sqlparse.parse(sql_query)[0]
    tokens = parsed.tokens
    idx = 0

    while idx < len(tokens):
        token = tokens[idx]
        if token.ttype is Whitespace:
            idx += 1
            continue
        if token.ttype is DML and token.value.upper() == 'SELECT':
            idx += 1
            continue
        if token.ttype is Keyword and token.value.upper() == 'FROM':
            from_seen = True
            idx += 1
            continue
        if from_seen:
            if isinstance(token, IdentifierList):
                identifiers = token.get_identifiers()
            elif isinstance(token, Identifier):
                identifiers = [token]
            else:
                identifiers = []
            for idf in identifiers:
                table_name = idf.get_real_name().lower()
                alias = (idf.get_alias() or table_name).lower()
                alias_map[alias] = table_name
                global_alias_map[alias] = table_name
            from_seen = False
        if isinstance(token, Where):
            conditions = parse_conditions(token)
            for left, op, right in conditions:
                op = op.strip()
                if op == '=' and '.' in left and '.' in right:
                    left_table_alias = left.split('.')[0]
                    right_table_alias = right.split('.')[0]
                    left_col = alias_map.get(left_table_alias, left_table_alias) + '.' + left.split('.')[1]
                    right_col = alias_map.get(right_table_alias, right_table_alias) + '.' + right.split('.')[1]
                    join_condition = frozenset([left_col.lower(), right_col.lower()])
                    join_set.add(join_condition)
                else:
                    if '.' in left:
                        table_alias = left.split('.')[0]
                        column_name = left.split('.')[1]
                        col_full_name = alias_map.get(table_alias, table_alias) + '.' + column_name.lower()
                        filter_set.add(col_full_name.lower())
            break
        idx += 1

    return join_set, filter_set

def extract_join_and_filter_lists(sql_queries: List[str]) -> Tuple[List[str], List[str], Dict[str, str]]:
    global_alias_map = {}

    join_sets = []
    filter_sets = []

    with ThreadPoolExecutor(max_workers=15) as executor:
        futures = {executor.submit(process_query, query, global_alias_map): query for query in sql_queries}
        for future in as_completed(futures):
            join_set, filter_set = future.result()
            join_sets.append(join_set)
            filter_sets.append(filter_set)

    # 合并所有集合
    final_join_set = set().union(*join_sets)
    final_filter_set = set().union(*filter_sets)

    return list(final_join_set), list(final_filter_set), global_alias_map
