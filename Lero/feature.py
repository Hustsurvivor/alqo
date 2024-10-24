import json
from abc import ABCMeta, abstractmethod
from ImportantConfig import RANGE_DICT_PATH
import numpy as np

FEATURE_LIST = ['Node Type', 'Startup Cost',
                'Total Cost', 'Plan Rows', 'Plan Width']
LABEL_LIST = ['Actual Startup Time', 'Actual Total Time', 'Actual Self Time']

UNKNOWN_OP_TYPE = "Unknown"
SCAN_TYPES = ["Seq Scan", "Index Scan", "Index Only Scan", 'Bitmap Heap Scan']
JOIN_TYPES = ["Nested Loop", "Hash Join", "Merge Join"]
OTHER_TYPES = ['Bitmap Index Scan']
OP_TYPES = [UNKNOWN_OP_TYPE, "Hash", "Materialize", "Sort", "Aggregate", "Incremental Sort", "Limit"] \
    + SCAN_TYPES + JOIN_TYPES + OTHER_TYPES


def json_str_to_json_obj(json_data):
    json_obj = json.loads(json_data)
    if type(json_obj) == list:
        assert len(json_obj) == 1
        json_obj = json_obj[0]
        assert type(json_obj) == dict
    return json_obj

def extract_conditions(filter_str: str):
    # 去掉外层的空格和括号
    filter_str = filter_str.strip()
    if filter_str.startswith('(') and filter_str.endswith(')'):
        filter_str = filter_str[1:-1].strip()
    
    parts = filter_str.split('AND')
    if len(parts) == 1: 
        return parts

    conditions = []
    for part in parts:
        conditions.extend(extract_conditions(part))
    
    return conditions

def decompose_condition(condition: str):
    # 去掉空格并分割字符串
    condition = condition.strip()
    parts = []

    # 查找运算符的位置
    for operator in ['!=', '<=', '>=', '<>', '<', '>', '=']:
        if operator in condition:
            left, right = condition.split(operator, 1)  # 分割一次
            parts.append(left.strip())  # 去掉空格
            parts.append(operator)  # 添加运算符
            parts.append(right.strip())  # 去掉空格
            break

    return parts


class FeatureGenerator():

    def __init__(self) -> None:
        self.normalizer = None
        self.scaler = None
        self.feature_parser = None

    def fit(self, trees):
        exec_times = []
        startup_costs = []
        total_costs = []
        rows = []
        input_relations = set()
        
        join_filters = set()
        filter_fields = set()
        
        rel_type = set()
        
        def recurse(n):
            startup_costs.append(n["Startup Cost"])
            total_costs.append(n["Total Cost"])
            rows.append(n["Plan Rows"])
            rel_type.add(n["Node Type"])
            if "Relation Name" in n:
                # base table
                input_relations.add(n["Relation Name"])
            
            if "Join Filter" in n or "Index Cond" in n:
                filter_str =  n["Join Filter"] if "Join Filter" in n else n["Index Cond"]
                
                conditions = extract_conditions(filter_str)
                for condition in conditions:
                    parts = decompose_condition(condition)
                    if parts[1] == '=':
                        fields = sorted([parts[0], parts[2]])
                        join_filters.add(fields[0]+' '+parts[1]+' '+fields[1])
                    else:
                        join_filters.add(condition)
            
            if "Filter" in n:
                conditions = extract_conditions(n["Filter"])
                for condition in conditions:
                    filter_fields.add(decompose_condition(condition)[0])
            
            if "Plans" in n:
                for child in n["Plans"]:
                    recurse(child)

        for tree in trees:
            json_obj = json_str_to_json_obj(tree)
            if "Execution Time" in json_obj:
                exec_times.append(float(json_obj["Execution Time"]))
            recurse(json_obj["Plan"])

        startup_costs = np.array(startup_costs)
        total_costs = np.array(total_costs)
        rows = np.array(rows)

        startup_costs = np.log(startup_costs + 1)
        total_costs = np.log(total_costs + 1)
        rows = np.log(rows + 1)

        startup_costs_min = np.min(startup_costs)
        startup_costs_max = np.max(startup_costs)
        total_costs_min = np.min(total_costs)
        total_costs_max = np.max(total_costs)
        rows_min = np.min(rows)
        rows_max = np.max(rows)

        print("RelType : ", rel_type)
        print("join_filters: ", join_filters)
        print("filter_fields: ", filter_fields)

        if len(exec_times) > 0:
            exec_times = np.array(exec_times)
            exec_times = np.log(exec_times + 1)
            exec_times_min = np.min(exec_times)
            exec_times_max = np.max(exec_times)
            self.normalizer = Normalizer(
                {"Execution Time": exec_times_min, "Startup Cost": startup_costs_min,
                 "Total Cost": total_costs_min, "Plan Rows": rows_min},
                {"Execution Time": exec_times_max, "Startup Cost": startup_costs_max,
                 "Total Cost": total_costs_max, "Plan Rows": rows_max})
        else:
            self.normalizer = Normalizer(
                {"Startup Cost": startup_costs_min,
                 "Total Cost": total_costs_min, "Plan Rows": rows_min},
                {"Startup Cost": startup_costs_max,
                 "Total Cost": total_costs_max, "Plan Rows": rows_max})
            
        with open(RANGE_DICT_PATH, 'r') as f:
            ranges = json.load(f)
        self.scaler = MinMaxScaler(ranges)
        
        self.feature_parser = AnalyzeJsonParser(self.normalizer, self.scaler, list(input_relations), list(join_filters), list(filter_fields))

    def transform(self, trees):
        local_features = []
        y = []
        for tree in trees:
            json_obj = json_str_to_json_obj(tree)
            if type(json_obj["Plan"]) != dict:
                json_obj["Plan"] = json.loads(json_obj["Plan"])
            local_feature = self.feature_parser.extract_feature(
                json_obj["Plan"])
            local_features.append(local_feature)

            if "Execution Time" in json_obj:
                label = float(json_obj["Execution Time"])
                if self.normalizer.contains("Execution Time"):
                    label = self.normalizer.norm(label, "Execution Time")
                y.append(label)
            else:
                y.append(None)
        return local_features, y


class SampleEntity():
    def __init__(self, node_type: np.ndarray, startup_cost: float, total_cost: float,
                 rows: float, width: int,
                 left, right,
                 startup_time: float, total_time: float,
                 input_tables: list, encoded_input_tables: list, 
                 join_filters:list, encoded_join_filters: list,
                 filters: list,encoded_filters: list) -> None:
        self.node_type = node_type
        self.startup_cost = startup_cost
        self.total_cost = total_cost
        self.rows = rows
        self.width = width
        self.left = left
        self.right = right
        self.startup_time = startup_time
        self.total_time = total_time
        self.input_tables = input_tables
        self.encoded_input_tables = encoded_input_tables
        self.join_filters = join_filters
        self.encoded_join_filters = encoded_join_filters
        self.filters = filters
        self.encoded_filters = encoded_filters
        

    def __str__(self):
        return "{%s, %s, %s, %s, %s, [%s], [%s], %s, %s, [%s], [%s]}" % (self.node_type,
                                                                        self.startup_cost, self.total_cost, self.rows,
                                                                        self.width, self.left, self.right,
                                                                        self.startup_time, self.total_time,
                                                                        self.input_tables, self.encoded_input_tables)

    def get_feature(self):
        # return np.hstack((self.node_type, np.array([self.width, self.rows])))
        return np.hstack((self.node_type, 
                          np.array(self.encoded_input_tables),  
                          np.array([self.width, self.rows]),
                          np.array(self.encoded_join_filters),
                          np.array(self.encoded_filters),))

    def get_left(self):
        return self.left

    def get_right(self):
        return self.right

    def subtrees(self):
        trees = []
        trees.append(self)
        if self.left is not None:
            trees += self.left.subtrees()
        if self.right is not None:
            trees += self.right.subtrees()
        return trees


class Normalizer():
    def __init__(self, mins: dict, maxs: dict) -> None:
        self._mins = mins
        self._maxs = maxs

    def norm(self, x, name):
        if name not in self._mins or name not in self._maxs:
            raise Exception("fail to normalize " + name)

        return (np.log(x + 1) - self._mins[name]) / (self._maxs[name] - self._mins[name])

    def inverse_norm(self, x, name):
        if name not in self._mins or name not in self._maxs:
            raise Exception("fail to inversely normalize " + name)

        return np.exp((x * (self._maxs[name] - self._mins[name])) + self._mins[name]) - 1

    def contains(self, name):
        return name in self._mins and name in self._maxs

class MinMaxScaler:
    def __init__(self, ranges):
        self.ranges = ranges
    
    def normalize(self, data, table_name, field_name):
        min_val, max_val = self.ranges[table_name][field_name]
        return [(x - min_val) / (max_val - min_val) for x in data]


class FeatureParser(metaclass=ABCMeta):

    @abstractmethod
    def extract_feature(self, json_data) -> SampleEntity:
        pass


# the json file is created by "EXPLAIN (ANALYZE, VERBOSE, COSTS, BUFFERS, TIMING, SUMMARY, FORMAT JSON) ..."
class AnalyzeJsonParser(FeatureParser):

    def __init__(self, normalizer: Normalizer, scaler: MinMaxScaler, input_relations: list, join_filters: list, filter_fields: list) -> None:
        self.normalizer = normalizer
        self.input_relations = input_relations
        self.join_filters = {f: i for i, f in enumerate(join_filters)}
        self.filter_fields = {f: i for i, f in enumerate(filter_fields)}
        self.scaler = scaler

        self.operators = {'=': 0, '!=': 1, '<>': 1, '<': 2, '>': 3, '<=': 4, '>=': 5}
        
    def extract_feature(self, json_rel) -> SampleEntity:
        left = None
        right = None
        input_relations = []

        if 'Plans' in json_rel:
            children = json_rel['Plans']
            assert len(children) <= 2 and len(children) > 0
            left = self.extract_feature(children[0])
            input_relations += left.input_tables

            if len(children) == 2:
                right = self.extract_feature(children[1])
                input_relations += right.input_tables
            else:
                right = SampleEntity(op_to_one_hot(UNKNOWN_OP_TYPE), 0, 0, 0, 0,
                                     None, None, 0, 0, [], self.encode_relation_names([]),
                                     [], self.encode_join_filters([]),
                                     [], self.encode_filters([])
                                     )

        node_type = op_to_one_hot(json_rel['Node Type'])
        # startup_cost = self.normalizer.norm(float(json_rel['Startup Cost']), 'Startup Cost')
        # total_cost = self.normalizer.norm(float(json_rel['Total Cost']), 'Total Cost')
        startup_cost = None
        total_cost = None
        rows = self.normalizer.norm(float(json_rel['Plan Rows']), 'Plan Rows')
        width = int(json_rel['Plan Width'])

        if json_rel['Node Type'] in SCAN_TYPES:
            input_relations.append(json_rel["Relation Name"])

        startup_time = None
        if 'Actual Startup Time' in json_rel:
            startup_time = float(json_rel['Actual Startup Time'])
        total_time = None
        if 'Actual Total Time' in json_rel:
            total_time = float(json_rel['Actual Total Time'])

        ### 增加filter编码
        join_filters = []
        if "Join Filter" in json_rel or "Index Cond" in json_rel:
                filter_str =  json_rel["Join Filter"] if "Join Filter" in json_rel else json_rel["Index Cond"]
                
                conditions = extract_conditions(filter_str)
                for condition in conditions:
                    parts = decompose_condition(condition)
                    if parts[1] == '=':
                        fields = sorted([parts[0], parts[2]])
                        join_filters.append(fields[0]+' '+parts[1]+' '+fields[1])
                    else:
                        join_filters.append(condition)
        
        filters = []
        if 'Filter' in json_rel:
            import sys
            conditions = extract_conditions(json_rel['Filter'])
            for condition in conditions:
                parts = decompose_condition(condition)
                
                value = float(parts[2].replace('::integer', '').replace('\'',''))
                table, field = parts[0].split('.')
                value = self.scaler.normalize([value], table.strip(), field.strip())[0]
                filters.append([parts[0], parts[1], value])  
                     

        return SampleEntity(node_type, startup_cost, total_cost, rows, width, left,
                            right, startup_time, total_time,
                            input_relations, self.encode_relation_names(input_relations),
                            join_filters, self.encode_join_filters(join_filters),
                            filters, self.encode_filters(filters)                            
                            )

    # 将表名进行编码。也是one_hot形式
    def encode_relation_names(self, l):
        encode_arr = np.zeros(len(self.input_relations) + 1)

        for name in l:
            if name not in self.input_relations:
                # -1 means UNKNOWN
                encode_arr[-1] += 1
            else:
                encode_arr[list(self.input_relations).index(name)] += 1
        return encode_arr
    
    # 将join_filter进行编码 one_hot形式 形如p.id = pl.id
    def encode_join_filters(self, l):
        encode_arr = np.zeros(len(self.join_filters) + 1, dtype=float)
        
        for name in l:
            if name in self.join_filters:
                encode_arr[self.join_filters[name]]+1
            else:
                encode_arr[-1] += 1
        
        return encode_arr
    
    # 将其他filter进行编码，形如p.id >= 1
    # l 是一个三元组列表，其中元素形如 (filed, op, value)
    # 对每一个field,编码为 ['=', '!=', '<>', '<', '>', '<=', '>='，value]
    def encode_filters(self, l):
        base = len(self.operators) -1 + 1 # 因为!= 和 <> 相同所以-1
        encode_arr = np.zeros(len(self.filter_fields)*base + 1, dtype=float)
        
        for field_filter in l:
            assert(len(field_filter) == 3)
            
            field = field_filter[0]
            op = field_filter[1]
            value = field_filter[2]
            
            if field in self.filter_fields:
                field_index = self.filter_fields[field]
                
                if op in self.operators:
                    op_index = self.operators[op]
                    
                    # 计算编码数组中的位置
                    encode_index = field_index * base + op_index
                    encode_arr[encode_index] += 1
                    value_index = field_index * base + base - 1
                    encode_arr[value_index] = value 
                
                # todo: 处理没见过的op
            else:
                encode_arr[-1] += 1 # 标记没有该field
            
        return encode_arr 
        

# 将操作类型（op_name）编码成one_hot形式
def op_to_one_hot(op_name):
    arr = np.zeros(len(OP_TYPES))
    if op_name not in OP_TYPES:
        arr[OP_TYPES.index(UNKNOWN_OP_TYPE)] = 1
    else:
        arr[OP_TYPES.index(op_name)] = 1
    return arr

########################################################## new

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

if __name__ == '__main__':
    X1, X2 = _load_pairwise_plans('../data/labeled_sql.txt')
    feature_generator = FeatureGenerator()
    feature_generator.fit(X1[:10] + X2[:10])
    
    X1, Y1 = feature_generator.transform(X1[10:20])
    X2, Y2 = feature_generator.transform(X2[10:20])