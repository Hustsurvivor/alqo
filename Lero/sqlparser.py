from psqlparse import parse_dict
import psqlparse

comp_dict = {'=': 0, '>=': 1, '<=': 2}

def transExpr(expr):
    result = None
    if 'ColumnRef' in expr:
        result = expr['ColumnRef']['fields'][1]['String']['str']
    elif 'A_Const' in expr:
        result = expr['A_Const']['val'][0]['String']['str']
    
    return result
    
def extract_tables_and_filters(sql):
    # 解析SQL

    parse_result = parse_dict(sql)[0]["SelectStmt"]
    
    target = []
    tables = set()
    filters = set()
    
    ## get tables
    for x in parse_result['fromClause']:
        tables.add(x['RangeVar']['relname'])
    
    assert(len(parse_result['whereClause']) == 1)
    for x in parse_result['whereClause']['BoolExpr']['args']:
        expr = x['A_Expr']
        
        lexpr = transExpr(expr['lexpr'])
        rexpr = transExpr(expr['rexpr'])
        comp = expr['name'][0]['String']['str']
        comp = comp_dict[comp]
        filters.add((lexpr, comp, rexpr))

# 示例SQL查询
sql = "SELECT * FROM users as u, tags as t WHERE u.id = tags.id AND u.age > 20 AND t.country = 'USA'"
tables, filters = extract_tables_and_filters(sql)

print("Tables:", tables)
print("Filters:", filters)
