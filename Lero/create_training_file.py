import argparse
SEP = '#####'
## 对训练集进行处理，把训练集的plan前的qid删掉
def create_training_file(training_data_file, input_file_path):
    plans_list = []
    with open(input_file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            plans = line.strip().split(SEP)[1:]
            plans_list.append(SEP.join(plans))
    
    str = "\n".join(plans_list)
    
    with open(training_data_file, 'w') as f2:
        f2.write(str)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("create training file")
    parser.add_argument("--plan_path",
                        metavar="PATH",
                        help="Load the plans")
    parser.add_argument("--output_path",
                        metavar="PATH",
                        help="Output path")
    args = parser.parse_args()
    
    print('create exploration file:')
    print('plan path: ', args.plan_path)
    print('ourput path: ', args.output_path)
    create_training_file(args.output_path, args.plan_path)                   