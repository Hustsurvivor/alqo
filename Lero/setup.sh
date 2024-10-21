#!/bin/bash
plan_path="example/combined_imdb_train_processed.txt"
test_plan_path="example/combined_imdb_test_processed.txt"

training_data="combined_imdb_train_processed_100%.exploration"
model_name="lero_model_imdb_100%"
dict_dir="./"

# 创建exploration文件
python create_training_file.py \
    --plan_path $plan_path \
    --output_path $training_data \

# 训练
python lero_train.py \
    --training_data $training_data \
    --model_name $model_name \
    --training_type 1

# 预测
python lero_test.py \
    --model_path $model_name \
    --plan_path $test_plan_path \
    --dict_dir $dict_dir
