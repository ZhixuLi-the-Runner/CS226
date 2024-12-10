#!/bin/bash

# 定义输入文件
demo_file="/home/jiachenl/Long_Horizon_Manipulation/Code/demo_paths.txt"
model_file="testing_demos"

# 判断文件是否存在
if [ ! -f "$demo_file" ]; then
  echo "File $demo_file not found!"
  exit 1
fi

if [ ! -f "$model_file" ]; then
  echo "File $model_file not found!"
  exit 1
fi

# 使用 paste 命令将两个文件的内容逐行并列读取
paste -d' ' "$model_file" "$demo_file" | while IFS=' ' read -r model_path demo_path
do
  # 确保当前行不为空
  if [ -n "$model_path" ] && [ -n "$demo_path" ]; then
    echo "Running script with model_path: $model_path and demo_path: $demo_path"
    DISPLAY=:1 python main.py --cur_model_dir "$model_path" --cur_demo_dir "$demo_path"
  fi
done
