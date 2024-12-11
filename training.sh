#

# designate the path to the training config, dataset, and training script
training_config="/home/jiachenl/cs226/Config/bc.json"
train_py_file="/home/jiachenl/cs226/robomimic/robomimic/scripts/train.py"


# read dataset file paths from a txt file and store them in the datasets variable
dataset_file="/home/jiachenl/cs226/Config/demo_paths.txt"
datasets=$(cat "$dataset_file")



# run the training script
for dataset in $datasets ; do
    echo "Training on dataset $dataset"
    python "$train_py_file" --config "$training_config" --dataset "$dataset"
done

DISPLAY=:1 python "$train_py_file" --config "$training_config" --dataset "$dataset"