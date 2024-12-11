# CS226 Project
## Group: 
Name: Saturday

Members:

Zhixu Li

Zhaorui Yang


...


## 1.Env installation
First, you need to install `robomimic` and `calvin` environment.
You can do it by executing the following commands:
```bash
conda create -n robomimic_venv python=3.8.0
conda activate robomimic_venv
conda install pytorch==2.0.0 torchvision==0.15.1 -c pytorch
pip install robomimic
git clone --recurse-submodules https://github.com/mees/calvin.git
export CALVIN_ROOT=$(pwd)/calvin
cd $CALVIN_ROOT
sh install.sh
```
Otherwise, here is the link for robomimic:https://robomimic.github.io/docs/introduction/installation.html

And here is the link for calvin: https://github.com/mees/calvin

Once you've set the environment, you can download data from the following link:
http://calvin.cs.uni-freiburg.de/dataset/task_ABCD_D.zip

Since the data is very large, and it requires our code to do the pre-processing, you can also download a small test set that we've processed:
https://drive.google.com/drive/folders/1SYjYjfmmZcw73r-8vZerfVtLkxnab1XI

If you download the large full dataset, you can goto section 2 to process the original data.

If you download the small test set, you can goto section 3 to train the model.

## 2. Data Processing with Spark
To process the original data, you need to install spark and pyspark following the documentation here: https://spark.apache.org/docs/latest/api/python/getting_started/install.html

Then you can run the following python scripts once you modify the data path for them:
```bash
python random_trim.py
spark-submit validate_with_spark.py
```


## 3. Training models
To train the model, you need to modify the `training_config` ,`train_py_file` and `dataset_file` in `training.sh`

Then you can just run the following command:
```bash
bash training.sh
```
You may asked to install wandb (https://wandb.ai/site/), which is a tool for tracking the training process.
## 4. Testing models
After training the model, you can test the model by running the following command:
```bash
python main.py
```
And if you want to costumize the test, you can modify the `models_dir` and  `cur_demo_dir` in `Simulation/Testing.py`