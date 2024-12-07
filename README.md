# CS226 Project
## Group: 
Name: Saturday

Members:

Zhixu Li

...


## Env installation
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


