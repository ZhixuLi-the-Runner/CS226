import argparse  # 导入 argparse 模块，用于命令行参数解析
from Ours_code.Environment import Environment
import hydra  # 导入 Hydra 库，用于配置管理
from omegaconf import OmegaConf
from hydra import initialize, compose


def load_config():
    # 初始化 Hydra 并指定配置文件路径
    with initialize(config_path="./calvin/calvin_env/conf"):
        # 手动加载指定的配置文件
        cfg = compose(config_name="config_data_collection")
        print(OmegaConf.to_yaml(cfg))  # 打印配置内容以进行验证
    return cfg


# 使用 Hydra 进行配置
def main():
    # 加载配置
    cfg = load_config()



    # 创建 Environment 实例并运行
    env = Environment(cfg=cfg)
    env.run()


if __name__ == '__main__':
    main()