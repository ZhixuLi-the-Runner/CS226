import logging  # 导入用于记录日志的模块
from math import pi  # 导入数学模块中的常数 pi
import os  # 导入操作系统接口模块
from pathlib import Path  # 导入用于处理文件路径的模块
import pickle  # 导入用于序列化和反序列化 Python 对象的模块
import pkgutil  # 导入用于 Python 包的实用工具模块
import re  # 导入正则表达式模块
import sys  # 导入系统特定参数和函数模块
import time  # 导入时间处理模块

import cv2  # 导入 OpenCV 库，用于图像处理
import gym  # 导入 OpenAI Gym 库，用于创建和使用环境
import gym.utils  # 导入 Gym 的实用工具
import gym.utils.seeding  # 导入 Gym 的随机种子工具
import hydra  # 导入 Hydra 库，用于配置管理
import numpy as np  # 导入 NumPy 库，用于数值计算
import pybullet as p  # 导入 PyBullet 库，用于物理仿真
import pybullet_utils.bullet_client as bc  # 导入 PyBullet 客户端工具
from sympy.physics.units import action
from traitlets import observe

from calvin_env.calvin_env.utils.utils import FpsController, get_git_commit_hash  # 从 calvin_env 的 utils 中导入 FPS 控制器和获取 Git 提交哈希的函数
from numpy.lib.utils import deprecate_with_doc
import torch

# 为该文件设置一个日志记录器
log = logging.getLogger(__name__)
from rich.traceback import install  # 从 rich 库中导入用于改进错误追踪的工具

install(show_locals=True)  # 安装 rich 的 traceback，显示局部变量


# 定义一个类 PlayTableSimEnv，继承自 Gym 的 Env 基类
class PlayTableSimEnv(gym.Env):
    def __init__(
            self,
            robot_cfg,  # 机器人配置
            seed,  # 随机种子
            use_vr,  # 是否使用虚拟现实模式
            bullet_time_step,  # PyBullet 时间步长
            cameras,  # 相机配置
            show_gui,  # 是否显示图形界面
            scene_cfg,  # 场景配置
            use_scene_info,  # 是否使用场景信息
            use_egl,  # 是否使用 EGL 渲染
            control_freq=30,  # 控制频率，默认为 30
    ):
        self.p = p  # 设置 PyBullet 客户端
        self.t = time.time()  # 初始化 FPS 计算的时间戳
        self.prev_time = time.time()  # 初始化前一帧时间戳
        self.fps_controller = FpsController(bullet_time_step)  # 创建一个 FPS 控制器实例
        self.use_vr = use_vr  # 是否使用 VR 模式
        self.show_gui = show_gui  # 是否显示 GUI
        self.use_scene_info = use_scene_info  # 是否使用场景信息
        self.cid = -1  # 客户端 ID 初始化为 -1
        self.ownsPhysicsClient = False  # 标记是否拥有物理客户端
        self.use_egl = use_egl  # 是否使用 EGL 渲染
        self.control_freq = control_freq  # 控制频率
        self.action_repeat = int(bullet_time_step // control_freq)  # 动作重复次数，用于加速仿真
        render_width = max([cameras[cam].width for cam in cameras]) if cameras else None  # 渲染宽度，取最大值
        render_height = max([cameras[cam].height for cam in cameras]) if cameras else None  # 渲染高度，取最大值
        self.initialize_bullet(bullet_time_step, render_width, render_height)  # 初始化 PyBullet
        self.np_random = None  # 初始化随机数生成器
        self.seed(seed)  # 设置随机种子
        # 实例化机器人和场景对象
        self.robot = hydra.utils.instantiate(robot_cfg, cid=self.cid)  # 使用 Hydra 实例化机器人配置
        self.scene = hydra.utils.instantiate(scene_cfg, p=self.p, cid=self.cid, np_random=self.np_random)  # 使用 Hydra 实例化场景配置

        self.load()  # 加载环境

        # 初始化相机，在场景加载后进行初始化以获取机器人 ID
        self.cameras = [
            hydra.utils.instantiate(
                cameras[name], cid=self.cid, robot_id=self.robot.robot_uid, objects=self.scene.get_objects()
            )
            for name in cameras
        ]

    def __del__(self):
        self.close()  # 析构函数，在对象销毁时关闭环境

    # 初始化 PyBullet 物理仿真
    def initialize_bullet(self, bullet_time_step, render_width, render_height):
        if self.cid < 0:  # 如果客户端 ID 小于 0，表示尚未连接到 PyBullet 服务器
            self.ownsPhysicsClient = True  # 设置拥有物理客户端标志
            if self.use_vr:  # 如果使用 VR 模式
                self.p = bc.BulletClient(connection_mode=p.SHARED_MEMORY)  # 使用共享内存模式连接
                cid = self.p._client  # 获取客户端 ID
                if cid < 0:  # 如果连接失败
                    log.error("Failed to connect to SHARED_MEMORY bullet server.\n" " Is it running?")  # 记录错误日志
                    sys.exit(1)  # 退出程序
                self.p.setRealTimeSimulation(enableRealTimeSimulation=1, physicsClientId=cid)  # 设置实时仿真模式
            elif self.show_gui:  # 如果显示 GUI
                self.p = bc.BulletClient(connection_mode=p.GUI)  # 使用 GUI 模式连接
                cid = self.p._client  # 获取客户端 ID
                if cid < 0:  # 如果连接失败
                    log.error("Failed to connect to GUI.")  # 记录错误日志
            elif self.use_egl:  # 如果使用 EGL 渲染
                options = f"--width={render_width} --height={render_height}"  # 设置 EGL 渲染选项
                self.p = p  # 使用直接模式连接
                cid = self.p.connect(p.DIRECT, options=options)  # 连接到 PyBullet 服务器
                p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=cid)  # 禁用 GUI
                p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0, physicsClientId=cid)  # 禁用分割预览
                p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0, physicsClientId=cid)  # 禁用深度缓冲预览
                p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0, physicsClientId=cid)  # 禁用 RGB 缓冲预览
                egl = pkgutil.get_loader("eglRenderer")  # 获取 EGL 渲染器插件
                log.info("Loading EGL plugin (may segfault on misconfigured systems)...")  # 记录加载 EGL 插件的日志
                if egl:  # 如果插件存在
                    plugin = p.loadPlugin(egl.get_filename(), "_eglRendererPlugin")  # 加载 EGL 渲染插件
                else:  # 如果插件不存在
                    plugin = p.loadPlugin("eglRendererPlugin")  # 加载默认的 EGL 渲染插件
                if plugin < 0:  # 如果插件加载失败
                    log.error("\nPlugin Failed to load!\n")  # 记录错误日志
                    sys.exit()  # 退出程序
                os.environ["PYOPENGL_PLATFORM"] = "egl"  # 设置环境变量用于 Tacto 渲染器
                log.info("Successfully loaded egl plugin")  # 记录成功加载插件的日志
            else:  # 使用直接连接模式
                self.p = bc.BulletClient(connection_mode=p.DIRECT)  # 直接连接到 PyBullet 服务器
                cid = self.p._client  # 获取客户端 ID
                if cid < 0:  # 如果连接失败
                    log.error("Failed to start DIRECT bullet mode.")  # 记录错误日志
            log.info(f"Connected to server with id: {cid}")  # 记录连接到服务器的 ID

            self.cid = cid  # 设置当前客户端 ID
            self.p.resetSimulation(physicsClientId=self.cid)  # 重置物理仿真
            self.p.setPhysicsEngineParameter(deterministicOverlappingPairs=1, physicsClientId=self.cid)  # 设置物理引擎参数
            self.p.configureDebugVisualizer(self.p.COV_ENABLE_GUI, 0)  # 配置调试可视化
            log.info(f"Connected to server with id: {self.cid}")  # 记录连接到服务器的 ID
            self.p.setTimeStep(1.0 / bullet_time_step, physicsClientId=self.cid)  # 设置时间步长
            return cid  # 返回客户端 ID

    def load(self):
        log.info("Resetting simulation")  # 记录重置仿真
        self.p.resetSimulation(physicsClientId=self.cid)  # 重置仿真环境
        log.info("Setting gravity")  # 记录设置重力
        self.p.setGravity(0, 0, -9.8, physicsClientId=self.cid)  # 设置重力参数

        self.robot.load()  # 加载机器人
        self.scene.load()  # 加载场景

    def close(self):
        if self.ownsPhysicsClient:  # 如果拥有物理客户端
            print("disconnecting id %d from server" % self.cid)  # 打印断开连接的信息
            if self.cid >= 0 and self.p is not None:  # 检查客户端 ID 和 PyBullet 实例
                try:
                    self.p.disconnect(physicsClientId=self.cid)  # 尝试断开连接
                except TypeError:
                    pass  # 如果发生 TypeError，忽略
        else:
            print("does not own physics client id")  # 如果没有拥有物理客户端，打印提示信息

    def render(self, mode="human"):
        """render 是 gym 的兼容函数"""
        rgb_obs, depth_obs = self.get_camera_obs()  # 获取相机观测
        if mode == "human":  # 如果是人类模式
            if "rgb_static" in rgb_obs:
                img = rgb_obs["rgb_static"][:, :, ::-1]  # 将 RGB 图像转换为 BGR
                cv2.imshow("simulation cam", cv2.resize(img, (500, 500)))  # 显示静态相机的图像
            if "rgb_gripper" in rgb_obs:
                img2 = rgb_obs["rgb_gripper"][:, :, ::-1]  # 将抓手相机图像转换为 BGR
                cv2.imshow("gripper cam", cv2.resize(img2, (500, 500)))  # 显示抓手相机的图像
            cv2.waitKey(1)  # 等待 1 毫秒
        elif mode == "rgb_array":  # 如果是 RGB 数组模式
            assert "rgb_static" in rgb_obs, "Environment does not have static camera"  # 确保有静态相机
            return rgb_obs["rgb_static"]  # 返回静态相机的 RGB 图像
        else:
            raise NotImplementedError  # 如果模式未实现，抛出异常

    def get_scene_info(self):
        return self.scene.get_info()  # 获取场景信息

    def reset(self, robot_obs=None, scene_obs=None):
        self.scene.reset(scene_obs)  # 重置场景
        self.robot.reset(robot_obs)  # 重置机器人
        self.p.stepSimulation(physicsClientId=self.cid)  # 进行一步仿真
        return self.get_obs()  # 返回环境观测

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)  # 使用 Gym 工具设置随机种子
        return [seed]  # 返回种子

    def get_camera_obs(self):
        assert self.cameras is not None  # 确保相机已初始化
        # 确保相机已初始化
        if not self.cameras:
            print("Error: No cameras initialized!")
            return {}, {}

        # 打印相机初始化情况
        #print(f"Initialized cameras: {[cam.name for cam in self.cameras]}")
        rgb_obs = {}  # 初始化 RGB 观测字典
        depth_obs = {}  # 初始化深度观测字典
        for cam in self.cameras:  # 对每个相机进行渲染
            rgb, depth = cam.render()  # 渲染获取 RGB 和深度图像
            #print(f">>>>>>>>>>>>>>>>>>>>Camera {cam.name} - RGB shape: {np.shape(rgb)}, Depth shape: {np.shape(depth)}")  # 打印调试信息
            rgb_obs[f"rgb_{cam.name}"] = rgb  # 存储 RGB 观测
            depth_obs[f"depth_{cam.name}"] = depth  # 存储深度观测
        return rgb_obs, depth_obs  # 返回 RGB 和深度观测

    def get_obs(self):
        """收集相机、机器人和场景的观测。"""
        rgb_obs, depth_obs = self.get_camera_obs()  # 获取相机观测
        obs = {"rgb_obs": rgb_obs, "depth_obs": depth_obs}  # 将相机观测存入观测字典
        obs.update(self.get_state_obs())  # 更新字典，加入状态观测
        return obs  # 返回综合观测

    def get_state_obs(self):
        """
        收集状态观测字典
        --state_obs
            --robot_obs
                --robot_state_full
                    -- [tcp_pos, tcp_orn, gripper_opening_width]
                --gripper_opening_width
                --arm_joint_states
                --gripper_action
            --scene_obs
        """
        robot_obs, robot_info = self.robot.get_observation()  # 获取机器人的观测和信息
        scene_obs = self.scene.get_obs()  # 获取场景的观测
        obs = {"robot_obs": robot_obs, "scene_obs": scene_obs}  # 组合为字典
        return obs  # 返回状态观测

    def get_info(self):
        _, robot_info = self.robot.get_observation()  # 获取机器人的信息
        info = {"robot_info": robot_info}  # 构建信息字典
        if self.use_scene_info:  # 如果使用场景信息
            info["scene_info"] = self.scene.get_info()  # 添加场景信息
        return info  # 返回信息

    def step(self, action):
        # 在 VR 模式下启用实时仿真，因此不需要手动调用 p.stepSimulation()
        if self.use_vr:
            log.debug(f"SIM FPS: {(1 / (time.time() - self.t)):.0f}")  # 记录仿真 FPS
            self.t = time.time()  # 更新时间戳
            current_time = time.time()  # 获取当前时间
            delta_t = current_time - self.prev_time  # 计算时间差
            if delta_t >= (1.0 / self.control_freq):  # 如果时间差大于控制频率的倒数
                log.debug(f"Act FPS: {1 / delta_t:.0f}")  # 记录动作 FPS
                self.prev_time = time.time()  # 更新前一帧时间戳
                self.robot.apply_action(action)  # 应用动作到机器人
            self.fps_controller.step()  # 控制 FPS
        # 对于强化学习，调用仿真步数重复
        else:
            self.robot.apply_action(action)  # 应用动作到机器人
            for i in range(self.action_repeat):  # 重复仿真多步
                self.p.stepSimulation(physicsClientId=self.cid)  # 进行一步仿真
        self.scene.step()  # 场景仿真一步
        obs = self.get_obs()  # 获取当前观测
        info = self.get_info()  # 获取当前信息
        # 返回观测、奖励、是否结束标志和信息
        return obs, 0, False, info

    def reset_from_storage(self, filename):
        """
        Args:
            filename: 要加载的文件名。
        Returns:
            observation
        """
        with open(filename, "rb") as file:  # 打开文件
            data = pickle.load(file)  # 加载数据

        self.robot.reset_from_storage(data["robot"])  # 从存储中重置机器人
        self.scene.reset_from_storage(data["scene"])  # 从存储中重置场景

        self.p.stepSimulation(physicsClientId=self.cid)  # 进行一步仿真

        return data["state_obs"], data["done"], data["info"]  # 返回状态观测、完成标志和信息

    def serialize(self):
        data = {
            "time": time.time_ns() / (10**9),  # 当前时间的纳秒数
            "robot": self.robot.serialize(),  # 序列化机器人
            "scene": self.scene.serialize()  # 序列化场景
        }
        return data  # 返回序列化的数据


# 获取环境
def get_env(dataset_path, obs_space=None, show_gui=True, **kwargs):
    from pathlib import Path  # 导入路径模块
    from omegaconf import OmegaConf  # 导入 OmegaConf 模块

    render_conf = OmegaConf.load(Path(dataset_path) / ".hydra" / "merged_config.yaml")  # 加载渲染配置

    if obs_space is not None:  # 如果指定了观测空间
        exclude_keys = set(render_conf.cameras.keys()) - {
            re.split("_", key)[1] for key in obs_space["rgb_obs"] + obs_space["depth_obs"]
        }  # 获取要排除的相机键
        for k in exclude_keys:  # 排除不需要的相机
            del render_conf.cameras[k]
    if "scene" in kwargs:  # 如果有场景参数
        scene_cfg = OmegaConf.load(Path(calvin_env.__file__).parents[1] / "conf/scene" / f"{kwargs['scene']}.yaml")  # 加载场景配置
        OmegaConf.merge(render_conf, scene_cfg)  # 合并渲染和场景配置
    if not hydra.core.global_hydra.GlobalHydra.instance().is_initialized():  # 检查 Hydra 是否已初始化
        hydra.initialize(".")  # 初始化 Hydra
    env = hydra.utils.instantiate(render_conf.env, show_gui=show_gui, use_vr=False, use_scene_info=True)  # 实例化环境
    return env  # 返回环境实例





class Simulator:

    def __init__(self,cfg):
        self.env= hydra.utils.instantiate(cfg.env, show_gui=True, use_vr=False, use_scene_info=True)  # 实例化环境
        print(f"[Simulator]Initiated the env")

    def reset(self,scene_obs=None,robot_obs=None):
        obs=self.env.reset(scene_obs=scene_obs,robot_obs=robot_obs)
        self.step_cnt = 0
        return obs

    def run_a_step(self,in_action=None, act_type="cartesian_rel"):
        self.step_cnt += 1  # 记录步数
        print(f"Run step {self.step_cnt}")

        # 如果 action 是一个 PyTorch Tensor，并且需要转换为 numpy
        if isinstance(in_action, torch.Tensor):
            in_action = in_action.detach().numpy()  # 使用 detach() 来断开梯度追踪

        # relative action in joint space
        action = {"action": in_action, "type": act_type}
        #action = {"action": in_action,"type": "cartesian_abs"}


        #action = {"action": np.array((0., 0, 0, 0, 0, 0, self.gripper_state)),
        # 设置动作数组，最后一个值控制抓手状态：-1 表示打开，1 表示关闭
        #reward=0, save the reward in default
        obs, _, _, info=self.env.step(action)#in default, it's cartesian position and orientation


        #print("Content of obs:", obs)  # 打印 obs 的内容
        #print("========Content of info:", info)
        # action = {"action": np.array((0., 0, 0, 0, 0, 0, 1)),
        #           "type": "cartesian_rel"}
        # cartesian actions can also be input directly as numpy arrays
        # action = np.array((0., 0, 0, 0, 0, 0, 1))

        # relative action in joint space
        # action = {"action": np.array((0., 0, 0, 0, 0, 0, 0, 1)),
        #           "type": "joint_rel"}

        # 更新角度以沿圆形轨迹移动
        time.sleep(0.05)  # 休眠 50 毫秒，确保动画更流畅,越小越流畅

        return obs, info  # 返回观测和信息

@hydra.main(config_path="../../conf", config_name="config_data_collection")  # 使用 Hydra 进行配置
def main(cfg):
    simulation = Simulator(cfg)  # 运行环境
    simulation.reset()
    #simulation.env.render()  # 如果需要渲染环境，取消注释

    simulation.run_a_step()


if __name__ == "__main__":
    main()
