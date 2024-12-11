import torch
import os
import cv2
from PIL import Image
import numpy as np
import h5py
from calvin.calvin_env.calvin_env.envs.play_table_env import Simulator
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils

class Low_level_planner:
    '''
    low-level planner, responsible for executing action sequences
    '''

    def __init__(self, cfg):
        '''
        state_dim: the dimension of the state vector 39
        action_dim: the dimension of the action vector 9
        cur_models_dir: the dir where all trained models that you currently want to use (BC, BC_RNN, etc.)
        '''
        self.max_step = 100  # 最大步数
        self.models = {}
        self.task_ids = []

        self.pre_green_light_state = None
        self.pre_pink_z_axis = None

        self.annotator = None
        self.video = None
        self.models_dir = "./Config/model_epoch_200.pth"  # 模型路径
        self.cur_demo_dir= "./Config/env_demo.hdf5"  # 演示数据路径
        self.action= None

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_models()
        self.simulator = self.get_simulator(cfg)

        self.previous_action = None
        self.obs = {  # 初始化用于存储观测数据的字典，结构保持与解析时一致
            'robot_obs': {
                'tcp_position': [],
                'tcp_orientation': [],
                'gripper_opening_width': 0,
                'arm_joint_states': [],
                'gripper_action': 0
            },
            'scene_obs': {
                'sliding_door_state': 0,
                'drawer_state': 0,
                'button_state': 0,
                'switch_state': 0,
                'lightbulb_state': 0,
                'green_light_state': 0,
                'red_block': {
                    'position': [],
                    'orientation': []
                },
                'blue_block': {
                    'position': [],
                    'orientation': []
                },
                'pink_block': {
                    'position': [],
                    'orientation': []
                }
            }
        }

    def Low_level_testing(self, Primitive_seq):
        '''
        receive the primitive sequence from high-level planner
        return state_sequence, action_sequence
        '''
        self.video = []  # 用于存储视频帧
        self.video_cnt = 0  # 用于保存视频的计数

        #state_sequence = []  # 存储状态序列
        #action_sequence = []  # 存储动作序列
        self.step = 0  # 初始化 step 计数器

        # 在这里读入两个obs，robot_obs=None, scene_obs=None
        step_cnt = 0

        _, _, robo_obs, scene_obs = self.test_hdf5(step_cnt)
        cur_obs = self.simulator.reset(scene_obs=scene_obs, robot_obs=robo_obs)

        self.parse_obs(cur_obs)
        while True:
            action = None

            """action: 7-dim vector"""
            cur_model = self.load_policy()  # load corresponding model by primitive_id (task_id)
            #state = self.state

            # 获取机器人观测输入数据
            input_scene_obs = self.obs['original_scene_obs']
            input_robot_obs = self.obs['original_robot_obs']
            tcp_position = self.obs['robot_obs']['tcp_position']  # 形状 (3,)
            tcp_orientation = self.obs['robot_obs']['tcp_orientation']  # 形状 (3,)
            gripper_opening_width = self.obs['robot_obs']['gripper_opening_width']  # 形状 (1,)
            button_state = self.obs['scene_obs']['button_state'] = scene_obs[2]  # 按钮状态
            green_light_state = self.obs['scene_obs']['green_light_state'] = scene_obs[5]


            #print("\n\n\n block pink\n\n\n", pink_block)
            #print(f"tcp_position: {tcp_position}, tcp_orientation: {tcp_orientation}, gripper_opening_width: {gripper_opening_width}")
            # print(f"tcp_position: {tcp_position}, tcp_orientation: {tcp_orientation}, gripper_opening_width: {gripper_opening_width}")

            # 计算 tcp 到物体的距离
            # @Zhixu，如果出问题，暂时注释掉

            # 创建输入字典，键名与模型输入键匹配
            input_data = {
                "robot0_eef_euler": torch.tensor(tcp_orientation, dtype=torch.float32).to(self.device),
                "robot0_eef_pos": torch.tensor(tcp_position, dtype=torch.float32).to(self.device),
                "robot0_gripper_qpos": torch.tensor([gripper_opening_width], dtype=torch.float32).to(
                    self.device),
            }

            # # 获取模型预测的动作和演示动作
            action = cur_model(input_data)  # 模型预测的动作
            demo_action, assemble_action, _, _ = self.test_hdf5(step_cnt)  # 演示动作

            # demo_action, assemble_action, _, _ = self.test_hdf5(step_cnt)  # 演示动作
            step_cnt += 1
            # MARK: To test demo
            #action=demo_action

            self.step += 1
            #print(f"------------The action input is {action}")
            action = self.gripper_action_binary(action)
            cur_obs, info = self.simulator.run_a_step(action)
            self.parse_obs(cur_obs)
            #self.state = self.get_state_from_obs(cur_obs)

            #state_sequence.append(self.state.copy())  # 将新状态加入状态序列
            #action_sequence.append(action.copy())  # 将动作加入动作序列

            if self.obs['scene_obs']['green_light_state'] == 1:
                return True

            if step_cnt >= self.max_step:
                return False






    def apply_action(self, action, act_type="cartesian_rel"):
        print(f"------------The action input is {action}")
        action = self.gripper_action_binary(action)
        cur_obs, info = self.simulator.run_a_step(action,act_type)
        self.action = action
        self.parse_obs(cur_obs)
        #self.state = self.get_state_from_obs(cur_obs)







    def get_tcp_obj_distance(self, tcp_position, obj_position):
        '''
        get the distance between tcp and object
        函数只提取前三维度，所以对obj_position形状没有要求
        '''
        tcp_position = np.array(tcp_position)
        obj_position = np.array(obj_position)
        distance = tcp_position[:3] - obj_position[:3]  # direct do the substraction
        return distance

    # def callback_save_print(self, state_sequence, action_sequence, primitive_type):
    #     '''
    #     save the state_sequence and action_sequence to files
    #     '''
    #     # 定义文件路径
    #     state_seq_file = 'Result/Video/state_seq.txt'
    #     action_seq_file = 'Result/Video/action_seq.txt'
    #
    #     # 将 state_sequence 写入文件
    #     with open(state_seq_file, 'w') as state_file:
    #         state_file.write(">>>>> state seq:\n")
    #         for state in state_sequence:
    #             state_file.write(f"{state}\n")
    #
    #     # 将 action_sequence 写入文件
    #     with open(action_seq_file, 'w') as action_file:
    #         action_file.write(">>>>> action seq:\n")
    #         for action in action_sequence:
    #             action_file.write(f"{action}\n")
    #
    #     print(f">>>>>>>>>>State sequence and action sequence saved to {state_seq_file} and {action_seq_file}.")
    #     self.frames2video(primitive_type)

    def test_hdf5(self, step_cnt):
        """
        Load episode data from HDF5 file and print scene_obs, robot_obs, and actions for a specific step.

        Args:
            hdf5_file_path (str): Path to the HDF5 file.
            step_cnt (int): Step number to extract data for.
        """
        # 初始化返回值，防止出错时未定义
        action_step = None
        assembled_action = None
        rst_robo_obs = None
        rst_scene_obs = None

        try:
            # 打开HDF5文件
            hdf5_file_path = self.cur_demo_dir

            with h5py.File(hdf5_file_path, 'r') as f:
                # 解析data组
                data_group = f['data']
                demo_name = list(data_group.keys())[0]  # 选择第一个demo，假设只有一个
                #print(f"~~~!!!~~~!!!~~~!!!~~~Demo name: {demo_name}")
                demo_group = data_group[demo_name]

                # 解析actions, robot_obs, scene_obs
                actions = demo_group['actions'][()]
                robot_obs = demo_group['obs/robot_obs'][()]
                scene_obs = demo_group['obs/scene_obs'][()]
                rst_robo_obs = robot_obs[step_cnt]
                rst_scene_obs = scene_obs[step_cnt]

                robot0_eef_euler = demo_group['obs/robot0_eef_euler'][()]
                robot0_eef_pos = demo_group['obs/robot0_eef_pos'][()]
                robot0_prev_gripper_action = demo_group['obs/robot0_prev_gripper_action'][()]

                # 直接读取 action 的特定步骤数据
                action_step = actions[step_cnt]

                # 重新排列顺序（如果需要修正原有顺序，例如位置和角度反了的情况）
                tcp_pos = action_step[:3]  # 假设前 3 个元素是位置
                tcp_orientation = action_step[3:6]  # 假设中间 3 个元素是角度
                gripper_action = action_step[6:]  # 最后一个元素是抓手动作
                assembled_action = np.concatenate((tcp_orientation, tcp_pos, gripper_action))


        except Exception as e:
            print(f"Error loading HDF5 file: {e}")

        # 保证即使出错，也返回初始化的值
        return action_step, assembled_action, rst_robo_obs, rst_scene_obs

    def parse_obs(self, obs):
        """
        解析当前的观测状态并直接更新 self.obs。

        参数:
        - obs: 字典，包含当前的观测数据，包括 'robot_obs' 和 'scene_obs'。
        """

        #print("//================= Parsing observation =================//")

        # 初始化 self.obs 如果没有被初始化
        if not hasattr(self, 'obs'):
            self.obs = {'robot_obs': {}, 'scene_obs': {}, 'rgb_obs': {}, 'rgb_depth': {}}


        # 解析 robot_obs
        robot_obs = obs.get('robot_obs', [])
        if robot_obs.size > 0:
            self.obs['robot_obs']['tcp_position'] = robot_obs[:3].tolist()  # TCP 位置 (x, y, z)
            self.obs['robot_obs']['tcp_orientation'] = robot_obs[3:6].tolist()  # TCP 方向（欧拉角 x, y, z）
            #print(f"/////tcp position: {self.obs['robot_obs']['tcp_position']} and tcp orientation: {self.obs['robot_obs']['tcp_orientation']}////\n")
            self.obs['robot_obs']['gripper_opening_width'] = robot_obs[6]  # 抓手开口宽度 (米)
            self.obs['robot_obs']['arm_joint_states'] = robot_obs[7:14].tolist()  # 机械臂关节状态 (7 个关节角度)
            self.obs['robot_obs']['gripper_action'] = robot_obs[14]  # 抓手动作 (-1 表示关闭，1 表示打开)
            self.obs['original_robot_obs'] = robot_obs  # 保存原始 robot_obs 数据

        # 解析 scene_obs
        scene_obs = obs.get('scene_obs', [])
        if scene_obs.size == 24:
            self.obs['scene_obs']['sliding_door_state'] = scene_obs[0]  # 滑动门状态
            self.obs['scene_obs']['drawer_state'] = scene_obs[1]  # 抽屉状态
            self.obs['scene_obs']['button_state'] = scene_obs[2]  # 按钮状态
            self.obs['scene_obs']['switch_state'] = scene_obs[3]  # 开关状态
            self.obs['scene_obs']['lightbulb_state'] = scene_obs[4]  # 灯泡状态（开=1，关=0）
            self.obs['scene_obs']['green_light_state'] = scene_obs[5]  # 绿灯状态（开=1，关=0）
            if self.step == 0:
                self.pre_green_light_state =self.obs['scene_obs']['green_light_state']



            self.obs['original_scene_obs'] = scene_obs  # 保存原始 scene_obs 数据
            # 解析三个颜色的方块的位置和方向
            self.obs['scene_obs']['red_block'] = scene_obs[6:12].tolist()
            self.obs['scene_obs']['blue_block'] = scene_obs[12:18].tolist()
            self.obs['scene_obs']['pink_block'] = scene_obs[18:24].tolist()
            if self.step == 0:
                self.pre_pink_z_axis = self.obs['scene_obs']['pink_block'][2]

        # 确保保存路径存在
        save_path = "Result/Image"
        os.makedirs(save_path, exist_ok=True)

        # 解析 rgb_obs
        rgb_obs = obs.get('rgb_obs', {})
        if isinstance(rgb_obs, dict):
            # 解析 rgb_static
            rgb_static = rgb_obs.get('rgb_static', None)
            if rgb_static is not None:
                # 将 rgb_static 数据转换为 PIL.Image 对象
                image_static = Image.fromarray(np.uint8(rgb_static)).convert("RGB")
                self.obs['rgb_static'] = {
                    'shape': image_static.size,
                    'image': image_static  # 直接保存为 PIL.Image 对象
                }

                if self.PRINT:
                    # 保存 rgb_static 图像为文件，包含 step 计数
                    static_filename = f'{save_path}/rgb_static_{self.step}.png'
                    image_static.save(static_filename)
                    #print(f"Saved rgb_static image as '{static_filename}'.")

                # 将 PIL 图像转换为 OpenCV 格式并添加到视频帧列表中
                self.video.append(cv2.cvtColor(np.array(image_static), cv2.COLOR_RGB2BGR))

            # 解析 rgb_gripper
            rgb_gripper = rgb_obs.get('rgb_gripper', None)
            if rgb_gripper is not None:
                # 将 rgb_gripper 数据转换为 PIL.Image 对象
                image_gripper = Image.fromarray(np.uint8(rgb_gripper)).convert("RGB")
                self.obs['rgb_gripper'] = {
                    'shape': image_gripper.size,
                    'image': image_gripper  # 直接保存为 PIL.Image 对象
                }

                if self.PRINT:
                    # 保存 rgb_gripper 图像为文件，包含 step 计数
                    gripper_filename = f'{save_path}/rgb_gripper_{self.step}.png'
                    image_gripper.save(gripper_filename)
                    #print(f"Saved rgb_gripper image as '{gripper_filename}'.")

        # 打印更新的观测信息
        # print("Updated self.obs:")
        # for key, value in self.obs.items():
        #     print(f"\nParsed {key}:")
        #     if isinstance(value, dict):
        #         for sub_key, sub_value in value.items():
        #             print(f"  {sub_key}: {sub_value}")
        print("//===================================//")
        # Call function to estimate object positions




    def frames2video(self, pri_type):
        """
        将保存的图像帧合成为一个视频并保存到 Result/Video/ 文件夹中。
        """
        self.video_cnt += 1
        save_video_path = "Result/Video"
        os.makedirs(save_video_path, exist_ok=True)

        # 设置视频参数
        video_filename = f'{save_video_path}/rgb_static_video_{self.video_cnt}_{pri_type}.mp4'
        if self.video:
            height, width, _ = self.video[0].shape
            video_writer = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*'mp4v'), 10, (width, height))

            for frame in self.video:
                video_writer.write(frame)

            video_writer.release()
            print(f"Saved video as '{video_filename}'.")
        else:
            print("No frames to create video.")


    def gripper_action_binary(self, action):
        # TODO @Zhixu，这里的 action 是 7 维的，最后一维不是 gripper 的状态，这个函数需要改。我先继续往下做code review

        """
        将动作的第七位（抓手状态）严格映射为 -1 或 1。

        参数:
        action -- 7维数组，包含 TCP 位置、方向和抓手状态。如果是 PyTorch Tensor，先转换为 numpy 数组。

        返回:
        经过处理的 7维动作数组
        """
        # 如果 action 是 PyTorch Tensor，先使用 detach() 断开梯度，再转换为 numpy
        if isinstance(action, torch.Tensor):
            action = action.detach().numpy()

        if len(action) != 7:
            raise ValueError("Action must be一个7-dimensional array.")

        # 提取原始抓手状态
        gripper_value = action[6]

        # 将抓手状态映射为 -1 或 1
        gripper_binary = 1 if gripper_value >= 0 else -1

        # 更新 action 的第七位
        action[6] = gripper_binary

        return action


    def gripper_action_binary_new(self, action):
        pass


    # def get_state_from_obs(self, obs):
    #     ''' contract state from obs '''
    #     robot_obs = obs['robot_obs']
    #     scene_obs = obs['scene_obs']
    #     state = np.concatenate((robot_obs, scene_obs), axis=0)
    #     return state


    def get_simulator(self, cfg):
        '''get simulator'''
        # Zhixu: 已设置为类内
        # 创建仿真器
        simulator = Simulator(cfg)
        return simulator


    def load_policy(self):
        model = self.model
        #model.eval()  # Set the model to evaluation mode
        return model

    def load_models(self):
        '''Load all trained models and recognize task_ids'''
        model_path = self.models_dir
        print(f"..........................Loading ROBOMIMIC model from {model_path}..........................")
        try:
            device = TorchUtils.get_torch_device(try_to_use_cuda=True)
            # restore policy
            policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=model_path, device=device, verbose=True)
            self.model = policy
        except Exception as e:
            print(f"Error loading ROBOMIMIC model from {model_path}")
            print(f"Error details: {str(e)}")
            raise RuntimeError(f"Failed to load ROBOMIMIC model from {model_path}") from e

        return self.model


# test script
if __name__ == '__main__':
    pass
