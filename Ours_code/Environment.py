import copy
import numpy as np
import os
import re
import ast
import pickle
from Ours_code.High_level_planner import High_level_planner
from Ours_code.Testing import Low_level_planner
from Ours_code.Data_Processing import get_tasks



class Environment:
    def __init__(self,task_input_path, obj_input_path,output_path,primitive_path ,cur_model_dir, model,cur_demo_dir=None,PRINT = True, mode="training",cfg=None, scenario="Real_World", high_level=False):
        self.environment=None
        self.action_space=None
        self._mode=mode

        self.current_task=None
        self.current_state=None
        self.test_h=high_level
        self.obj_input_path=obj_input_path
        self.task_input_path = task_input_path

        self.output_path=output_path
        self.model=model
        self.PRINT=PRINT
        self.cfg=cfg
        self.hplanner = High_level_planner(check_semantic=True, check_syntactic=True,primitive_path=primitive_path,output_path=self.output_path,
                                           model_name=self.model, PRINT=self.PRINT,scenario=scenario)
        self.lplanner = Low_level_planner(state_dim=39, action_dim=7,cfg=self.cfg,cur_models_dir=cur_model_dir,cur_demo_dir=cur_demo_dir ,VLM=False, ROBOMIMIC=True)

        self.tasks = get_tasks(self.task_input_path)
        return



    #TODO For testing
    def save_primitive_seqs(self, primitive_seqs, file_path):
        """保存 Primitive_seqs 到文件"""
        with open(file_path, 'wb') as file:
            pickle.dump(primitive_seqs, file)

    #TODO For testing
    def load_primitive_seqs(self, file_path):
        """从文件加载 Primitive_seqs"""
        with open(file_path, 'rb') as file:
            return pickle.load(file)

    """
    Run a single step
    """
    def run_a_step(self):
        return

    """
    Run both level planning to to execute the task
    """
    def run(self):
        result_log='Result/Logs.txt'
        Is_success = self.lplanner.Low_level_testing(None)

        with open(result_log, 'a') as log_file:
            log_file.write(f"|{Is_success}| for turn_on_led \n")


        # if self.test_h == True:
        #     Primitive_seqs = self.hplanner.all_task_high_level_planning(self.tasks, self.obj_input_path)
        #     self.save_primitive_seqs(Primitive_seqs, save_path)
        # else:
        #     if os.path.exists(save_path):
        #         Primitive_seqs = self.load_primitive_seqs(save_path)
        #         if self.PRINT:
        #             print("Loaded Primitive_seqs from file.")
        #     else:
        #         exit("No primitive sequences pkl file found")
        #
        #
        #     print(f"=====================Primitive sequences generated======================")
        #
        #     """Here the Primitive_seqs[task_index] is a primitive sequences"""
        #     for task_index, Primitive_per_task in enumerate(Primitive_seqs):
        #         if self._mode=='testing':
        #             self.lplanner.Low_level_testing(Primitive_per_task)
        #         else:
        #             raise ValueError("Invalid mode type")
        
        return


