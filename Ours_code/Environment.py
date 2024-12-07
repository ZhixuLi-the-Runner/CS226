from Ours_code.Testing import Low_level_planner

class Environment:
    def __init__(self,cfg=None):
        self.cfg=cfg
        self.lplanner = Low_level_planner(cfg=self.cfg)
        return



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
