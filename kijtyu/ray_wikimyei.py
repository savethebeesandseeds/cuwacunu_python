# --- ---- --- 
# ray_wikimyei
# --- ---- --- 
import os
import ast
import sys
import torch
import logging
import re
import copy
# --- ---- --- 
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from functools import partial
# --- --- --- ---
# CWCN_KIJTIYU_FOLDER="../kijtyu"
# os.environ['CWCN_KIJTYU_FOLDER']=os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(__file__)),CWCN_KIJTIYU_FOLDER))
# sys.path.append(os.environ['CWCN_KIJTYU_FOLDER'])
# --- ---- --- 
import cwcn_config
import cwcn_wikimyei_nebajke
import cwcn_wikimyei_piaabo
import cwcn_jkimyei_nebajke
import cwcn_tsinuu_piaabo
import cwcn_kemu_piaabo
# --- ---- --- 
os.environ['CWCN_CONFIG_SYSTEM']='ray_system' # redundant
# --- ---- --- 
RAY_CURRENT_CONFIG=copy.deepcopy(cwcn_config.CWCN_CONFIG())
# --- ---- --- 
class RAY_ORDER_JKIMYEI: # use pytorch/ray to optimize hyperparameters
    def __init__(self):
        # --- --- --- 
        logging.ray_logging("--- RAY system is initialized ----")
        os.environ.CWCN_CONFIG_SYSTEM='ray_system' #FIXME environ is leakaged
        self.checkpoint_file=os.path.join(cwcn_config.CWCN_CONFIG().RAY_CHECKPOINTS_FOLDER,"checkpoint")
        self.__rjk_config=cwcn_config.CWCN_CONFIG().__dict__ #FIXME ray config is not dinamic (sure, and?)
        self.ray_wikimyei=None
        # --- --- --- 
    def _report_(self,_reward):
        # --- --- --- 
        tune.report(reward=_reward)
        # --- --- --- 
    def _ray_iteration_(self,config):
        # --- --- --- 
        
        #... #FIXME

        logging.ray_logging("--- ray step ---")
        self.ray_wikimyei=cwcn_wikimyei_nebajke.WIKIMYEI(config)
        self.ray_wikimyei.jkimyei._jkimyei_wikimyei_()
        c_reward=self.ray_wikimyei._test_wikimyei_on_ahpa_(render_flag=False)
        self.ray_wikimyei._save_wikimyei_(self.checkpoint_file)
        self._report_(c_reward) # tune.report(imibajcho=(val_imibajcho / val_steps), accuracy=correct / total)
        logging.ray_logging("--- [REPORT_LEVEL]_:reward:_{}_:config:_{}_:end:_".format(c_reward,self.ray_wikimyei.wk_config))
        logging.ray_logging("--- ray step ended ---")
        # --- --- --- 
    def _export_trail_(self,result):
        # --- --- --- 
        # best_trial = result.get_best_trial("reward", "max", "last")
        # print("Best trial config: {}".format(best_trial.config))
        # print("Best trial final validation reward: {}".format(best_trial.last_result["reward"]))
        # print("Best trial test set reward: {}".format(c_reward))
        # logging.ray_logging("[RESULTS:] \n{} ".format(list(result.results_df.columns.values)))
        # --- --- --- 
        # aux_list=result.results_df['reward'].apply(lambda x:x.item())
        # logging.ray_logging("[RESULTS:] \n{} ".format(result.results_df))
        # logging.ray_logging("[BEST  RESULT:] \n{} ".format(result.results_df.iloc[aux_list.tolist().index(max(aux_list))]))
        # --- --- --- 
        result.results_df.to_csv("ray_result.csv") #FixME  
        # --- --- --- 
        self._read_ray_logs_()
        # --- --- --- 
    def _ray_main_(self):
        # --- --- --- 
        logging.ray_logging("--- ray main ---")
        ray.init()
        scheduler = ASHAScheduler(
            metric="reward",
            mode="max",
            max_t=0xFFFFFFFF,
            grace_period=1,
            reduction_factor=2)
        reporter = CLIReporter(# parameter_columns=["config in general", "l2", "lr", "batch_size"],
            metric_columns=["reward", "training_iteration"],
            parameter_columns=["reward", "training_iteration"],
            sort_by_metric=True)
        assert(not torch.cuda.is_available()), "add when cuda is aviable (maybe the only cuda bug, fix next two fixmes and go on)" #FIXME 
        result = tune.run(
            self._ray_iteration_,
            resources_per_trial={"cpu": os.cpu_count(), "gpu": 0}, #FIXME add when cuda is aviable
            config=self.__rjk_config,
            num_samples=self.__rjk_config['RAY_N_TRAILS'],
            scheduler=scheduler,
            progress_reporter=reporter,
            raise_on_failed_trial=False)
        # --- --- --- 
        if(cwcn_config.CWCN_CONFIG.export_trail):
            self._export_trail_(result)
        # --- --- --- 
        # self.ray_wikimyei=cwcn_wikimyei_nebajke.WIKIMYEI(self.checkpoint_file)
        # c_reward=self.ray_wikimyei._test_wikimyei_on_ahpa_(render_flag=True)
        # --- --- --- 
        if(cwcn_config.CWCN_CONFIG.close_at_finish):
            ray.shutdown()
        logging.ray_logging("--- ray main ended ---")
        # --- --- --- 
        return result

    def _read_ray_logs_(self,_n_winrs): # read the best n
        _logs_info=[]
        current_log_file=logging.getLoggerClass().root.handlers[1].baseFilename
        log_folder=os.path.split(current_log_file)[0]
        for _pth in os.listdir(log_folder):
            some_file = os.path.join(log_folder,_pth)
            with open(some_file) as _f:
                c_log_content=_f.readlines()
            # c_log_content=c_log_content.split("\n")
            c_log_content=list(filter(lambda x: "[RAY ]" in x,c_log_content))
            c_log_content=list(filter(lambda x: "[REPORT_LEVEL]" in x,c_log_content))
            for _c in c_log_content:
                # print("--- -- --- -- -- - \n\n{}\n\n--- -- - -- - - ".format(_c))
                aux_info={}
                aux_info["reward"]=float(re.findall(r"(?<=_:reward:_)(.*)(?=_:config:_)",_c)[0])
                aux_info["config"]=ast.literal_eval(re.findall(r"(?<=_:config:_)(.*)(?=_:end:_)",_c)[0]) # interesting function
                _logs_info.append(aux_info)
        _logs_info=sorted(_logs_info,key=(lambda x: x['reward']), reverse=True)
        logging.ray_logging("[RAY RESULTS REDED FROM LOG,] best trails :")
        if(cwcn_config.CWCN_CONFIG().ray_prtnt_flag):
            def _prtnt_lgs_(_lgs, _n_winrs):
                for _idx in range(_n_winrs):
                    logging.ray_logging(" --- --- N°{} --- ---".format(_idx+1))
                    cwcn_kemu_piaabo.kemu_pretty_print_object(_lgs[_idx])
            _prtnt_lgs_(_logs_info,_n_winrs)
        # logging.ray_logging(" --- --- N°{} --- ---".format(4))
        # cwcn_kemu_piaabo.kemu_pretty_print_object(_logs_info[3])
        # logging.ray_logging(" --- --- N°{} --- ---".format(5))
        # cwcn_kemu_piaabo.kemu_pretty_print_object(_logs_info[4])
        return _logs_info[:_n_winrs]
# class RAY_OPTIMIZATION_PATTERN:
#     def __init___(self):
#         pass
#     def _get_best_trial_(self):
        
#     def _permute_n_survivals_in_p_rounds_(self,_n_count,_p_count):
#         _c_count=0
#         while(True):
#             _c_count+=1
#             self._ray_main_()[:_n_count]
#             modify config
#             if(_p_count<_c_count):
#                 break
if __name__ == "__main__":

    c_ray_jkimyei=RAY_ORDER_JKIMYEI()
    c_ray_jkimyei._ray_main_()
    c_ray_jkimyei._read_ray_logs_(3)
    