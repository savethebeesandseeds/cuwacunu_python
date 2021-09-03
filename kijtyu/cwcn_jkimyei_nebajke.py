# --- --- --- --- 
import torch
import numpy as np
from torch.cuda import is_available
import torch.optim as optim
# --- --- --- ---
import re
from matplotlib import pyplot as plt
# --- --- --- ---
import os
import ast
import logging
# --- --- --- ---
import cwcn_config
import cwcn_wikimyei_nebajke
import cwcn_wikimyei_piaabo
import cwcn_tsinuu_piaabo
import cwcn_kemu_piaabo
# --- --- --- --- 
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from functools import partial
# --- --- --- --- 
class LEARNING_PROFILE:
    def __init__(self):
        # self.p_trayectory= None # profile reference to original trayectory
        self.ratio              = None
        self.surr1              = None
        self.surr2              = None
        self.uwaabo_imibajcho   = None
        self.munaajpi_imibajcho = None
        self.imibajcho          = None
        self.index              = None
        self.batch_size         = None
# --- --- --- ---
# --- --- --- ---
class RAY_ORDER_JKIMYEI: # use pytorch/ray to optimize hyperparameters
    def __init__(self):
        # --- --- --- 
        logging.ray_log("--- RAY system is initialized ----")
        os.environ.CWCN_CONFIG_SYSTEM='ray_system' #FIXME environ is leakaged
        self.checkpoint_file=os.path.join(cwcn_config.CWCN_CONFIG().RAY_CHECKPOINTS_FOLDER,"checkpoint")
        self.__rjk_config=cwcn_config.CWCN_CONFIG().__dict__ #FIXME ray config is not dinamic (sure, and?)
        self.ray_wikimyei=None
        # --- --- --- 
    def _report_(self,_imu):
        # --- --- --- 
        tune.report(imu=_imu)
        # --- --- --- 
    def _ray_iteration_(self,config):
        # --- --- --- 
        logging.ray_log("--- ray step ---")
        self.ray_wikimyei=cwcn_wikimyei_nebajke.WIKIMYEI(config)
        self.ray_wikimyei.jkimyei._wikimyei_jkimyei_()
        c_imu=self.ray_wikimyei._test_wikimyei_on_ahpa_(render_flag=False)
        self.ray_wikimyei._save_wikimyei_(self.checkpoint_file)
        self._report_(c_imu) # tune.report(imibajcho=(val_imibajcho / val_steps), accuracy=correct / total)
        logging.ray_log("--- [REPORT_LEVEL]_:imu:_{}_:config:_{}_:end:_".format(c_imu,self.ray_wikimyei.wk_config))
        logging.ray_log("--- ray step ended ---")
        # --- --- --- 
    def _export_best_trail_(self,result):
        # --- --- --- 
        # best_trial = result.get_best_trial("imu", "max", "last")
        # print("Best trial config: {}".format(best_trial.config))
        # print("Best trial final validation imu: {}".format(best_trial.last_result["imu"]))
        # print("Best trial test set imu: {}".format(c_imu))
        # logging.ray_log("[RESULTS:] \n{} ".format(list(result.results_df.columns.values)))
        # --- --- --- 
        # aux_list=result.results_df['imu'].apply(lambda x:x.item())
        # logging.ray_log("[RESULTS:] \n{} ".format(result.results_df))
        # logging.ray_log("[BEST  RESULT:] \n{} ".format(result.results_df.iloc[aux_list.tolist().index(max(aux_list))]))
        # --- --- --- 
        result.results_df.to_csv("ray_result.csv")
        # --- --- --- 
        self._read_ray_logs_()
        # --- --- --- 
    def _ray_main_(self,):
        # --- --- --- 
        logging.ray_log("--- ray main ---")
        ray.init()
        scheduler = ASHAScheduler(
            metric="imu",
            mode="max",
            max_t=0xFFFFFFFF,
            grace_period=1,
            reduction_factor=2)
        reporter = CLIReporter(# parameter_columns=["config in general", "l2", "lr", "batch_size"],
            metric_columns=["imu", "training_iteration"],
            parameter_columns=["imu", "training_iteration"],
            sort_by_metric=True)
        assert(not torch.cuda.is_available()), "add when cuda is aviable (maybe the only cuda bug, fix next two fixmes and go on)" #FIXME 
        result = tune.run(
            self._ray_iteration_,
            resources_per_trial={"cpu": os.cpu_count(), "gpu": 0}, #FIXME add when cuda is aviable
            config=self.__rjk_config,
            num_samples=self.__rjk_config.RAY_N_TRAILS,
            scheduler=scheduler,
            progress_reporter=reporter,
            raise_on_failed_trial=False)
        # --- --- --- 
        self._export_best_trail_(result)
        # --- --- --- 
        # self.ray_wikimyei=cwcn_wikimyei_nebajke.WIKIMYEI(self.checkpoint_file)
        # c_imu=self.ray_wikimyei._test_wikimyei_on_ahpa_(render_flag=True)
        # --- --- --- 
        logging.ray_log("--- ray main ended ---")
        ray.shutdown()
        # --- --- --- 
    def _read_ray_logs_(self):
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
                aux_info={}
                aux_info["imu"]=float(re.findall(r"(?<=_:imu:_)(.*)(?=_:config:_)",_c)[0])
                aux_info["config"]=ast.literal_eval(re.findall(r"(?<=_:config:_)(.*)(?=_:end:_)",_c)[0]) # interesting function
                _logs_info.append(aux_info)
        _logs_info=sorted(_logs_info,key=(lambda x: x['imu']), reverse=True)
        logging.ray_log("[RAY RESULTS REDED FROM LOG,] best trails :")
        logging.ray_log(" --- --- N°{} --- ---".format(1))
        cwcn_kemu_piaabo.kemu_pretty_print_object(_logs_info[0])
        logging.ray_log(" --- --- N°{} --- ---".format(2))
        cwcn_kemu_piaabo.kemu_pretty_print_object(_logs_info[1])
        logging.ray_log(" --- --- N°{} --- ---".format(3))
        cwcn_kemu_piaabo.kemu_pretty_print_object(_logs_info[2])
        logging.ray_log(" --- --- N°{} --- ---".format(4))
        cwcn_kemu_piaabo.kemu_pretty_print_object(_logs_info[3])
        logging.ray_log(" --- --- N°{} --- ---".format(5))
        cwcn_kemu_piaabo.kemu_pretty_print_object(_logs_info[4])
        return _logs_info

# --- --- --- ---
# Based on https://github.com/higgsfield/RL-Adventure-2/blob/master/3.ppo.ipynb
# Based on https://github.com/colinskow/move37/tree/master/ppo
class JKIMYEI_PPO:
    def __init__(self,_wikimyei):
        self.load_queue=None
        self.mini_batch_size=_wikimyei.wk_config['MINI_BATCH_COUNT']
        self.optimizer=optim.Adam(_wikimyei.model.parameters(), lr=_wikimyei.wk_config['TEHDUJCO_LEARNING_RATE'])
        self.munaajpi_imibajcho_fun=torch.nn.MSELoss()
        self.learning_queue=cwcn_wikimyei_piaabo.LOAD_QUEUE()
        # self.hist_learning_queue=cwcn_wikimyei_piaabo.LOAD_QUEUE()
        self.load_queue=cwcn_wikimyei_piaabo.LOAD_QUEUE()
        self.jk_wikimyei=_wikimyei
    def _jkmimyei_gae_(self):
        assert(self.load_queue is not None), "Impossible to compute GAE, Jkimyei Queue found to be None"
        gamma=self.jk_wikimyei.wk_config['TEHDUJCO_GAMMA']
        lam=self.jk_wikimyei.wk_config['TEHDUJCO_GAE_LAMBDA']
        _, next_value = self.jk_wikimyei.model(self.jk_wikimyei.wk_state.c_alliu)
        c_load_dict = self.load_queue._dict_vectorize_queue_()
        c_load_dict['value'].append(next_value)
        gae = 0
        returns = []
        gae_hist = []
        advantage = []
        delta_hist = []
        for step in reversed(range(self.load_queue.load_size)):
            delta = c_load_dict['imu'][step] + gamma * c_load_dict['value'][step + 1] * c_load_dict['mask'][step] - c_load_dict['value'][step]
            gae = delta + gamma * lam * c_load_dict['mask'][step] * gae
            delta_hist.insert(0,delta)
            gae_hist.insert(0,gae)                                  # prepend to get correct order back
            returns.insert(0, gae + c_load_dict['value'][step])    # prepend to get correct order back
            advantage.insert(0,gae + c_load_dict['value'][step] - c_load_dict['value'][step])   # prepend to get correct order back
            # print("waka: ",c_load_dict['index'][step],gae + c_load_dict['value'][step] - c_load_dict['value'][step])
        advantage = cwcn_kemu_piaabo.kemu_normalize(torch.cat(advantage))
        returns = cwcn_kemu_piaabo.kemu_normalize(torch.cat(returns))
        self.load_queue._load_vect_to_queue_('gae',gae_hist)
        self.load_queue._load_vect_to_queue_('delta',delta_hist)
        self.load_queue._load_vect_to_queue_('returns',returns)
        self.load_queue._load_vect_to_queue_('advantage',advantage)
    def _random_queue_yield_(self):
        for _ in range(self.load_queue.load_size // self.mini_batch_size):
            rand_ids = np.random.randint(0, self.load_queue.load_size, self.mini_batch_size)
            yield self.load_queue._dict_vectorize_queue_list_batch_([self.load_queue.load_queue[_i] for _i in rand_ids],True)
    def _jkimyei_ppo_update_(self):
        # PPO EPOCHS is the number of times we will go through ALL the training data to make updates
        clip_param=self.jk_wikimyei.wk_config['ReferencesToNoMeButWhoThoseAllWhoMadeRechableTheImplementationOfThisAlgorithm_TEHDUJCO_EPSILON']
        count_steps=0
        for _ in range(self.jk_wikimyei.wk_config['TRAINING_EPOCHS']):
            # grabs random mini-batches several times until we have covered all data
            for _trayectory in self._random_queue_yield_():
                jk_profile=LEARNING_PROFILE()
                jk_profile.batch_size=len(_trayectory['alliu'])
                # jk_profile.p_trayectory=_trayectory #FIXME, not in use, not appended correctly
                # --- ---
                # _trayectory['alliu'].requires_graph=True
                # _trayectory['action'].requires_graph=True
                # _trayectory['log_prob'].requires_graph=True
                # print("STATE FORM:",_trayectory['alliu'])
                alliu=torch.detach(_trayectory['alliu'])
                old_log_probs=torch.detach(_trayectory['log_prob'])
                advantage=torch.detach(_trayectory['advantage'])
                returns=torch.detach(_trayectory['returns'])
                # print("[advantage!:] {}".format(advantage))
                # print("[index!:] {}".format([_trayectory['index']]))
                # --- --- 
                alliu.requires_grad=True
                old_log_probs.requires_grad=True
                advantage.requires_grad=True
                returns.requires_grad=True
                # --- --- 
                dist, value=self.jk_wikimyei.model(alliu)
                # print("[size of sate:] {}, [dist:] {}".format(alliu.shape, dist))
                entropy=dist.entropy().mean()
                _, __, new_log_probs,___=self.jk_wikimyei._dist_to_tsane_(dist)
                # --- ---
                jk_profile.ratio=(new_log_probs - old_log_probs).exp()
                jk_profile.surr1=jk_profile.ratio * advantage
                jk_profile.surr2=torch.clamp(jk_profile.ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage
                # --- ---
                # IITEPI_BETA... #FIXME add iitepi as the exploration network
                jk_profile.uwaabo_imibajcho =- self.jk_wikimyei.wk_config['UWAABO_BETA'] * torch.min(jk_profile.surr1, jk_profile.surr2).mean() #FIXME mean is tricky, util only due to pytorch great tools
                # jk_profile.munaajpi_imibajcho=self.jk_wikimyei.wk_config['MUNAAJPI_BETA'] * (returns - value).pow(2).mean() #FIXME mean is tricky to implement in c, using a sign multiplication
                jk_profile.munaajpi_imibajcho=self.jk_wikimyei.wk_config['MUNAAJPI_BETA'] * self.munaajpi_imibajcho_fun(returns, value)
                jk_profile.uwaabo_imibajcho=torch.clamp(jk_profile.uwaabo_imibajcho,min=self.jk_wikimyei.wk_config['IMIBAJCHO_MIN'],max=self.jk_wikimyei.wk_config['IMIBAJCHO_MAX'])
                jk_profile.munaajpi_imibajcho=torch.clamp(jk_profile.munaajpi_imibajcho,min=self.jk_wikimyei.wk_config['IMIBAJCHO_MIN'],max=self.jk_wikimyei.wk_config['IMIBAJCHO_MAX'])
                jk_profile.imibajcho= jk_profile.munaajpi_imibajcho + jk_profile.uwaabo_imibajcho - self.jk_wikimyei.wk_config['TEHDUJCO_ENTROPY_BETA'] * entropy
                # logging.info("uwaabo_imibajcho: {}, \t munaajpi_imibajcho: {}, \t imibajcho: {}".format(jk_profile.uwaabo_imibajcho.size(),jk_profile.munaajpi_imibajcho.size(),jk_profile.imibajcho.size()))
                # logging.info("uwaabo_imibajcho: {:.4f}, \t munaajpi_imibajcho: {:.4f}, \t imibajcho: {:.4}".format(jk_profile.uwaabo_imibajcho,jk_profile.munaajpi_imibajcho, jk_profile.imibajcho))
                # if(abs(jk_profile.uwaabo_imibajcho)>=min(abs(self.jk_wikimyei.wk_config['IMIBAJCHO_MAX']),abs(self.jk_wikimyei.wk_config['IMIBAJCHO_MIN'])) or abs(jk_profile.munaajpi_imibajcho)>=min(abs(self.jk_wikimyei.wk_config['IMIBAJCHO_MAX']),abs(self.jk_wikimyei.wk_config['IMIBAJCHO_MIN']))):
                #     logging.info("[jk_profile] : {}".format([(_k,jk_profile.__dict__[_k]) for j_k in jk_profile.__dict__.keys()]))
                #     logging.info("[jk_profile] : {}".format([(_k,jk_profile.__dict__[_k].shape) for _k in jk_profile.__dict__.keys() if _k not in ['p_trayectory','index','batch_size']]))
                #     input("STOP...")
                # --- ---
                self.optimizer.zero_grad()
                jk_profile.imibajcho.backward()
                self.optimizer.step()
                # --- ---
                self.learning_queue._append_(jk_profile)
                # --- ---
                count_steps+=1
    def _wikimyei_jkimyei_(self):
        # logging.info(" + + + [New jkimyei iteration]")
        self.learning_queue._reset_queue_()
        self.load_queue._reset_queue_()
        self.jk_wikimyei._reset_()
        for _ in range(self.jk_wikimyei.wk_config['TRAINING_STEPS']):
            # --- ---
            __wt=cwcn_wikimyei_piaabo.TRAYECTORY()
            done=self.jk_wikimyei._wk_step_(__wt)
            # --- ---
            self.load_queue._append_(__wt)
            if(done):
                self.mini_batch_size= \
                    self.load_queue.load_size \
                        if(self.load_queue.load_size<self.jk_wikimyei.wk_config['MINI_BATCH_COUNT']) \
                            else self.jk_wikimyei.wk_config['MINI_BATCH_COUNT']
                break
        self.load_queue._load_normalize_(['imu'],'tensor') # not in use due to duuruva
        # if(>self.best_imu):
        # self.hist_learning_queue._import_queue_(self.learning_queue)
        self._jkmimyei_gae_()
        self._jkimyei_ppo_update_()
        # --- --- 
    def _standalone_wikimyei_jkimyei_ppo_loop_(self):
        assert(self.jk_wikimyei.wk_config['TRAINING_STEPS']>=self.jk_wikimyei.wk_config['MINI_BATCH_COUNT'])
        train_epoch = 0
        self.best_imu = None
        self.early_stop = False
        # self.hist_learning_queue._reset_queue_()
        while not self.early_stop:
            train_epoch += 1
            # --- --- --- TRAIN
            self._wikimyei_jkimyei_()
            # --- --- --- Eval
            if train_epoch % self.jk_wikimyei.wk_config['VALIDATION_EPOCH'] == 0:
                test_imu = np.mean([self.jk_wikimyei._test_wikimyei_on_ahpa_(render_flag=False) for _ in range(self.jk_wikimyei.wk_config['NUM_TESTS'])])
                logging.info('[STAND ALONE: INFO] epoch: %s. imu: %s' % (train_epoch, test_imu))
                if(cwcn_config.CWCN_OPTIONS.RENDER_FLAG):
                    self.jk_wikimyei._test_wikimyei_on_ahpa_(render_flag=cwcn_config.CWCN_OPTIONS.RENDER_FLAG)
                if self.best_imu is None or self.best_imu < test_imu:
                    if self.best_imu is not None:
                        name = "%s_best_%+.3f.dat" % (self.jk_wikimyei.wk_config['AHPA_ID'], test_imu)
                        logging.info("[STAND ALONE: INFO:] Best imu updated: %.3f -> %.3f : %s" % (self.best_imu, test_imu,name))
                        fname = os.path.join(self.jk_wikimyei.wk_config['CHECKPOINTS_FOLDER'], name)
                        self.jk_wikimyei._save_wikimyei_(fname)
                    self.best_imu = test_imu
                if test_imu > self.jk_wikimyei.wk_config['BREAK_TRAIN_IMU']:
                    logging.info("[STAND ALONE: WARNING:] exit jkimyei loop by BREAK_TRAIN_IMU")
                    self.early_stop = True
            if(cwcn_config.CWCN_OPTIONS.PLOT_FLAG):
                self.load_queue._plot_itm_('imu')
                # self.load_queue._plot_itm_('action,imu,alliu')
                # self.learning_queue._plot_itm_('munaajpi_imibajcho,uwaabo_imibajcho')
                # self.learning_queue._plot_itm_('ratio,surr1,surr2')
                # self.load_queue._plot_itm_('returns,advantage,gae,delta')
                self.load_queue._plot_itm_('returns,value')
                plt.show()
            if(train_epoch > self.jk_wikimyei.wk_config['BREAK_TRAIN_EPOCH']):
                logging.info("[STAND ALONE: WARNING:] exit jkimyei loop by BREAK_TRAIN_EPOCH")
                self.early_stop = True
        if(cwcn_config.CWCN_OPTIONS.RENDER_FLAG):
            self.jk_wikimyei._test_wikimyei_on_ahpa_(render_flag=cwcn_config.CWCN_OPTIONS.RENDER_FLAG)
