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
        self.ratio      = None
        self.surr1      = None
        self.surr2      = None
        self.uwaabo_loss = None
        self.munaajpi_loss = None
        self.loss       = None
        self.index      = None
        self.batch_size = None
# --- --- --- ---
# --- --- --- ---
class RAY_ORDER_JKIMYEI: # use pytorch/ray to optimize hyperparameters
    def __init__(self):
        # --- --- --- 
        logging.ray_log("--- RAY system is initialized ----")
        os.environ['CWCN_CONFIG_SYSTEM']='ray_system'
        self.checkpoint_file=os.path.join(cwcn_config.CWCN_CONFIG().RAY_CHECKPOINTS_FOLDER,"checkpoint")
        self.__rjk_config=cwcn_config.CWCN_CONFIG().__dict__
        self.ray_wikimyei=None
        # --- --- --- 
    def _report_(self,_reward):
        # --- --- --- 
        tune.report(reward=_reward)
        # --- --- --- 
    def _ray_iteration_(self,config):
        # --- --- --- 
        logging.ray_log("--- ray step ---")
        self.ray_wikimyei=cwcn_wikimyei_nebajke.WIKIMYEI(config)
        self.ray_wikimyei.jkimyei._wikimyei_jkimyei_()
        c_reward=self.ray_wikimyei._test_wikimyei_on_env_(render_flag=False)
        self.ray_wikimyei._save_wikimyei_(self.checkpoint_file)
        self._report_(c_reward) # tune.report(loss=(val_loss / val_steps), accuracy=correct / total)
        logging.ray_log("--- [REPORT_LEVEL]_:reward:_{}_:config:_{}_:end:_".format(c_reward,self.ray_wikimyei.wk_config))
        logging.ray_log("--- ray step ended ---")
        # --- --- --- 
    def _export_best_trail_(self,result):
        # --- --- --- 
        # best_trial = result.get_best_trial("reward", "max", "last")
        # print("Best trial config: {}".format(best_trial.config))
        # print("Best trial final validation reward: {}".format(best_trial.last_result["reward"]))
        # print("Best trial test set reward: {}".format(c_reward))
        # logging.ray_log("[RESULTS:] \n{} ".format(list(result.results_df.columns.values)))
        # --- --- --- 
        # aux_list=result.results_df['reward'].apply(lambda x:x.item())
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
            metric="reward",
            mode="max",
            max_t=9999999,
            grace_period=1,
            reduction_factor=2)
        reporter = CLIReporter(# parameter_columns=["config in general", "l2", "lr", "batch_size"],
            metric_columns=["reward", "training_iteration"],
            parameter_columns=["reward", "training_iteration"],
            sort_by_metric=True)
        assert(not torch.cuda.is_available())
        result = tune.run(
            self._ray_iteration_,
            resources_per_trial={"cpu": 4, "gpu": 0}, #FIXME add when cuda is aviable
            config=self.__rjk_config,
            num_samples=self.__rjk_config['RAY_N_TRAILS'],
            scheduler=scheduler,
            progress_reporter=reporter,
            raise_on_failed_trial=False)
        # --- --- --- 
        self._export_best_trail_(result)
        # --- --- --- 
        # self.ray_wikimyei=cwcn_wikimyei_nebajke.WIKIMYEI(self.checkpoint_file)
        # c_reward=self.ray_wikimyei._test_wikimyei_on_env_(render_flag=True)
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
                aux_info["reward"]=float(re.findall(r"(?<=_:reward:_)(.*)(?=_:config:_)",_c)[0])
                aux_info["config"]=ast.literal_eval(re.findall(r"(?<=_:config:_)(.*)(?=_:end:_)",_c)[0]) # interesting function
                _logs_info.append(aux_info)
        _logs_info=sorted(_logs_info,key=(lambda x: x['reward']), reverse=True)
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
        self.mini_batch_size=_wikimyei.wk_config['MINI_BATCH_SIZE']
        self.optimizer=optim.Adam(_wikimyei.model.parameters(), lr=_wikimyei.wk_config['LEARNING_RATE'])
        self.munaajpi_loss_fun=torch.nn.MSELoss()
        self.learning_queue=cwcn_wikimyei_piaabo.LOAD_QUEUE()
        # self.hist_learning_queue=cwcn_wikimyei_piaabo.LOAD_QUEUE()
        self.load_queue=cwcn_wikimyei_piaabo.LOAD_QUEUE()
        self.jk_wikimyei=_wikimyei
    def _jkmimyei_gae_(self):
        assert(self.load_queue is not None), "Impossible to compute GAE, Jkimyei Queue found to be None"
        gamma=self.jk_wikimyei.wk_config['GAMMA']
        lam=self.jk_wikimyei.wk_config['GAE_LAMBDA']
        _, next_value = self.jk_wikimyei.model(self.jk_wikimyei.w_state.c_state)
        c_load_dict = self.load_queue._dict_vectorize_queue_()
        c_load_dict['value'].append(next_value)
        gae = 0
        returns = []
        gae_hist = []
        advantage = []
        delta_hist = []
        for step in reversed(range(self.load_queue.load_size)):
            delta = c_load_dict['reward'][step] + gamma * c_load_dict['value'][step + 1] * c_load_dict['mask'][step] - c_load_dict['value'][step]
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
        clip_param=self.jk_wikimyei.wk_config['PPO_EPSILON']
        count_steps=0
        for _ in range(self.jk_wikimyei.wk_config['PPO_EPOCHS']):
            # grabs random mini-batches several times until we have covered all data
            for _trayectory in self._random_queue_yield_():
                jk_profile=LEARNING_PROFILE()
                jk_profile.batch_size=len(_trayectory['state'])
                # jk_profile.p_trayectory=_trayectory #FIXME, not in use, not appended correctly
                # --- ---
                # _trayectory['state'].requires_graph=True
                # _trayectory['action'].requires_graph=True
                # _trayectory['log_prob'].requires_graph=True
                # print("STATE FORM:",_trayectory['state'])
                state=torch.detach(_trayectory['state'])
                old_log_probs=torch.detach(_trayectory['log_prob'])
                advantage=torch.detach(_trayectory['advantage'])
                returns=torch.detach(_trayectory['returns'])
                # print("[advantage!:] {}".format(advantage))
                # print("[index!:] {}".format([_trayectory['index']]))
                # --- --- 
                state.requires_grad=True
                old_log_probs.requires_grad=True
                advantage.requires_grad=True
                returns.requires_grad=True
                # --- --- 
                dist, value=self.jk_wikimyei.model(state)
                # print("[size of sate:] {}, [dist:] {}".format(state.shape, dist))
                entropy=dist.entropy().mean()
                _, __, new_log_probs,___=self.jk_wikimyei._dist_to_action_(dist)
                # --- ---
                jk_profile.ratio=(new_log_probs - old_log_probs).exp()
                jk_profile.surr1=jk_profile.ratio * advantage
                jk_profile.surr2=torch.clamp(jk_profile.ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage
                # --- ---
                # IITEPI_BETA... #FIXME add iitepi as the exploration network
                jk_profile.uwaabo_loss =- self.jk_wikimyei.wk_config['UWAABO_BETA'] * torch.min(jk_profile.surr1, jk_profile.surr2).mean() #FIXME mean is tricky, util only due to pytorch great tools
                # jk_profile.munaajpi_loss=self.jk_wikimyei.wk_config['MUNAAJPI_BETA'] * (returns - value).pow(2).mean() #FIXME mean is tricky to implement in c, using a sign multiplication
                jk_profile.munaajpi_loss=self.jk_wikimyei.wk_config['MUNAAJPI_BETA'] * self.munaajpi_loss_fun(returns, value)
                jk_profile.uwaabo_loss=torch.clamp(jk_profile.uwaabo_loss,min=self.jk_wikimyei.wk_config['LOSS_MIN'],max=self.jk_wikimyei.wk_config['LOSS_MAX'])
                jk_profile.munaajpi_loss=torch.clamp(jk_profile.munaajpi_loss,min=self.jk_wikimyei.wk_config['LOSS_MIN'],max=self.jk_wikimyei.wk_config['LOSS_MAX'])
                jk_profile.loss= jk_profile.munaajpi_loss + jk_profile.uwaabo_loss - self.jk_wikimyei.wk_config['ENTROPY_BETA'] * entropy
                # logging.info("uwaabo_loss: {}, \t munaajpi_loss: {}, \t loss: {}".format(jk_profile.uwaabo_loss.size(),jk_profile.munaajpi_loss.size(),jk_profile.loss.size()))
                # logging.info("uwaabo_loss: {:.4f}, \t munaajpi_loss: {:.4f}, \t loss: {:.4}".format(jk_profile.uwaabo_loss,jk_profile.munaajpi_loss, jk_profile.loss))
                # if(abs(jk_profile.uwaabo_loss)>=min(abs(self.jk_wikimyei.wk_config['LOSS_MAX']),abs(self.jk_wikimyei.wk_config['LOSS_MIN'])) or abs(jk_profile.munaajpi_loss)>=min(abs(self.jk_wikimyei.wk_config['LOSS_MAX']),abs(self.jk_wikimyei.wk_config['LOSS_MIN']))):
                #     logging.info("[jk_profile] : {}".format([(_k,jk_profile.__dict__[_k]) for j_k in jk_profile.__dict__.keys()]))
                #     logging.info("[jk_profile] : {}".format([(_k,jk_profile.__dict__[_k].shape) for _k in jk_profile.__dict__.keys() if _k not in ['p_trayectory','index','batch_size']]))
                #     input("STOP...")
                # --- ---
                self.optimizer.zero_grad()
                jk_profile.loss.backward()
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
        for _ in range(self.jk_wikimyei.wk_config['PPO_STEPS']):
            # --- ---
            __wt=cwcn_wikimyei_piaabo.TRAYECTORY()
            done=self.jk_wikimyei._wk_step_(__wt)
            # --- ---
            self.load_queue._append_(__wt)
            if(done):
                if(self.load_queue.load_size<self.jk_wikimyei.wk_config['MINI_BATCH_SIZE']):
                    self.mini_batch_size=self.load_queue.load_size
                else:
                    self.mini_batch_size=self.jk_wikimyei.wk_config['MINI_BATCH_SIZE']
                break
        # self.load_queue._load_normalize_(['reward'],'tensor') # not in use due to duuruva
        # if(>self.best_reward):
        # self.hist_learning_queue._import_queue_(self.learning_queue)
        self._jkmimyei_gae_()
        self._jkimyei_ppo_update_()
        # --- --- 
    def _standalone_wikimyei_jkimyei_loop_(self):
        assert(self.jk_wikimyei.wk_config['PPO_STEPS']>=self.jk_wikimyei.wk_config['MINI_BATCH_SIZE'])
        train_epoch = 0
        self.best_reward = None
        self.early_stop = False
        # self.hist_learning_queue._reset_queue_()
        while not self.early_stop:
            train_epoch += 1
            # --- --- --- TRAIN
            self._wikimyei_jkimyei_()
            # --- --- --- Eval
            if train_epoch % self.jk_wikimyei.wk_config['TEST_EPOCHS'] == 0:
                test_reward = np.mean([self.jk_wikimyei._test_wikimyei_on_env_(render_flag=False) for _ in range(self.jk_wikimyei.wk_config['NUM_TESTS'])])
                logging.info('[STAND ALONE: INFO] epoch: %s. reward: %s' % (train_epoch, test_reward))
                if(cwcn_config.CWCN_OPTIONS.RENDER_FLAG):
                    self.jk_wikimyei._test_wikimyei_on_env_(render_flag=cwcn_config.CWCN_OPTIONS.RENDER_FLAG)
                if self.best_reward is None or self.best_reward < test_reward:
                    if self.best_reward is not None:
                        name = "%s_best_%+.3f.dat" % (self.jk_wikimyei.wk_config['ENV_ID'], test_reward)
                        logging.info("[STAND ALONE: INFO:] Best reward updated: %.3f -> %.3f : %s" % (self.best_reward, test_reward,name))
                        fname = os.path.join(self.jk_wikimyei.wk_config['CHECKPOINTS_FOLDER'], name)
                        self.jk_wikimyei._save_wikimyei_(fname)
                    self.best_reward = test_reward
                if test_reward > self.jk_wikimyei.wk_config['TARGET_REWARD']:
                    logging.info("[STAND ALONE: WARNING:] exit jkimyei loop by TARGET_REWARD")
                    self.early_stop = True
            if(cwcn_config.CWCN_OPTIONS.PLOT_FLAG):
                self.load_queue._plot_itm_('reward')
                # self.load_queue._plot_itm_('action,reward,state')
                # self.learning_queue._plot_itm_('munaajpi_loss,uwaabo_loss')
                # self.learning_queue._plot_itm_('ratio,surr1,surr2')
                # self.load_queue._plot_itm_('returns,advantage,gae,delta')
                self.load_queue._plot_itm_('returns,value')
                plt.show()
            if(train_epoch > self.jk_wikimyei.wk_config['BREAK_TRAIN_EPOCH']):
                logging.info("[STAND ALONE: WARNING:] exit jkimyei loop by BREAK_TRAIN_EPOCH")
                self.early_stop = True
        if(cwcn_config.CWCN_OPTIONS.RENDER_FLAG):
            self.jk_wikimyei._test_wikimyei_on_env_(render_flag=cwcn_config.CWCN_OPTIONS.RENDER_FLAG)
