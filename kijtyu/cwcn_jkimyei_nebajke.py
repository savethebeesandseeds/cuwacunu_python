# --- --- --- --- 
# cwcn_jkimyei_nebajke.py
# --- --- --- --- 
import torch
import numpy as np
import torch.optim as optim
import os
import logging
from matplotlib import pyplot as plt
# --- --- --- ---
import cwcn_config
import cwcn_wikimyei_nebajke
import cwcn_wikimyei_piaabo
import cwcn_tsinuu_piaabo
import cwcn_kemu_piaabo
import cwcn_jkimyei_piaabo
# --- --- --- --- 
# --- --- --- ---
# Based on https://github.com/higgsfield/RL-Adventure-2/blob/master/3.ppo.ipynb
# Based on https://github.com/colinskow/move37/tree/master/ppo
# --- --- --- ---
class JKIMYEI_PPO:
    def __init__(self,_wikimyei):
        # --- --- 
        self.jk_wikimyei=_wikimyei
        # --- --- 
        self.optimizer=optim.Adam(_wikimyei.model.parameters(), lr=_wikimyei.wk_config['TEHDUJCO_LEARNING_RATE'])
        self.munaajpi_imibajcho_fun=torch.nn.MSELoss()
        # --- --- 
        self.ahdo_queue=cwcn_wikimyei_piaabo.AHDO_LOAD_QUEUE() # step profile queue
        self.hyper_ahdo_profile=cwcn_jkimyei_piaabo.HYPER_PROFILE_QUEUE(\
            _wikimyei.wk_config['HIPER_PROFILE_BUFFER_COUNT']) # load of step profile
        self.learning_queue=cwcn_wikimyei_piaabo.LEARNING_LOAD_QUEUE() # load of training profile
        # --- --- 
        self.uwaabo_forecast_imibajcho_fun=torch.nn.MSELoss()
        # --- --- 
    def _jkmimyei_gae_(self):
        if(not cwcn_config.ALLOW_TRAIN):
            logging.warning("Training is not allowed, skipping training")
        # --- --- 
        assert(self.ahdo_queue is not None), "Impossible to compute GAE, Jkimyei Queue found to be None"
        # --- --- 
        if(cwcn_config.CWCN_DUURUVA_CONFIG.NORMALIZE_IMU):
            self.ahdo_queue._load_normalize_(['imu'],'tensor') # not in use due to duuruva
        # --- --- 
        gamma=self.jk_wikimyei.wk_config['TEHDUJCO_GAMMA']
        lam=self.jk_wikimyei.wk_config['TEHDUJCO_GAE_LAMBDA']
        _, next_value,__ = self.jk_wikimyei.model(self.jk_wikimyei.wk_state.c_alliu.unsqueeze(0)) # dist, value, energy, certainty
        c_load_dict = self.ahdo_queue._dict_vectorize_queue_()
        c_load_dict['value'].append(next_value)
        gae = 0
        returns = []
        gae_hist = []
        advantage = []
        delta_hist = []
        for step in reversed(range(self.ahdo_queue.load_size)):
            delta = c_load_dict['imu'][step] + gamma * c_load_dict['value'][step + 1] * c_load_dict['mask'][step] - c_load_dict['value'][step]
            gae = delta + gamma * lam * c_load_dict['mask'][step] * gae
            delta_hist.insert(0,delta)
            gae_hist.insert(0,gae)                                  # prepend to get correct order back
            returns.insert(0, gae + c_load_dict['value'][step])    # prepend to get correct order back
            advantage.insert(0,gae + c_load_dict['value'][step] - c_load_dict['value'][step])   # prepend to get correct order back
            # print("waka: ",c_load_dict['index'][step],gae + c_load_dict['value'][step] - c_load_dict['value'][step])
        if(cwcn_config.CWCN_DUURUVA_CONFIG.NORMALIZE_ADVANTAGE):advantage = cwcn_kemu_piaabo.kemu_normalize(torch.cat(advantage))
        else:advantage = torch.cat(advantage)
        if(cwcn_config.CWCN_DUURUVA_CONFIG.NORMALIZE_RETURNS):returns = cwcn_kemu_piaabo.kemu_normalize(torch.cat(returns))
        else:returns = torch.cat(returns)
        self.ahdo_queue._load_vect_to_queue_('gae',gae_hist)
        self.ahdo_queue._load_vect_to_queue_('delta',delta_hist)
        self.ahdo_queue._load_vect_to_queue_('returns',returns)
        self.ahdo_queue._load_vect_to_queue_('advantage',advantage)
    def _jkimyei_ppo_update_(self):
        if(not cwcn_config.ALLOW_TRAIN):
            logging.warning("Training is not allowed, skipping training")
            return None
        # PPO EPOCHS is the number of times we will go through ALL the training data to make updates
        # --- --- --- 
        clip_param=self.jk_wikimyei.wk_config['ReferencesToNoMeButWhoThoseAllWhoMadeRechableTheImplementationOfThisAlgorithm_TEHDUJCO_EPSILON']
        # --- --- --- 
        self.hyper_ahdo_profile._standarize_select_prob_()
        # --- --- --- 
        count_epoch=0
        count_steps=0
        for c_trayectory_load in self.hyper_ahdo_profile._random_queue_yield_by_prob_(\
                _yield_count=self.jk_wikimyei.wk_config['TRAINING_EPOCHS']):
            logging.jkimyei_logging("--- Train epoch {}".format(count_epoch))
            # grabs random mini-batches several times until we have covered all data
            # --- ---
            # --- ---
            c_trayectory_load._standarize_select_prob_()
            # --- ---
            for _trayectory in c_trayectory_load._random_queue_yield_by_prob_(_dict_vectorize_flag=True,_tensorize_flag=True):
                jk_profile=cwcn_jkimyei_piaabo.LEARNING_PROFILE()
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
                dist, value, certainty=self.jk_wikimyei.model(alliu) # dist, value, energy
                # print("[size of sate:] {}, [dist:] {}".format(alliu.shape, dist))
                entropy=dist.entropy().mean()
                _, new_probs, new_log_probs,___=self.jk_wikimyei._dist_to_tsane_(dist)
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
                # --- ---                 
                if(cwcn_config.PAPER_INSTRUMENT and cwcn_config.TRAIN_ON_FORECAST):
                    jk_profile.imibajcho=\
                        self.jk_wikimyei.wk_config['UWAABO_BETA'] * self.uwaabo_forecast_imibajcho_fun(new_probs,_trayectory['tsane_non_uwaabo']) \
                            + jk_profile.munaajpi_imibajcho 
                                # + jk_profile.uwaabo_imibajcho \
                                #     - self.jk_wikimyei.wk_config['TEHDUJCO_ENTROPY_BETA'] * entropy
                else:
                    jk_profile.imibajcho=\
                        jk_profile.munaajpi_imibajcho \
                            + jk_profile.uwaabo_imibajcho \
                                - self.jk_wikimyei.wk_config['TEHDUJCO_ENTROPY_BETA'] * entropy
                # logging.jkimyei_logging("uwaabo_imibajcho: {}, \t munaajpi_imibajcho: {}, \t imibajcho: {}".format(jk_profile.uwaabo_imibajcho.size(),jk_profile.munaajpi_imibajcho.size(),jk_profile.imibajcho.size()))
                # logging.jkimyei_logging("uwaabo_imibajcho: {:.4f}, \t munaajpi_imibajcho: {:.4f}, \t imibajcho: {:.4}".format(jk_profile.uwaabo_imibajcho,jk_profile.munaajpi_imibajcho, jk_profile.imibajcho))
                # if(abs(jk_profile.uwaabo_imibajcho)>=min(abs(self.jk_wikimyei.wk_config['IMIBAJCHO_MAX']),abs(self.jk_wikimyei.wk_config['IMIBAJCHO_MIN'])) or abs(jk_profile.munaajpi_imibajcho)>=min(abs(self.jk_wikimyei.wk_config['IMIBAJCHO_MAX']),abs(self.jk_wikimyei.wk_config['IMIBAJCHO_MIN']))):
                #     logging.jkimyei_logging("[jk_profile] : {}".format([(_k,jk_profile.__dict__[_k]) for j_k in jk_profile.__dict__.keys()]))
                #     logging.jkimyei_logging("[jk_profile] : {}".format([(_k,jk_profile.__dict__[_k].shape) for _k in jk_profile.__dict__.keys() if _k not in ['p_trayectory','index','batch_size']]))
                #     input("STOP...")
                # --- ---
                self.optimizer.zero_grad()
                jk_profile.imibajcho.backward()
                self.optimizer.step()
                # --- ---
                self.learning_queue._append_(jk_profile,_detach_flag=True)
                # --- ---
                logging.jkimyei_logging("--- Train epoch {} / step : {} :: imibajcho: {:.4f},\tuwaabo_imibajcho: {:.4f},\tmunaajpi_imibajcho: {:.4f}".format(count_epoch,count_steps, jk_profile.imibajcho, jk_profile.uwaabo_imibajcho,jk_profile.munaajpi_imibajcho))
                count_steps+=1
            count_epoch+=1
    def _jkimyei_wikimyei_(self):
        if(not cwcn_config.ALLOW_TRAIN):
            logging.warning("Training is not allowed, skipping training")
            return None
        # logging.jkimyei_logging(" + + + [New jkimyei iteration]")
        self.learning_queue._reset_queue_()
        self.ahdo_queue._reset_queue_()
        self.jk_wikimyei._reset_wikimyei_()
        self.jk_wikimyei.model.train()
        for _ in range(self.jk_wikimyei.wk_config['AHDO_STEPS']):
            # --- ---
            done,ahdo_t=self.jk_wikimyei._wk_step_()
            # --- ---
            self.ahdo_queue._append_(ahdo_t,_detach_flag=True)
            if(done):
                break
        self._jkmimyei_gae_()
        self.hyper_ahdo_profile._hyper_append_(\
            _item=self.ahdo_queue,
            _total_imu=self.jk_wikimyei.wk_state.accomulated_imu,
            _selec_prob=self.jk_wikimyei.wk_state.accomulated_imu) #FIXME not uniform
        self._jkimyei_ppo_update_()
        if(cwcn_config.CWCN_CONFIG().ALWAYS_SAVING_MODEL):
            self.jk_wikimyei._save_wikimyei_(cwcn_config.CWCN_CONFIG().ALWAYS_SAVING_MODEL_PATH)
        # --- --- 
    def _standalone_wikimyei_jkimyei_ppo_loop_(self):
        if(not cwcn_config.ALLOW_TRAIN):
            logging.warning("Training is not allowed, skipping training")
            return None
        assert(self.jk_wikimyei.wk_config['AHDO_STEPS']>=self.jk_wikimyei.wk_config['MINI_BATCH_COUNT'])
        train_epoch = 0
        test_imu = None
        early_stop = False
        while not early_stop:
            train_epoch += 1
            logging.jkimyei_logging("EPOCH : {} --- --- --- --- --- --- --- --- --- --- ".format(train_epoch))
            # --- --- --- TRAIN
            self._jkimyei_wikimyei_()
            # --- --- --- Eval
            if train_epoch % self.jk_wikimyei.wk_config['VALIDATION_EPOCH'] == 0:
                early_stop, test_imu = self.jk_wikimyei._test_wikimyei_on_ahpa_(render=cwcn_config.CWCN_OPTIONS.RENDER_FLAG)
            # --- --- --- PRINT
            if(cwcn_config.CWCN_OPTIONS.PLOT_FLAG and train_epoch%cwcn_config.CWCN_OPTIONS.PLOT_INTERVAL==0):
                for _ps in cwcn_config.CWCN_OPTIONS.AHDO_PLOT_SETS:
                    self.ahdo_queue._plot_itm_(_ps)
                for _ps in cwcn_config.CWCN_OPTIONS.LEARNING_PLOT_SETS:
                    self.learning_queue._plot_itm_(_ps)
                plt.show()
            # --- --- --- Breaks
            if(test_imu is not None and test_imu > self.wk_config['BREAK_TRAIN_IMU']):
                logging.jkimyei_logging("[TEST wk: WARNING:] exit jkimyei loop by BREAK_TRAIN_IMU")
                early_stop = True
            if(train_epoch > self.jk_wikimyei.wk_config['BREAK_TRAIN_EPOCH']):
                logging.jkimyei_logging("[STAND ALONE: WARNING:] exit jkimyei loop by BREAK_TRAIN_EPOCH")
                early_stop = True
        # --- --- --- Render final result
        if(cwcn_config.CWCN_OPTIONS.RENDER_FLAG):
            _,__=self.jk_wikimyei._test_wikimyei_on_ahpa_(render_flag=cwcn_config.CWCN_OPTIONS.RENDER_FLAG)
