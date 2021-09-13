# --- --- --- ---
from numpy import nan
import torch
import torch.nn as nn
import gym
# --- --- --- ---
import os
import sys
import logging
# --- --- --- ---
import cwcn_config
import cwcn_wikimyei_piaabo
import cwcn_jkimyei_nebajke
import cwcn_tsinuu_piaabo
import cwcn_kemu_piaabo
import cwcn_ujcamei_cajtucu_piaabo
# --- --- --- ---
# --- --- --- ---
class WIKIMYEI:
    def __init__(self,_config,_load_file=None):
        # --- --- 
        logging.info("[WIKIMYEI:] building with config: {}".format(_config))
        self.wk_config=_config
        # --- --- 
        self.model=cwcn_tsinuu_piaabo.TSINUU_ACTOR_CRITIC(
            ALLIU_SIZE=cwcn_config.CWCN_UJCAMEI_CAJTUCU_CONFIG.ALLIU_COUNT, 
            UWAABO_SIZE=cwcn_config.CWCN_UJCAMEI_CAJTUCU_CONFIG.TSANE_COUNT, 
            RECURRENT_TYPE=self.wk_config['RECURRENT_TYPE'],
            RECURRENT_SEQ_SIZE=cwcn_config.CWCN_UJCAMEI_CAJTUCU_CONFIG.UJCAMEI_ALLIU_SEQUENCE_SIZE, 
            RECURRENT_HIDEN_SIZE=self.wk_config['RECURRENT_HIDEN_SIZE'], 
            RECURRENT_N_LAYERS=self.wk_config['RECURRENT_N_LAYERS'], 
            UWAABO_HIDDEN_SIZE=self.wk_config['UWAABO_HIDDEN_SIZE'],
            MUNAAJPI_HIDDEN_SIZE=self.wk_config['MUNAAJPI_HIDDEN_SIZE'])
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1: # requires cuda
                self.model = nn.DataParallel(self.model) #FIXME understand this
        self.model.to(cwcn_config.device)
        # --- --- 
        self.wk_state=cwcn_wikimyei_piaabo.WIKIMYEI_STATE(
            _wikimyei_config=self.wk_config
        )
        # --- --- 
        self.jkimyei=cwcn_jkimyei_nebajke.JKIMYEI_PPO(self)
        # --- --- 
        if(_load_file is not None and os.path.isfile(_load_file)):
            self._load_wikimyei_(_load_file)
        # --- --- 
        self.hyper_load = None
        # --- --- 
        assert(_config['AHPA_ID'] == 'UjcameiCajtucu-v0'), "Please use other version of wikimyei_nebajke, or select the enviroment (Ahdo) as <UjcameiCajtucu-v0>"
        self.wk_config['ALLIU_COUNT']=cwcn_config.CWCN_UJCAMEI_CAJTUCU_CONFIG.ALLIU_COUNT
        self.wk_config['TSANE_COUNT']=cwcn_config.CWCN_UJCAMEI_CAJTUCU_CONFIG.TSANE_COUNT
        self.ahpa=cwcn_ujcamei_cajtucu_piaabo.KUJTIYU_UJCAMEI_CAJTUCU(self)
        # --- --- 
    def _save_wikimyei_(self,__path): # add model config
        logging.info("saving model to file <{}>".format(__path))
        torch.save((self.model.state_dict(), self.jkimyei.optimizer.state_dict(), self.wk_config), __path)
    def _load_wikimyei_(self,__path):
        logging.info("loading model from file <{}>".format(__path))
        tsinuu_state, optimizer_state, config_state = torch.load(__path)
        self.model.load_state_dict(tsinuu_state)
        self.jkimyei.optimizer.load_state_dict(optimizer_state)
        self.wk_config = config_state
    def _dist_to_tsane_(self,dist_d,deterministic=True):
        # action = self.trayectory.dist.mean.detach().cpu().numpy()[0] if deterministic \
        #         else self.trayectory.dist.sample().cpu().numpy()[0]
        sample=dist_d.sample()
        assert(torch.is_tensor(sample))
        entropy=dist_d.entropy().mean()
        log_probs=dist_d.log_prob(sample)
        if(torch.any(torch.isnan(sample)) or torch.any(torch.isnan(log_probs))):
            logging.warning("[nan case] sample:{}, log_probs:{} entropy:{}".format(sample, log_probs, entropy))
        return sample, sample, log_probs, entropy # Returns [action, sample, logprob, entropy]
    def _transform_imu_(self, imu_value : torch.Tensor):
        # --- --- 
        c_imu=cwcn_kemu_piaabo.kemu_to_tensor(self.wk_config['TEHDUJCO_IMU_BETA']*imu_value)
        # --- --- 
        c_imu=self.wk_state.imu_duuruva._duuruva_value_wrapper_(c_imu) if cwcn_config.CWCN_DUURUVA_CONFIG.ENABLE_DUURUVA_IMU else c_imu
        # --- --- 
        self._imu_state = c_imu
        # --- --- 
        return c_imu
    def _reset_wikimyei_(self):
        self.wk_state._reset_()
        self.wk_state.c_alliu=self.ahpa.reset()
        logging.wikimyei_logging("[reset]")
        
    def _wk_step_(self,ahdo_trayectory : cwcn_wikimyei_piaabo.AHDO_PROFILE = None):
        # --- ---
        if(self.wk_state.c_alliu is None):
            self._reset_wikimyei_()
        if(ahdo_trayectory is None):
            ahdo_trayectory=cwcn_wikimyei_piaabo.AHDO_PROFILE()
        # --- ---
        dist, ahdo_trayectory.value, energy = self.model(self.wk_state.c_alliu.unsqueeze(0))
        ahdo_trayectory.tsane, _, ahdo_trayectory.log_prob, ahdo_trayectory.entropy=self._dist_to_tsane_(dist)
        # print("[size of sate:] {}, [tsane:] {}".format(self.wk_state.c_alliu, ahdo_trayectory.tsane))
        # --- ---
        c_alliu, c_imu, c_done, _=self.ahpa.step(ahdo_trayectory.tsane)
        # logging.info("reward : {}, State : {}".format(imu,next_state))
        # --- ---
        self.wk_state.c_alliu=c_alliu
        ahdo_trayectory.alliu=self.wk_state.c_alliu
        # --- ---
        ahdo_trayectory.imu=self._transform_imu_(c_imu)
        self.wk_state.accomulated_imu+=ahdo_trayectory.imu
        # logging.info("Reward : {}, transformed_reward: {}".format(c_imu, ahdo_trayectory.imu))
        ahdo_trayectory.mask=cwcn_kemu_piaabo.kemu_to_tensor(1 - bool(c_done))
        ahdo_trayectory.done=cwcn_kemu_piaabo.kemu_to_tensor(c_done)
        ahdo_trayectory.selec_prob=1.0 #FIXME make it not uniform
        # logging.info("[STATE:] {}, [ACTION:] {}, [DONE:] {}".format(ahdo_trayectory.alliu,ahdo_trayectory.action,ahdo_trayectory.done))
        return c_done, ahdo_trayectory
    def _test_wikimyei_on_ahpa_(self,render_flag=False):
        total_imu=cwcn_kemu_piaabo.kemu_to_tensor(0.)
        ctx_steps=0
        self._reset_wikimyei_()
        while True: # Test until is ready
            ctx_steps+=1
            done, ahdo_t=self._wk_step_()
            total_imu+=ahdo_t.imu
            if(render_flag):
                self.ahpa.render()
            if(done):
                break
        total_imu/=ctx_steps
        logging.wikimyei_logging("tested, total mean imu : {}".format(total_imu))
        return total_imu
# --- --- --- ---