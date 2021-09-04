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
# --- --- --- ---
# --- --- --- ---
class WIKIMYEI:
    def __init__(self,_config,_load_file=None):
        # --- --- 
        cwcn_config.device
        # --- --- 
        logging.info("[WIKIMYEI:] building with config: {}".format(_config))
        self.ahpa=gym.make(_config['AHPA_ID'])
        # --- --- 
        self.wk_config=_config
        # --- --- 
        self.wk_config['ALLIU_COUNT']=self.ahpa.observation_space.shape[0]
        try:
            self.wk_config['TSANE_COUNT']=self.ahpa.action_space.shape[0]
        except:
            self.wk_config['TSANE_COUNT']=self.ahpa.action_space.n
        # --- --- 
        self.model=cwcn_tsinuu_piaabo.TSINUU_ACTOR_CRITIC(
            alliu_size=self.wk_config['ALLIU_COUNT'], 
            uwaabo_size=self.wk_config['TSANE_COUNT'], 
            UWAABO_HIDDEN_SIZE=self.wk_config['UWAABO_HIDDEN_SIZE'],
            MUNAAJPI_HIDDEN_SIZE=self.wk_config['MUNAAJPI_HIDDEN_SIZE'],
            sigma=1.0)
        # --- --- 
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
            logging.info("[nan case] sample:{}, log_probs:{} entropy:{}".format(sample, log_probs, entropy))
        if(self.wk_config['AHPA_ID']=="Pendulum-v0"):
            return torch.multiply(sample,2.0), sample, log_probs, entropy
        elif(self.wk_config['AHPA_ID']=="MountainCarContinuous-v0"):
            return torch.multiply(sample,2.0), sample, log_probs, entropy
        else:
            return sample, sample, log_probs, entropy # Returns [action, sample, logprob, entropy]
    def _transform_imu_(self, imu_value : torch.Tensor):
        # --- --- 
        if(self.wk_config['AHPA_ID']=="MountainCarContinuous-v0"):
            # return cwcn_kemu_piaabo.kemu_to_tensor(imu_value) + torch.dot(cwcn_kemu_piaabo.kemu_to_tensor([2.0,0.1]),cwcn_kemu_piaabo.kemu_to_tensor([self.wk_state.c_alliu[0]+0.5,abs(self.wk_state.c_alliu[1])])) + (0 if torch.round(self.wk_state.c_alliu[0]*10**3)/(10**3)==0.5 else 100)
            c_imu=self.wk_config['TEHDUJCO_IMU_BETA']*(torch.dot(cwcn_kemu_piaabo.kemu_to_tensor(
                        [250.0,500.0]),
                        cwcn_kemu_piaabo.kemu_to_tensor([(self.wk_state.c_alliu[0]-0.5),abs(self.wk_state.c_alliu[1])
                    ])) \
                + (1000 if torch.round(self.wk_state.c_alliu[0]*10**1)/(10**1)==0.5 else 0) \
                + (10 if self.wk_state.c_alliu[0]>-.2 else 0) \
                + (10 if self.wk_state.c_alliu[0]>0.0 else 0) \
                + (10 if self.wk_state.c_alliu[0]>0.2 else 0) \
                + (50 if self.wk_state.c_alliu[0]>0.4 else 0) \
                + (0 if abs(self.wk_state.c_alliu[1])>0.005 else -50) \
                + (0 if self.wk_state.c_alliu[0]>-1.1 else -100))
        else:
            c_imu=cwcn_kemu_piaabo.kemu_to_tensor(imu_value)
        # --- --- 
        c_imu=self.wk_state.imu_duuruva.duuruva_value_wrapper(c_imu) if cwcn_config.CWCN_DUURUVA_CONFIG.ENABLE_DUURUVA_IMU else c_imu
        # --- --- 
        self._imu_state = c_imu
        # --- --- 
        return c_imu
    def _reset_(self):
        self.wk_state.c_alliu=cwcn_kemu_piaabo.kemu_to_tensor(self.ahpa.reset())
        self.wk_state.accomulated_imu=cwcn_kemu_piaabo.kemu_to_tensor(0.0)
    def _wk_step_(self,ahdo_trayectory : cwcn_wikimyei_piaabo.AHDO_PROFILE = None):
        # --- ---
        if(self.wk_state.c_alliu is None):
            self._reset_()
        if(ahdo_trayectory is None):
            ahdo_trayectory=cwcn_wikimyei_piaabo.AHDO_PROFILE()
        # --- ---
        dist, ahdo_trayectory.value=self.model(self.wk_state.c_alliu)
        ahdo_trayectory.tsane, _, ahdo_trayectory.log_prob, ahdo_trayectory.entropy=self._dist_to_tsane_(dist)
        # print("[size of sate:] {}, [tsane:] {}".format(self.wk_state.c_alliu, ahdo_trayectory.tsane))
        # --- ---
        c_alliu, c_imu, c_done, _=self.ahpa.step(ahdo_trayectory.tsane.cpu().numpy())
        # logging.info("reward : {}, State : {}".format(imu,next_state))
        # --- ---
        self.wk_state.c_alliu=cwcn_kemu_piaabo.kemu_to_tensor(c_alliu)
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
        self._reset_()
        while True: # Test until is ready
            ctx_steps+=1
            done, ahdo_t=self._wk_step_()
            total_imu+=ahdo_t.imu
            if(render_flag):
                self.ahpa.render()
            if(done):
                break
        total_imu/=ctx_steps
        logging.info("[WIKIMYEI:] tested, total mean imu : {}".format(total_imu))
        return total_imu
# --- --- --- ---