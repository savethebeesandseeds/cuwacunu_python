# --- --- --- ---
# cwcn_wikimyei_nebajke.py
# --- --- --- ---
# --- --- --- --- extrange imports keep beeing imported not by me
import torch
import torch.nn as nn
import math
import os
import logging
import numpy as np
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
        self.model=cwcn_tsinuu_piaabo.TSINUU_as_marginal_method(
            ALLIU_SIZE=cwcn_config.CWCN_UJCAMEI_CAJTUCU_CONFIG.ALLIU_COUNT, 
            UWAABO_SIZE=cwcn_config.CWCN_UJCAMEI_CAJTUCU_CONFIG.TSANE_COUNT, 
            MUNAAJPI_SIZE=1,
            HIDDEN_BELLY_SIZE=self.wk_config['HIDDEN_BELLY_SIZE'],
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
        self.best_imu = None
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
        probs=dist_d.probs
        if(torch.any(torch.isnan(sample)) or torch.any(torch.isnan(log_probs))):
            logging.warning("[nan case] sample:{}, log_probs:{} entropy:{}".format(sample, log_probs, entropy))
        return sample, probs, log_probs, entropy # Returns [action, sample, logprob, entropy]
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
        
    def _wk_step_(self,ahdo_profile : cwcn_wikimyei_piaabo.AHDO_PROFILE = None):
        # --- ---
        if(self.wk_state.c_alliu is None):
            self._reset_wikimyei_()
        if(ahdo_profile is None):
            ahdo_profile=cwcn_wikimyei_piaabo.AHDO_PROFILE()
        # --- ---
        dist, ahdo_profile.value, certainty = self.model(self.wk_state.c_alliu.unsqueeze(0))
        ahdo_profile.tsane, _, ahdo_profile.log_prob, ahdo_profile.entropy=self._dist_to_tsane_(dist)
        # print("[size of sate:] {}, [tsane:] {}".format(self.wk_state.c_alliu, ahdo_profile.tsane))
        # --- ---
        c_alliu, c_imu, c_done, c_info=self.ahpa.step(
            _tsane=ahdo_profile.tsane,
            _certainty=certainty.squeeze(0)
            )
        if(cwcn_config.PAPER_INSTRUMENT and cwcn_config.TRAIN_ON_FORECAST): #here is unnessary, if the intention is just to model the reward
            ahdo_profile.forecast_non_uwaabo=c_info['non_uwaabo_forecast_was_value']
            def _mod_sech_(x):
                scale=9
                bias=0.001
                pot=2
                return pot/(math.exp(scale*x)+math.exp(-scale*x))+bias
            def _mod_tanh_(x):
                scale=6
                return 0.5*(math.exp(scale*x)-math.exp(-scale*x))/(math.exp(scale*x)+math.exp(-scale*x))+0.501
            c_t_map=list(cwcn_config.CWCN_UJCAMEI_CAJTUCU_CONFIG.TSANE_ACTION_DICT.values())
            ahdo_profile.tsane_non_uwaabo=[0]*len(c_t_map)
            ahdo_profile.tsane_non_uwaabo[c_t_map.index('put')]=_mod_tanh_(-ahdo_profile.forecast_non_uwaabo)
            ahdo_profile.tsane_non_uwaabo[c_t_map.index('pass')]=_mod_sech_(ahdo_profile.forecast_non_uwaabo)
            ahdo_profile.tsane_non_uwaabo[c_t_map.index('call')]=_mod_tanh_(ahdo_profile.forecast_non_uwaabo)
            ahdo_profile.tsane_non_uwaabo[c_t_map.index('put')]/=sum(ahdo_profile.tsane_non_uwaabo)
            ahdo_profile.tsane_non_uwaabo[c_t_map.index('pass')]/=sum(ahdo_profile.tsane_non_uwaabo)
            ahdo_profile.tsane_non_uwaabo[c_t_map.index('call')]/=sum(ahdo_profile.tsane_non_uwaabo)
            ahdo_profile.tsane_non_uwaabo=torch.Tensor(ahdo_profile.tsane_non_uwaabo).to(cwcn_config.device)
            
        ahdo_profile.price=c_info['price_was_value']
        ahdo_profile.certainty = c_info['certainty_was']
        ahdo_profile.put_certainty = c_info['put_certainty_was_value']
        ahdo_profile.pass_certainty = c_info['pass_certainty_was_value']
        ahdo_profile.call_certainty = c_info['call_certainty_was_value']
        # logging.info("reward : {}, State : {}".format(imu,next_state))
        # --- ---
        self.wk_state.c_alliu=c_alliu
        ahdo_profile.alliu=self.wk_state.c_alliu
        # --- ---
        ahdo_profile.imu=self._transform_imu_(c_imu)
        self.wk_state.accomulated_imu+=ahdo_profile.imu
        # logging.info("Reward : {}, transformed_reward: {}".format(c_imu, ahdo_profile.imu))
        ahdo_profile.mask=cwcn_kemu_piaabo.kemu_to_tensor(1 - bool(c_done))
        ahdo_profile.done=cwcn_kemu_piaabo.kemu_to_tensor(c_done)
        ahdo_profile.selec_prob=1.0 #FIXME make it not uniform
        # logging.info("[STATE:] {}, [ACTION:] {}, [DONE:] {}".format(ahdo_profile.alliu,ahdo_profile.action,ahdo_profile.done))
        return c_done, ahdo_profile
    
    def _run_wk_episode_(self, render_flag=False): 
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
        logging.wikimyei_logging("tested, total mean imu : {} : total_steps : {}".format(total_imu,ctx_steps))
        return total_imu
    
    def _test_wikimyei_on_ahpa_(self):
        # This method rund on model.eval mode
        self.model.eval()
        early_stop = False
        test_imu = np.mean([self._run_wk_episode_(render_flag=False) for _ in range(self.wk_config['NUM_TESTS'])])
        logging.wikimyei_logging('[TEST wk: INFO] imu: %s' % (test_imu))
        if(cwcn_config.CWCN_OPTIONS.RENDER_FLAG):
            self._run_wk_episode_(render_flag=cwcn_config.CWCN_OPTIONS.RENDER_FLAG)
        if self.best_imu is None or self.best_imu < test_imu:
            if self.best_imu is not None:
                name = "%s_best_%+.3f.dat" % (self.wk_config['AHPA_ID'], test_imu)
                logging.wikimyei_logging("[TEST wk: INFO:] Best imu updated: %.3f -> %.3f : %s" % (self.best_imu, test_imu,name))
                fname = os.path.join(self.wk_config['CHECKPOINTS_FOLDER'], name)
                self._save_wikimyei_(fname)
            self.best_imu = test_imu
        return early_stop, test_imu
        
    def _standalone_wikimyei_loop_(self, render_flag=False):
        total_imu=None
        try:
            self.model.eval()
            total_imu=self._run_wk_episode_()
            logging.wikimyei_logging("standalone, total mean imu : {}".format(total_imu))
        except Exception as e:
            logging.error("{}".format(e))
        return total_imu
# --- --- --- ---