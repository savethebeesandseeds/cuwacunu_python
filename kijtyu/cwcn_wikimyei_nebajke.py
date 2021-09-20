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
            FORECAST_HIDDEN_SIZE=self.wk_config['FORECAST_HIDDEN_SIZE'],
            FORECAST_N_HORIZONS=self.wk_config['FORECAST_N_HORIZONS'],
            UWAABO_HIDDEN_SIZE=self.wk_config['UWAABO_HIDDEN_SIZE'],
            MUNAAJPI_HIDDEN_SIZE=self.wk_config['MUNAAJPI_HIDDEN_SIZE'])
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1: # requires cuda
                self.model = nn.DataParallel(self.model) #FIXME understand this
        # --- --- 
        self.model.to(cwcn_config.device)
        # --- --- 
        self.jkimyei=cwcn_jkimyei_nebajke.JKIMYEI_PPO(self)
        # --- --- 
        if(_load_file is not None and os.path.isfile(_load_file)):
            self._load_wikimyei_(_load_file)
            logging.info("[WIKIMYEI:] running with config: {}".format(_config))
        # --- --- 
        for p in self.model.forescast_parameters: # only requires grad when train on forescat is solicited
            p.requires_grad=cwcn_config.PAPER_INSTRUMENT and cwcn_config.TRAIN_ON_FORECAST
        # --- --- 
        self.hyper_load = None
        self.wk_state=cwcn_wikimyei_piaabo.WIKIMYEI_STATE(
            _wikimyei_config=self.wk_config
        )
        self.best_imu = None
        # --- --- 
        assert(_config['AHPA_ID'] == 'UjcameiCajtucu-v0'), "Please use other version of wikimyei_nebajke, or select the enviroment (Ahdo) as <UjcameiCajtucu-v0>"
        self.wk_config['ALLIU_COUNT']=cwcn_config.CWCN_UJCAMEI_CAJTUCU_CONFIG.ALLIU_COUNT
        self.wk_config['TSANE_COUNT']=cwcn_config.CWCN_UJCAMEI_CAJTUCU_CONFIG.TSANE_COUNT
        self.ahpa=cwcn_ujcamei_cajtucu_piaabo.KUJTIYU_UJCAMEI_CAJTUCU(self)
        # --- --- 
    def _save_wikimyei_(self,__path): # add model config
        logging.info("saving model to file <{}>".format(__path))
        torch.save((self.model.state_dict(), self.jkimyei.rl_optimizer.state_dict(), self.jkimyei.forecast_optimizer.state_dict(), self.wk_config), __path)
    def _load_wikimyei_(self,__path):
        logging.info("loading model from file <{}>".format(__path))
        model_state_dict, rl_optimizer_state, forecast_optimizer_state, load_config_state = torch.load(__path)
        self.model.load_state_dict(model_state_dict)
        self.jkimyei.rl_optimizer.load_state_dict(rl_optimizer_state)
        self.jkimyei.forecast_optimizer.load_state_dict(forecast_optimizer_state)
        # self.wk_config = None # config_state # do not load the config into the model, inestable
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
        if(cwcn_config.GREEDY_TSANE_SAMPLE):
            sample=probs.argmax()
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
        tsane_dist, ahdo_profile.value, certainty, c_forecast_uwaabo = self.model(self.wk_state.c_alliu.unsqueeze(0))
        ahdo_profile.tsane, _, ahdo_profile.log_prob, ahdo_profile.entropy=self._dist_to_tsane_(tsane_dist)
        # print("[size of sate:] {}, [tsane:] {}".format(self.wk_state.c_alliu, ahdo_profile.tsane))
        # --- ---
        c_alliu, c_imu, c_done, c_info = self.ahpa.step(
            _tsane=ahdo_profile.tsane,
            _certainty=certainty.squeeze(0)
            )
        if(cwcn_config.PAPER_INSTRUMENT and cwcn_config.TRAIN_ON_FORECAST): #here is unnesesary, if the intention is just to model the reward
            # def _mod_sech_(x): #FIXME better
            #     scale=9
            #     bias=0.001
            #     pot=2
            #     return pot/(math.exp(scale*x)+math.exp(-scale*x))+bias
            # def _mod_tanh_(x): #FIXME better
            #     scale=6
            #     return 0.5*(math.exp(scale*x)-math.exp(-scale*x))/(math.exp(scale*x)+math.exp(-scale*x))+0.501
            # for c_forecast_horizon in cwcn_config.FORECAST_HORIZONS:
            #     ahdo_profile.forecast_uwaabo_vect[':{}'.format(c_forecast_horizon)]=c_info['non_uwaabo_forecast_was_value:{}'.format(c_forecast_horizon)]
            #     c_t_map=list(cwcn_config.CWCN_UJCAMEI_CAJTUCU_CONFIG.TSANE_ACTION_DICT.values())
            #     ahdo_profile.tsane_non_uwaabo[':{}'.format(c_forecast_horizon)]=[0]*len(c_t_map)
            #     ahdo_profile.tsane_non_uwaabo[':{}'.format(c_forecast_horizon)][c_t_map.index('put')]=_mod_tanh_(-ahdo_profile.forecast_uwaabo_vect[':{}'.format(c_forecast_horizon)])
            #     ahdo_profile.tsane_non_uwaabo[':{}'.format(c_forecast_horizon)][c_t_map.index('pass')]=_mod_sech_(ahdo_profile.forecast_uwaabo_vect[':{}'.format(c_forecast_horizon)])
            #     ahdo_profile.tsane_non_uwaabo[':{}'.format(c_forecast_horizon)][c_t_map.index('call')]=_mod_tanh_(ahdo_profile.forecast_uwaabo_vect[':{}'.format(c_forecast_horizon)])
            #     ahdo_profile.tsane_non_uwaabo[':{}'.format(c_forecast_horizon)][c_t_map.index('put')]/=sum(ahdo_profile.tsane_non_uwaabo[':{}'.format(c_forecast_horizon)])
            #     ahdo_profile.tsane_non_uwaabo[':{}'.format(c_forecast_horizon)][c_t_map.index('pass')]/=sum(ahdo_profile.tsane_non_uwaabo[':{}'.format(c_forecast_horizon)])
            #     ahdo_profile.tsane_non_uwaabo[':{}'.format(c_forecast_horizon)][c_t_map.index('call')]/=sum(ahdo_profile.tsane_non_uwaabo[':{}'.format(c_forecast_horizon)])
            #     ahdo_profile.tsane_non_uwaabo[':{}'.format(c_forecast_horizon)]=torch.Tensor(ahdo_profile.tsane_non_uwaabo[':{}'.format(c_forecast_horizon)]).to(cwcn_config.device)
            forecast_uwaabo_vect=[]
            forecast_non_uwaabo_vect=[]
            for c_index_oredx,c_forecast_horizon in enumerate(cwcn_config.FORECAST_HORIZONS):
                # this method defines the orden of forecast_uwaabo_vect as a vector
                forecast_uwaabo_vect.append(c_forecast_uwaabo)
                forecast_non_uwaabo_vect.append(c_info['non_uwaabo_forecast_was_value:{}'.format(c_forecast_horizon)])
                # ahdo_profile.tsane_non_uwaabo[':{}'.format(c_forecast_horizon)]=ahdo_profile.forecast_uwaabo_vect[':{}'.format(c_forecast_horizon)] #FIXME there is no tsane derived from forecast ... 
                # ahdo_profile.tsane_non_uwaabo[':{}'.format(c_forecast_horizon)]=None
            ahdo_profile.forecast_uwaabo_vect=torch.stack(forecast_uwaabo_vect)
            ahdo_profile.forecast_non_uwaabo_vect=torch.stack(forecast_non_uwaabo_vect)
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
        if(self.wk_config['APPEND_AHDO_QUEUE_PROFILE']=='uniform'):
            ahdo_profile.selec_prob=1.0 #FIXME make it not uniform
        elif(self.wk_config['APPEND_AHDO_QUEUE_PROFILE']=='imu'):
            ahdo_profile.selec_prob=abs(ahdo_profile.imu.detach().item())+0.01
        else:
            assert(False), "configure a valid APPEND_AHDO_QUEUE_PROFILE : ['uniform','imu']"
        # logging.info("[STATE:] {}, [ACTION:] {}, [DONE:] {}".format(ahdo_profile.alliu,ahdo_profile.action,ahdo_profile.done))
        return c_done, ahdo_profile
    
    def _run_wk_episode_(self, render_flag=False, n_episodes=None): 
        total_imu=cwcn_kemu_piaabo.kemu_to_tensor(0.)
        ctx_steps=0
        self._reset_wikimyei_()
        while True: # Test until is ready
            ctx_steps+=1
            done, ahdo_t=self._wk_step_()
            total_imu+=ahdo_t.imu
            if(render_flag):
                self.ahpa.render()
            if(done or ctx_steps>=self.wk_config['TEST_STEPS'] if n_episodes is None else n_episodes):
                break
        total_imu/=ctx_steps
        logging.wikimyei_logging("tested, total mean imu : {} : total_steps : {}".format(total_imu,ctx_steps))
        return total_imu
    
    def _test_wikimyei_on_ahpa_(self):
        # This method rund on model.eval mode
        self.model.eval()
        early_stop = False
        test_imu = np.mean([self._run_wk_episode_(render_flag=False, n_episodes=None) for _ in range(self.wk_config['NUM_TESTS'])])
        logging.wikimyei_logging('[TEST wk: INFO] imu: %s' % (test_imu))
        if(cwcn_config.CWCN_OPTIONS.RENDER_FLAG):
            self._run_wk_episode_(render_flag=cwcn_config.CWCN_OPTIONS.RENDER_FLAG,n_episodes=self.wk_config['AHDO_STEPS'])
        if self.best_imu is None or self.best_imu < test_imu:
            if self.best_imu is not None:
                name = "%s_best_%+.3f.dat" % (self.wk_config['AHPA_ID'], test_imu)
                logging.wikimyei_logging("[TEST wk: INFO:] Best imu updated: %.3f -> %.3f : %s" % (self.best_imu, test_imu,name))
                fname = os.path.join(self.wk_config['CHECKPOINTS_FOLDER'], name)
                self._save_wikimyei_(fname)
            self.best_imu = test_imu
        self.model.train()
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