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
# --- --- --- ---
# --- --- --- ---
class WIKIMYEI:
    def __init__(self,_config,_load_file=None):
        # --- --- 
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # --- --- 
        logging.info("[WIKIMYEI:] building with config: {}".format(_config))
        self.env=gym.make(_config['ENV_ID'])
        self.reward_size=1
        self.num_inputs=self.env.observation_space.shape[0]
        try:
            self.num_outputs=self.env.action_space.shape[0]
        except:
            self.num_outputs=self.env.action_space.n
        # --- --- 
        self.wk_config=_config
        self.wk_config['NUM_INPUTS']=self.num_inputs
        self.wk_config['NUM_OUTPUTS']=self.num_outputs
        # --- --- 
        self.model=cwcn_tsinuu_piaabo.TSINUU_ACTOR_CRITIC(
            alliu_size=self.wk_config['NUM_INPUTS'], 
            uwaabo_size=self.wk_config['NUM_OUTPUTS'], 
            UWAABO_HIDDEN_SIZE=self.wk_config['UWAABO_HIDDEN_SIZE'],
            MUNAAJPI_HIDDEN_SIZE=self.wk_config['MUNAAJPI_HIDDEN_SIZE'],
            sigma=1.0)
        # --- --- 
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                self.model = nn.DataParallel(self.model)
        self.model.to(self.device)
        # --- --- 
        self.w_state=cwcn_wikimyei_piaabo.WIKIMYEI_STATE(
            state_size=self.num_inputs,
            action_size=self.num_outputs,
            reward_size=self.reward_size)
        # --- --- 
        self.jkimyei=cwcn_jkimyei_nebajke.JKIMYEI_PPO(self)
        # --- --- 
        if(_load_file is not None and os.path.isfile(_load_file)):
            self._load_wikimyei_(_load_file)
        # --- --- 
    def _save_wikimyei_(self,__path): # add model config
        logging.info("saving model to file <{}>".format(__path))
        torch.save((self.model.state_dict(), self.jkimyei.optimizer.state_dict(), self.wk_config), __path)
    def _load_wikimyei_(self,__path):
        logging.info("loading model from file <{}>".format(__path))
        model_state, optimizer_state, config_state = torch.load(__path)
        self.model.load_state_dict(model_state)
        self.jkimyei.optimizer.load_state_dict(optimizer_state)
        self.wk_config = config_state
    def _transform_reward_(self, reward_v):
        if(self.wk_config['ENV_ID']=="MountainCarContinuous-v0"):
            # return self._to_tensor_(reward_v) + torch.dot(self._to_tensor_([2.0,0.1]),self._to_tensor_([self.w_state.c_state[0]+0.5,abs(self.w_state.c_state[1])])) + (0 if torch.round(self.w_state.c_state[0]*10**3)/(10**3)==0.5 else 100)
            c_reward=self.wk_config['REWARD_BETA']*(torch.dot(self._to_tensor_([10.0,10.0]),self._to_tensor_([self.w_state.c_state[0]+0.5,abs(self.w_state.c_state[1])])) \
                + (200 if torch.round(self.w_state.c_state[0]*10**1)/(10**1)==0.5 else 0) \
                + (10 if self.w_state.c_state[0]>-.2 else 0) \
                + (20 if self.w_state.c_state[0]>+.0 else 0) \
                + (30 if self.w_state.c_state[0]>+.2 else 0) \
                + (40 if self.w_state.c_state[0]>+.4 else 0) \
                + (0 if self.w_state.c_state[0]>-1.1 else -100))
        else:
            c_reward=self._to_tensor_(reward_v)
        return self.w_state.reward_duuruva.duuruva_value_wrapper(c_reward) if cwcn_config.CWCN_DUURUVA_CONFIG.ENABLE_DUURUVA_REWARD else c_reward
    def _dist_to_action_(self,dist_d,deterministic=True):
        # action = self.trayectory.dist.mean.detach().cpu().numpy()[0] if deterministic \
        #         else self.trayectory.dist.sample().cpu().numpy()[0]
        sample=dist_d.sample()
        assert(torch.is_tensor(sample))
        entropy=dist_d.entropy().mean()
        log_probs=dist_d.log_prob(sample)
        if(torch.any(torch.isnan(sample)) or torch.any(torch.isnan(log_probs))):
            logging.info("[nan case] sample:{}, log_probs:{} entropy:{}".format(sample, log_probs, entropy))
        if(self.wk_config['ENV_ID']=="Pendulum-v0"):
            return torch.multiply(sample,2.0), sample, log_probs, entropy
        elif(self.wk_config['ENV_ID']=="MountainCarContinuous-v0"):
            return torch.multiply(sample,2.0), sample, log_probs, entropy
        else:
            return sample, sample, log_probs, entropy # Returns [action, sample, logprob, entropy]
    def _to_tensor_(self,value_t):
        return torch.FloatTensor([value_t]).squeeze(0).to(self.device)
    def _reset_(self):
        self.w_state.c_state=self._to_tensor_(self.env.reset())
        self.w_state.accomulated_reward=self._to_tensor_(0.0)
    def _wk_step_(self,__wt : cwcn_wikimyei_piaabo.TRAYECTORY = None):
        # --- ---
        if(self.w_state.c_state is None):
            self._reset_()
        if(__wt is None):
            __wt=cwcn_wikimyei_piaabo.TRAYECTORY()
        # --- ---
        dist, __wt.value=self.model(self.w_state.c_state)
        __wt.action, _, __wt.log_prob, __wt.entropy=self._dist_to_action_(dist)
        # print("[size of sate:] {}, [action:] {}".format(self.w_state.c_state, __wt.action))
        # --- ---
        c_state, c_reward, c_done, _=self.env.step(__wt.action.cpu().numpy())
        # logging.info("Reward : {}, State : {}".format(reward,next_state))
        # --- ---
        self.w_state.c_state=self._to_tensor_(c_state)
        __wt.state=self._to_tensor_(c_state)
        # --- ---
        if(False): #FIXME find more about accolulated reward
            self.w_state.accomulated_reward+=self._transform_reward_(c_reward)
            __wt.reward=self.w_state.accomulated_reward
        else:
            __wt.reward=self._transform_reward_(c_reward)
            self.w_state.accomulated_reward+=__wt.reward
        # logging.info("Reward : {}, Transformed_Reward: {}".format(c_reward, __wt.reward))
        __wt.mask=self._to_tensor_(1 - bool(c_done))
        __wt.done=self._to_tensor_(c_done)
        # logging.info("[STATE:] {}, [ACTION:] {}, [DONE:] {}".format(__wt.state,__wt.action,__wt.done))
        return c_done
    def _test_wikimyei_on_env_(self,render_flag=False):
        total_reward=self._to_tensor_(0.)
        ctx_steps=0
        self._reset_()
        while True: # Test until is ready
            ctx_steps+=1
            __wt=cwcn_wikimyei_piaabo.TRAYECTORY()
            done=self._wk_step_(__wt)
            total_reward+=__wt.reward
            if(render_flag):
                self.env.render()
            if(done):
                break
        total_reward/=ctx_steps
        logging.info("[WIKIMYEI:] tested, total mean reward : {}".format(total_reward))
        return total_reward
# --- --- --- ---