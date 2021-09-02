# Based on https://github.com/higgsfield/RL-Adventure-2/blob/master/3.ppo.ipynb
import argparse
import math
import os
import random
import gym
import numpy as np
import copy
import logging

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

from cwcn_uwaabo_piaabo import ActorCritic

from matplotlib import pyplot as plt

ENV_ID              = "MountainCarContinuous-v0"#"Pendulum-v0"#"MountainCarContinuous-v0"
LEARNING_RATE       = 3e-4
GAMMA               = 0.99
GAE_LAMBDA          = 0.95
PPO_EPSILON         = 0.2
REWARD_BETA         = 0.01
UWAABO_BETA         = 0.05
MUNAAJPI_BETA       = 0.01
IITEPI_BETA         = 0.01
ENTROPY_BETA        = 0.0001
PPO_STEPS           = 256
MINI_BATCH_SIZE     = 64
PPO_EPOCHS          = 16
TEST_EPOCHS         = 16
NUM_TESTS           = 3
TARGET_REWARD       = 9999999999999999999999999999999

LOSS_MAX            = 0.5
LOSS_MIN            =-0.5

PLOT_FLAG           =False
# ... #FIXME assert comulative munaajpi is in place
# --- --- --- ---
class CWCN_COLORS:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    REGULAR = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] :: %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
# --- --- --- --- 
def mkdir(base, name):
    path = os.path.join(base, name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path
def normalize(x):
    x -= x.mean()
    x /= (x.std() + 1e-8)
    return x
# --- --- --- ---
class WIKIMYEI_STATE:
    def __init__(self):
        self.c_state    = None
        self.accomulated_reward = None
class TRAYECTORY:
    def __init__(self):
        self.reward     = None
        self.done       = None
        self.mask       = None
        self.returns    = None
        self.state      = None
        self.log_prob   = None
        self.value      = None
        # self.dist       = None
        self.action     = None
        self.advantage  = None
        self.gae        = None
        self.delta      = None
        self.entropy    = None
        self.index      = None
class LEARNING_PROFILE:
    def __init__(self):
        # self.p_trayectory= None # profile reference to original trayectory
        self.ratio      = None
        self.surr1      = None
        self.surr2      = None
        self.uwaabo_loss = None
        self.munaajpi_loss= None
        self.loss       = None
        self.index      = None
        self.batch_size = None
# --- --- --- ---
class LOAD_QUEUE:
    def __init__(self):
        self.only_data=True #FIXME this makes the detachs
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._reset_queue_()
    def _reset_queue_(self):
        self.load_size=0
        self.load_index=0
        self.load_queue=[]
    def _detach_queue_item_(self,_item):
        for _k in _item.__dict__.keys():
            if(_k not in ['index','batch_size','p_trayectory'] and _item.__dict__[_k] != None):
                torch.detach(_item.__dict__[_k])
        return _item
    def _append_(self,_item):
        # for _k in (_item.__dict__.keys()):
        #     if(torch.is_tensor(_item.__dict__[_k])):
        #         aux=_item.__dict__[_k].clone()
        #     else:
        #         aux=copy.deepcopy(_item.__dict__[_k])
        _item.index=self.load_size
        if(self.only_data):
            self.load_queue.append(self._detach_queue_item_(_item)) #FIXME understand it
        else:
            self.load_queue.append(_item)
        self.load_size=len(self.load_queue)
        self._load_healt_()
    # def _to_tensor_(self,value_t):
    #     return torch.FloatTensor([value_t]).squeeze(0).to(self.device)
    def _import_queue_(self,_ipt_queue):
        for _iq in _ipt_queue:
            self._append_(_iq)
    def _dict_vectorize_queue_list_batch_(self, _batch_queue, _tensorize=False):
        r_dict={}
        assert(len(_batch_queue)>0)
        for _k in list(_batch_queue[0].__dict__.keys()):
            r_dict[_k]=[]
            for _c in range(len(_batch_queue)):
                if(_tensorize and _k not in ['index','batch_size','p_trayectory']):
                    tensor_aux=_batch_queue[_c].__dict__[_k]
                    assert(torch.is_tensor(tensor_aux))
                    if(len(tensor_aux.shape)==0):
                        r_dict[_k].append(tensor_aux.unsqueeze(0))
                    else:
                        r_dict[_k].append(tensor_aux)
                else:
                    r_dict[_k].append(_batch_queue[_c].__dict__[_k])
            if(_tensorize and _k not in ['index','batch_size','p_trayectory']):
                r_dict[_k]=torch.stack(r_dict[_k])
            elif(_k=='index'):
                r_dict[_k]=r_dict[_k][0]
        return r_dict
    def _itm_vect_(self,_itm,_type='default'):
        r_vect=[]
        for _c in range(self.load_size):
            r_vect.append(self.load_queue[_c].__dict__[_itm])
        if(_itm not in ['index','batch_size','p_trayectory']):
            if(_type=='default'):
                r_vect=r_vect
            elif(_type=='tensor'):
                r_vect=r_vect=torch.stack(r_vect)
            elif(_type=='array'):
                # print("------- ------- ------- ")
                # print(_type,_itm,r_vect)
                r_vect=np.array(np.hstack([r_.detach().numpy() for r_ in r_vect]))
            else:
                assert(False),"BAD _dict_vectorize_queue_ configuration"
        else:
            r_vect=r_vect[0]
        return r_vect
    def _dict_vectorize_queue_(self,_type='default'):
        r_dict={}
        assert(self.load_size>0)
        for _k in list(self.load_queue[self.load_index].__dict__.keys()):
            r_dict[_k]=self._itm_vect_(_k,_type)
        return r_dict
    def _load_vect_to_queue_(self,_key,_vect):
        aux_str="_load_vect_to_queue_ : key <{}> is not found to be on Queue".format(_key)
        assert(_key in list(self.load_queue[self.load_index].__dict__.keys())), aux_str
        aux_str="_load_vect_to_queue_ vector size <{}> does not match queue size <{}>".format(len(_vect),self.load_size)
        assert(len(_vect) == self.load_size), aux_str
        for _idx,_v in enumerate(_vect):
            self.load_queue[_idx].__dict__[_key]=_v
    def _load_normalize_(self,_opts:list,_type='default'):
        for _o in _opts:
            aux_vect=normalize(self._itm_vect_(_o,_type))
            self._load_vect_to_queue_(_o,aux_vect)
    def _load_healt_(self):
        healt_flag=True
        for _i,_t in enumerate(self.load_queue):
            if(_t.index!=_i):
                healt_flag&=False
                logging.warning("[_load_heal_] : %s load index : {} does not match load placement : {} %s".format(_t.index,_i) % (CWCN_COLORS.WARNING, CWCN_COLORS.REGULAR))
        return healt_flag
    def _plot_itm_(self,itm):
        # --- ---
        d_vects=self._dict_vectorize_queue_(_type='array')
        # --- ---
        self.fig, self.ax = plt.subplots(1, 1)
        self.fig.canvas.manager.full_screen_toggle()
        self.fig.patch.set_facecolor((0,0,0))
        self.ax.set_title("{} - {} - {}".format(self.ax.get_title(),ENV_ID,itm),color=(1,1,1))
        self.ax.set_facecolor((0,0,0))
        self.ax.tick_params(colors='white',which='both')
        self.ax.spines['bottom'].set_color('white')
        self.ax.spines['top'].set_color('white')
        self.ax.spines['right'].set_color('white')
        self.ax.spines['left'].set_color('white')
        self.ax.xaxis.label.set_color('white')
        self.ax.yaxis.label.set_color('white')
        # Plot data
        aux_twinx={}
        for _c,_i in enumerate(itm.split(',')):
            # aux_twinx[_i]=self.ax.twinx()
            # aux_twinx[_i].spines.right.set_position(("axes", 1.+0.5*_c))
            self.ax.plot(d_vects[_i], linewidth=0.3, label=_i)
            self.ax.set_ylabel(_i)
            self.ax.legend(itm.split(','))
            # print(_i,d_vects[_i].shape)
            # input()
        # --- ---
# --- --- --- ---
class WIKIMYEI:
    def __init__(self):
        # --- --- 
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # --- --- 
        self.env=gym.make(ENV_ID)
        self.num_inputs=self.env.observation_space.shape[0]
        try:
            self.num_outputs=self.env.action_space.shape[0]
        except:
            self.num_outputs=self.env.action_space.n
        # --- --- 
        self.model=ActorCritic(self.num_inputs, self.num_outputs).to(self.device)
        # --- --- 
        self.w_state=WIKIMYEI_STATE()
        # --- --- 
    def _transform_reward_(self, reward_v):
        if(ENV_ID=="MountainCarContinuous-v0"):
            # return self._to_tensor_(reward_v) + torch.dot(self._to_tensor_([2.0,0.1]),self._to_tensor_([self.w_state.c_state[0]+0.5,abs(self.w_state.c_state[1])])) + (0 if torch.round(self.w_state.c_state[0]*10**3)/(10**3)==0.5 else 100)
            return REWARD_BETA*(torch.dot(self._to_tensor_([2.0,10.0]),self._to_tensor_([self.w_state.c_state[0]+0.5,abs(self.w_state.c_state[1])])) \
                + (100 if torch.round(self.w_state.c_state[0]*10**1)/(10**1)==0.5 else 0) \
                - (0 if self.w_state.c_state[0]>-0.9 else -10))
        else:
            return self._to_tensor_(reward_v)
    def _dist_to_action_(self,dist_d,deterministic=True):
        # action = self.trayectory.dist.mean.detach().cpu().numpy()[0] if deterministic \
        #         else self.trayectory.dist.sample().cpu().numpy()[0]
        sample=dist_d.sample()
        if(ENV_ID=="Pendulum-v0"):
            return torch.multiply(sample,2.0), sample
        elif(ENV_ID=="MountainCarContinuous-v0"):
            return torch.multiply(sample,2.0), sample
        else:
            return sample, sample
    def _to_tensor_(self,value_t):
        return torch.FloatTensor([value_t]).squeeze(0).to(self.device)
    def _reset_(self):
        self.w_state.c_state=self._to_tensor_(self.env.reset())
        self.w_state.accomulated_reward=self._to_tensor_(0.0)
    def _wk_step_(self,__wt : TRAYECTORY = None):
        # --- ---
        if(self.w_state.c_state is None):
            self._reset_()
        if(__wt is None):
            __wt=TRAYECTORY()
        # --- ---
        dist, __wt.value=self.model(self.w_state.c_state)
        __wt.action,_s_aux=self._dist_to_action_(dist)
        # print("[size of sate:] {}, [action:] {}".format(self.w_state.c_state, __wt.action))
        __wt.entropy=dist.entropy().mean()
        __wt.log_prob=dist.log_prob(_s_aux)
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
        self._reset_()
        while True: # Test until is ready
            __wt=TRAYECTORY()
            done=self._wk_step_(__wt)
            total_reward+=__wt.reward
            if(render_flag):
                self.env.render()
            if(done):
                break
        return total_reward
# --- --- --- ---
class JKIMYEI_PPO:
    def __init__(self,_wikimyei):
        self.wikimyei=_wikimyei
        self.load_queue=None
        self.mini_batch_size=MINI_BATCH_SIZE
        self.optimizer=optim.Adam(_wikimyei.model.parameters(), lr=LEARNING_RATE)
        self.learning_queue=LOAD_QUEUE()
        # self.hist_learning_queue=LOAD_QUEUE()
        self.load_queue=LOAD_QUEUE()
    def _jkmimyei_gae_(self,gamma=GAMMA, lam=GAE_LAMBDA):
        assert(self.load_queue is not None), "Impossible to compute GAE, Jkimyei Queue found to be None"
        _, next_value = self.wikimyei.model(self.wikimyei.w_state.c_state)
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
        advantage = normalize(torch.cat(advantage))
        self.load_queue._load_vect_to_queue_('gae',gae_hist)
        self.load_queue._load_vect_to_queue_('delta',delta_hist)
        self.load_queue._load_vect_to_queue_('returns',returns)
        self.load_queue._load_vect_to_queue_('advantage',advantage)
    def _random_queue_yield_(self):
        for _ in range(self.load_queue.load_size // self.mini_batch_size):
            rand_ids = np.random.randint(0, self.load_queue.load_size, self.mini_batch_size)
            yield self.load_queue._dict_vectorize_queue_list_batch_([self.load_queue.load_queue[_i] for _i in rand_ids],True)
    def _jkimyei_ppo_update_(self,clip_param=PPO_EPSILON):
        # PPO EPOCHS is the number of times we will go through ALL the training data to make updates
        count_steps=0
        for _ in range(PPO_EPOCHS):
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
                dist, value=self.wikimyei.model(state)
                # print("[size of sate:] {}, [dist:] {}".format(state.shape, dist))
                entropy=dist.entropy().mean()
                _, _s_aux=self.wikimyei._dist_to_action_(dist)
                assert(torch.is_tensor(_s_aux))
                new_log_probs=dist.log_prob(_s_aux)
                # --- ---
                jk_profile.ratio=(new_log_probs - old_log_probs).exp()
                jk_profile.surr1=jk_profile.ratio * advantage
                jk_profile.surr2=torch.clamp(jk_profile.ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage
                # --- ---
                # IITEPI_BETA... #FIXME add iitepi as the exploration network
                jk_profile.uwaabo_loss =- UWAABO_BETA * torch.min(jk_profile.surr1, jk_profile.surr2).mean() #FIXME mean is tricky, util only due to pytorch great tools
                jk_profile.munaajpi_loss=MUNAAJPI_BETA * (returns - value).pow(2).mean() #FIXME mean is tricky to implement in c
                jk_profile.uwaabo_loss=torch.clamp(jk_profile.uwaabo_loss,min=LOSS_MIN,max=LOSS_MAX)
                jk_profile.munaajpi_loss=torch.clamp(jk_profile.munaajpi_loss,min=LOSS_MIN,max=LOSS_MAX)
                jk_profile.loss= jk_profile.munaajpi_loss + jk_profile.uwaabo_loss - ENTROPY_BETA * entropy
                # logging.info("uwaabo_loss: {}, \t munaajpi_loss: {}, \t loss: {}".format(jk_profile.uwaabo_loss.size(),jk_profile.munaajpi_loss.size(),jk_profile.loss.size()))
                # logging.info("uwaabo_loss: {:.4f}, \t munaajpi_loss: {:.4f}, \t loss: {:.4}".format(jk_profile.uwaabo_loss,jk_profile.munaajpi_loss, jk_profile.loss))
                # if(abs(jk_profile.uwaabo_loss)>=min(abs(LOSS_MAX),abs(LOSS_MIN)) or abs(jk_profile.munaajpi_loss)>=min(abs(LOSS_MAX),abs(LOSS_MIN))):
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
        assert(PPO_STEPS>=MINI_BATCH_SIZE)
        frame_idx  = 0
        train_epoch = 0
        self.best_reward = None
        self.early_stop = False
        # self.hist_learning_queue._reset_queue_()
        while not self.early_stop:
            # logging.info(" + + + [New jkimyei iteration]")
            self.learning_queue._reset_queue_()
            self.load_queue._reset_queue_()
            self.wikimyei._reset_()
            for _ in range(PPO_STEPS):
                # --- ---
                __wt=TRAYECTORY()
                done=self.wikimyei._wk_step_(__wt)
                # --- ---
                self.load_queue._append_(__wt)
                frame_idx += 1
                if(done):
                    if(self.load_queue.load_size<MINI_BATCH_SIZE):
                        self.mini_batch_size=self.load_queue.load_size
                    else:
                        self.mini_batch_size=MINI_BATCH_SIZE
                    break
            self.load_queue._load_normalize_(['reward'],'tensor')
            # if(>self.best_reward):
            # self.hist_learning_queue._import_queue_(self.learning_queue)
            self._jkmimyei_gae_()
            self._jkimyei_ppo_update_()
            # --- --- 
            train_epoch += 1
            if train_epoch % TEST_EPOCHS == 0:
                test_reward = np.mean([self.wikimyei._test_wikimyei_on_env_(render_flag=False) for _ in range(NUM_TESTS)])
                logging.info('Frame %s. reward: %s' % (frame_idx, test_reward))
                self.wikimyei._test_wikimyei_on_env_(render_flag=True)
                if self.best_reward is None or self.best_reward < test_reward:
                    if self.best_reward is not None:
                        name = "%s_best_%+.3f_%d.dat" % (ENV_ID, test_reward, frame_idx)
                        logging.info("Best reward updated: %.3f -> %.3f : %s" % (self.best_reward, test_reward,name))
                        fname = os.path.join('.', 'checkpoints', name)
                        torch.save(self.wikimyei.model.state_dict(), fname)
                    self.best_reward = test_reward
                if test_reward > TARGET_REWARD:
                    self.early_stop = True
            if PLOT_FLAG:
                self.load_queue._plot_itm_('action,reward,state')
                # self.learning_queue._plot_itm_('loss,munaajpi_loss,uwaabo_loss')
                # self.learning_queue._plot_itm_('ratio,surr1,surr2')
                self.load_queue._plot_itm_('returns,advantage,gae,delta')
                plt.show()

if __name__ == "__main__":
    mkdir('.', 'checkpoints')
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-n", "--name", default=ENV_ID, help="Name of the run")
    # args = parser.parse_args()
    c_wikimyei=WIKIMYEI()
    c_jkimyei=JKIMYEI_PPO(c_wikimyei)
    c_jkimyei._wikimyei_jkimyei_()
    