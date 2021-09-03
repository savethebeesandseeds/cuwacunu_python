# --- --- --- --- 
import argparse
import math
import random
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import gym
# --- --- --- ---
from matplotlib import pyplot as plt
# --- --- --- ---
import os
import sys
import logging
# --- --- --- ---
import cwcn_config
import cwcn_wikimyei_nebajke
import cwcn_tsinuu_piaabo
import cwcn_kemu_piaabo
import cwcn_duuruva_piaabo
# --- --- --- --- 
class WIKIMYEI_STATE:
    def __init__(self,_wikimyei_config):
        # --- --- 
        self.c_config   = _wikimyei_config
        # --- --- 
        self.c_alliu    = None
        self.c_imu      = None
        self.accomulated_imu = None
        # --- --- 
        if(cwcn_config.CWCN_DUURUVA_CONFIG.ENABLE_DUURUVA_IMU):
            self.imu_duuruva = cwcn_duuruva_piaabo.DUURUVA(
                _duuruva_vector_size=self.wk_config['IMU_COUNT'],
                _wrapper_duuruva_normalize=cwcn_config.CWCN_DUURUVA_CONFIG.NORMALIZE_IMU)
        # --- --- 
        
class TRAYECTORY:
    def __init__(self):
        self.imu        = None
        self.done       = None
        self.mask       = None
        self.returns    = None
        self.alliu      = None
        self.log_prob   = None
        self.value      = None
        # self.dist       = None
        self.tsane     = None
        self.advantage  = None
        self.gae        = None
        self.delta      = None
        self.entropy    = None
        self.index      = None
# --- --- --- --- 
class LOAD_QUEUE: #FIXME load can be better
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
            aux_vect=cwcn_kemu_piaabo.kemu_normalize(self._itm_vect_(_o,_type))
            self._load_vect_to_queue_(_o,aux_vect)
    def _load_healt_(self):
        healt_flag=True
        for _i,_t in enumerate(self.load_queue):
            if(_t.index!=_i):
                healt_flag&=False
                logging.warning("[_load_heal_] : %s load index : {} does not match load placement : {} %s".format(_t.index,_i) % (cwcn_config.CWCN_COLORS.WARNING, cwcn_config.CWCN_COLORS.REGULAR))
        return healt_flag
    def _plot_itm_(self,itm):
        # --- ---
        d_vects=self._dict_vectorize_queue_(_type='array')
        # --- ---
        self.fig, self.ax = plt.subplots(1, 1)
        self.fig.canvas.manager.full_screen_toggle()
        self.fig.patch.set_facecolor((0,0,0))
        self.ax.set_title("{} - {} - {}".format(self.ax.get_title(),cwcn_config.CWCN_CONFIG().ENV_ID,itm),color=(1,1,1))
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
