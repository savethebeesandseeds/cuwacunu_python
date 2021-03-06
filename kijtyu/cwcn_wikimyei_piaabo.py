# --- --- --- --- 
# cwcn_wikimyei_piaabo
# --- --- --- --- 
import numpy as np
import torch
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
        if(cwcn_config.CWCN_DUURUVA_CONFIG.ENABLE_DUURUVA_IMU):
            self.imu_duuruva = cwcn_duuruva_piaabo.DUURUVA(
                _duuruva_vector_size=self.wk_config['IMU_COUNT'],
                _wrapper_duuruva_normalize=cwcn_config.CWCN_DUURUVA_CONFIG.NORMALIZE_IMU)
        # --- --- 
        self._reset_()
        # --- --- 
    def _reset_(self):
        self.c_alliu = None
        self.c_imu=None
        self.accomulated_imu=cwcn_kemu_piaabo.kemu_to_tensor(0.0)
# --- --- --- ---         
class AHDO_PROFILE: # the trayectory is a single step
    def __init__(self):
        # --- --- 
        self.imu        = None
        self.done       = None
        self.mask       = None
        self.returns    = None
        self.alliu      = None
        self.log_prob   = None
        self.value      = None
        self.price      = None
        # self.dist       = None
        self.tsane      = None
        self.advantage  = None
        self.gae        = None
        self.delta      = None
        self.entropy    = None
        self.index      = None
        # --- --- 
        self.selec_prob = None
        # --- --- 
        if(cwcn_config.PAPER_INSTRUMENT and cwcn_config.TRAIN_ON_FORECAST):
            self.forecast_uwaabo_vect = None # is a vector, or array, or list, not sure #FIXME?
            self.forecast_non_uwaabo_vect = None # is a vector, or array, or list, not sure #FIXME?
            # self.tsane_non_uwaabo = {}
        # --- --- 
        self.certainty=None # vector [put/pass/call] or as declared in cwcn_config.UJCAMEI....TSANE_ACTION_DICT
        self.put_certainty=None
        self.pass_certainty=None
        self.call_certainty=None
        # --- --- 
# --- --- --- --- 
class LEARNING_LOAD_QUEUE: #FIXME load can be better
    def __init__(self):
        self._reset_queue_()
    # --- --- 
    def _reset_queue_(self):
        self.load_size=0
        self.load_index=0
        self.load_queue=[]
    # --- --- 
    def _append_(self,_item,_detach_flag):
        _item.index=self.load_size
        if(_detach_flag):
            self.load_queue.append(self._detach_queue_item_(_item)) #FIXME understand it
        else:
            self.load_queue.append(_item)
        self.load_size=len(self.load_queue)
        assert(self._load_healt_()), "Unhealty LEARNING_LOAD_QUEUE"
    # --- --- 
    def _detach_queue_item_(self,_item):
        for _k in _item.__dict__.keys():
            if(_k not in ['index','batch_size','profile'] and _item.__dict__[_k] != None):
                torch.detach(_item.__dict__[_k])
        return _item
    # --- ---
    def _queue_itm_to_vect_(self,_itm,_type='default'):
        r_vect=[]
        for _c in range(self.load_size):
            r_vect.append(self.load_queue[_c].__dict__[_itm])
        if(_itm not in ['index','batch_size','selec_prob','p_selec_prob']):
            if(_type=='default'):   r_vect=r_vect
            elif(_type=='tensor'):  r_vect=r_vect=torch.stack(r_vect)
            elif(_type=='array'):   r_vect=np.array(np.hstack([r_.detach().numpy() for r_ in r_vect]))
            else:
                assert(False),"BAD _dict_vectorize_queue_ configuration"
        else: 
            r_vect=r_vect[0]
        return r_vect
    def _dict_vectorize_queue_(self,_type='default'):
        r_dict={}
        assert(self.load_size>0)
        for _k in list(self.load_queue[self.load_index].__dict__.keys()):
            r_dict[_k]=self._queue_itm_to_vect_(_k,_type)
        return r_dict
    # --- --- 
    def _load_healt_(self):
        healt_flag=True
        for _i,_t in enumerate(self.load_queue):
            if(_t.index!=_i):
                healt_flag&=False
                logging.warning("[_ll_load_heal_] : %s load index : {} does not match load placement : {} %s".format(_t.index,_i) % (cwcn_config.CWCN_COLORS.WARNING, cwcn_config.CWCN_COLORS.REGULAR))
        return healt_flag
    def _plot_itm_(self,itm):
        cwcn_kemu_piaabo.kemu_plot_queue_item(self,itm)
# --- --- --- --- 
class AHDO_LOAD_QUEUE: #FIXME load can be better
    # adho is the place, the whiakaju of things.
    def __init__(self):
        self._reset_queue_()
    # --- --- 
    def _reset_queue_(self):
        self.load_size=0
        self.load_index=0
        self.load_queue=[]
    # --- --- 
    def _append_(self,_item,_detach_flag):
        # for _k in (_item.__dict__.keys()):
        #     if(torch.is_tensor(_item.__dict__[_k])):
        #         aux=_item.__dict__[_k].clone()
        #     else:
        #         aux=copy.deepcopy(_item.__dict__[_k])
        _item.index=self.load_size
        if(_detach_flag):
            self.load_queue.append(self._detach_queue_item_(_item)) #FIXME understand it
        else:
            self.load_queue.append(_item)
        self.load_size=len(self.load_queue)
        assert(self._load_healt_()), "Unhealty AHDO_LOAD_QUEUE"
    def _import_queue_(self,_ipt_queue,_detach_flag):
        for _iq in _ipt_queue:
            self._append_(_iq,_detach_flag)
    # --- ---
    def _standarize_select_prob_(self,_fact='selec_prob'):
        p_sum=sum([_t.__dict__[_fact] for _t in self.load_queue])
        for _t in self.load_queue:
            _t.__dict__['p_selec_prob']=_t.__dict__[_fact]/p_sum
    # --- ---
    def _mini_batch_size_(self):
        return self.load_size \
            if(self.load_size<cwcn_config.CWCN_CONFIG().MINI_BATCH_COUNT) \
                else cwcn_config.CWCN_CONFIG().MINI_BATCH_COUNT
    # --- ---
    def _random_queue_yield_by_prob_(self,_dict_vectorize_flag,_tensorize_flag):
        c_batch_size=self._mini_batch_size_()
        for _ in range(self.load_size // c_batch_size):
            load_index=list(range(0, self.load_size))
            load_probs=list([_l_itm.__dict__['p_selec_prob'] for _l_itm in self.load_queue])
            rand_ids = np.random.choice(a=load_index,p=load_probs,size=c_batch_size) # item probability
            if(_dict_vectorize_flag):
                yield self._dict_vectorize_queue_list_batch_([self.load_queue[_i] for _i in rand_ids],_tensorize_flag)
            else:
                yield list([self.load_queue[_i] for _i in rand_ids])
    def _random_queue_yield_uniform_(self,_dict_vectorize_flag,_tensorize_flag):
        c_batch_size=self._mini_batch_size_()
        for _ in range(self.load_size // c_batch_size):
            rand_ids = np.random.randint(0, self, c_batch_size) # uniform
            if(_dict_vectorize_flag):
                yield self._dict_vectorize_queue_list_batch_([self.load_queue[_i] for _i in rand_ids],_tensorize_flag)
            else:
                yield list([self.load_queue[_i] for _i in rand_ids])
    # --- ---
    def _detach_queue_item_(self,_item):
        for _k in _item.__dict__.keys():
            if(_k not in ['index','batch_size','selec_prob','p_selec_prob'] and _item.__dict__[_k] != None):
                torch.detach(_item.__dict__[_k])
        return _item
    def _dict_vectorize_queue_list_batch_(self, _batch_queue, _tensorize_flag=False):
        r_dict={}
        assert(len(_batch_queue)>0)
        for _k in list(_batch_queue[0].__dict__.keys()):
            r_dict[_k]=[]
            for _c in range(len(_batch_queue)):
                if(_tensorize_flag and _k not in ['index','batch_size','selec_prob','p_selec_prob']):
                    tensor_aux=_batch_queue[_c].__dict__[_k]
                    assert(torch.is_tensor(tensor_aux))
                    if(len(tensor_aux.shape)==0):tensor_aux=tensor_aux.unsqueeze(0)
                    r_dict[_k].append(tensor_aux)
                else:
                    r_dict[_k].append(_batch_queue[_c].__dict__[_k])
            if(_tensorize_flag and _k not in ['index','batch_size','selec_prob','p_selec_prob']):
                r_dict[_k]=torch.stack(r_dict[_k])
            elif(_k=='index'):
                r_dict[_k]=r_dict[_k][0]
        return r_dict
    # --- --- 
    def _queue_itm_to_vect_(self,_itm,_type='default'):
        r_vect=[]
        for _c in range(self.load_size):
            r_vect.append(self.load_queue[_c].__dict__[_itm])
        if(_itm not in ['index','batch_size','selec_prob','p_selec_prob']):
            if(_type=='default'):   r_vect=r_vect
            elif(_type=='tensor'):  r_vect=r_vect=torch.stack(r_vect)
            elif(_type=='array'):   r_vect=np.array(np.hstack([r_.detach().numpy() for r_ in r_vect]))
            else:
                assert(False),"BAD _dict_vectorize_queue_ configuration"
        else: 
            r_vect=r_vect[0]
        return r_vect
    def _dict_vectorize_queue_(self,_type='default'):
        r_dict={}
        assert(self.load_size>0)
        for _k in list(self.load_queue[self.load_index].__dict__.keys()):
            r_dict[_k]=self._queue_itm_to_vect_(_k,_type)
        return r_dict
    def _load_vect_to_queue_(self,_key,_vect):
        aux_str="_load_vect_to_queue_ : key <{}> is not found to be on Ahdo Queue".format(_key)
        assert(_key in list(self.load_queue[self.load_index].__dict__.keys())), aux_str
        aux_str="_load_vect_to_queue_ vector size <{}> does not match Ahdo Queue size <{}>".format(len(_vect),self.load_size)
        assert(len(_vect) == self.load_size), aux_str
        for _idx,_v in enumerate(_vect):
            self.load_queue[_idx].__dict__[_key]=_v
    def _load_healt_(self):
        healt_flag=True
        for _i,_t in enumerate(self.load_queue):
            if(_t.index!=_i):
                healt_flag&=False
                logging.warning("[_al_load_heal_] : %s load index : {} does not match load placement : {} %s".format(_t.index,_i) % (cwcn_config.CWCN_COLORS.WARNING, cwcn_config.CWCN_COLORS.REGULAR))
        return healt_flag
    def _load_normalize_(self,_opts:list,_type='default'):
        for _o in _opts:
            aux_vect=cwcn_kemu_piaabo.kemu_normalize(self._queue_itm_to_vect_(_o,_type))
            self._load_vect_to_queue_(_o,aux_vect)
    def _plot_itm_(self,itm):
        cwcn_kemu_piaabo.kemu_plot_queue_item(self,itm)
# --- --- --- --- 