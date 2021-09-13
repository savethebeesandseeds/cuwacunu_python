# --- --- --- 
import torch
import numpy as np
import logging
# --- --- --- 
import cwcn_config
import cwcn_wikimyei_piaabo
import cwcn_wikimyei_nebajke
import cwcn_kemu_piaabo
# --- --- ---
# --- --- --- --- 
class LEARNING_PROFILE:
    def __init__(self):
        self.ratio              = None
        self.surr1              = None
        self.surr2              = None
        self.uwaabo_imibajcho   = None
        self.munaajpi_imibajcho = None
        self.imibajcho          = None
        self.index              = None
        self.batch_size         = None
        # --- --- 
        self.selec_prob         = None
# --- --- --- ---
# --- --- --- ---         
class HYPER_PROFILE_QUEUE:
    def __init__(self,buffer_size):
        # --- --- 
        assert(buffer_size>0), "configuration problem, buffer size for HYPER_PROFILE_QUEUE too short"
        # --- --- 
        self.buffer_size = buffer_size
        # --- ---
        self._reset_queue_() 
    # --- --- 
    def _reset_queue_(self):
        self.load_size=0
        self.load_index=0
        self.load_queue=[]
    # --- --- 
    def _hyper_append_(self,_item,_total_imu,_selec_prob):
        if(torch.torch.is_tensor(_total_imu)):
            _item.total_imu = _total_imu.detach().numpy() # not normalizable
        else:
            _item.total_imu = _total_imu # not normalizable
        if(torch.torch.is_tensor(_selec_prob)):
            _item.selec_prob = _selec_prob.detach().numpy()
        else:
            _item.selec_prob = _selec_prob
        _item.index=self.load_size
        self.load_queue.append(_item)
        self._filter_buffer_()
        self.load_size=len(self.load_queue)
        assert(self._hyper_load_healt_()), "Unhealty HYPER_PROFILE_QUEUE"
    # --- ---
    def _filter_buffer_(self):
        if(len(self.load_queue)>self.buffer_size):
            # allow the last to remain always
            if(self.buffer_size>1):
                self.load_queue=sorted(self.load_queue, reverse=True ,key=lambda x:x.__dict__['total_imu'])[:self.buffer_size-1]+self.load_queue[-1:]
            else:
                self.load_queue=self.load_queue[-1:]
    # --- ---
    def _standarize_select_prob_(self,_fact='selec_prob'):
        p_sum=sum([_t.__dict__[_fact] for _t in self.load_queue])
        for _t in self.load_queue:
            _t.__dict__['p_selec_prob']=_t.__dict__[_fact]/p_sum
    # --- ---
    def _random_queue_yield_by_prob_(self,_yield_count):
        for _ in range(_yield_count):
            load_index=list(range(0, self.load_size))
            load_probs=list([_l_itm.__dict__['p_selec_prob'] for _l_itm in self.load_queue])
            rand_idx = np.random.choice(a=load_index,p=load_probs) # item probability
            yield self.load_queue[rand_idx]
    def _random_queue_yield_uniform_(self,_yield_count):
        for _ in range(_yield_count):
            rand_idx = np.random.randint(0, self) # uniform
            yield self.load_queue[rand_idx]
    # --- --- 
    def _hyper_load_healt_(self):
        healt_flag=True
        if(len(self.load_queue)>self.buffer_size):
            healt_flag&=False
            logging.warning("[_ph_load_heal_] : %s load size : {} higher than buffer size : {} %s".format(self.load_size,self.buffer_size) % (cwcn_config.CWCN_COLORS.WARNING, cwcn_config.CWCN_COLORS.REGULAR))
        return healt_flag
    def _plot_itm_(self,itm):
        cwcn_kemu_piaabo.kemu_plot_queue_item(self,itm)