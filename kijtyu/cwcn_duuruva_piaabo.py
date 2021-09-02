# --- --- ---
import torch
from torch._C import Size
# --- --- ---
import cwcn_config
# --- --- ---
class DUURUVA:
    def __init__(self,_duuruva_vector_size : int,_wrapper_duuruva_normalize : bool=False):
        self._duuruva_vector_size=_duuruva_vector_size
        self._wrapper_duuruva_normalize=_wrapper_duuruva_normalize
        self._reset_duuruva_()
    def _reset_duuruva_(self):
        self._d_count=0
        self._duuruva=[]
        for _ in range(self._duuruva_vector_size):
            aux_d={}
            aux_d['value'] = 0
            aux_d['diff_1'] = 0
            aux_d['diff_2'] = 0
            aux_d['max'] = 0
            aux_d['min'] = 0
            aux_d['std'] = 0
            aux_d['mean'] = 0
            aux_d['M2'] = 0
            aux_d['M3'] = 0
            aux_d['M4'] = 0
            aux_d['kurtosis'] = 0
            aux_d['skewness'] = 0
            self._duuruva.append(aux_d)
    def _is_duuruva_ready_(self):
        return cwcn_config.CWCN_DUURUVA_CONFIG.READY_COUNT<=self._d_count
    def duuruva_value_wrapper(self,c_vect,_batch_size=None): #FIXME ugly method
        # batch_first if _batch_size != None
        assert(torch.is_tensor(c_vect)), "Duuruva is a tensor based method"
        struct_case=None
        if(_batch_size is not None):
            aux_str = "wrong batch size : {} != {}[0]".format(_batch_size,c_vect.Size())
            assert(_batch_size==c_vect.Size()[0]), aux_str
            assert(len(c_vect.size())==2), "Duuruva is only defined for vectors, and/or batches of vectors : batch first is asserted"
            v_size=c_vect.size()[1]
            struct_case=1
        else:
            if(len(c_vect.size())==1):
                assert(len(c_vect.size())==1), "Duuruva is only defined for vectors"
                v_size=c_vect.size()[0]
                struct_case=2
            elif(len(c_vect.size())==0):
                v_size=1
                struct_case=3
            else:
                assert(len(c_vect.size())==1), "Duuruva is confused about input size"
        for _b in range(_batch_size if _batch_size is not None else 1):
            self._d_count+=1
            _n = torch.Tensor([min(self._d_count,cwcn_config.CWCN_DUURUVA_CONFIG.DUURUVA_MAX_COUNT)]).squeeze(0)
            for _v in range(v_size):
                if(struct_case==1):
                    c_value = c_vect[_b][_v]
                elif(struct_case==2):
                    c_value = c_vect[_v]
                elif(struct_case==3):
                    c_value = c_vect
                self._duuruva[_v]['value']=c_value
                self._duuruva[_v]['max']=max(self._duuruva[_v]['max'], self._duuruva[_v]['value'])
                self._duuruva[_v]['min']=min(self._duuruva[_v]['min'], self._duuruva[_v]['value'])
                _delta = self._duuruva[_v]['value'] - self._duuruva[_v]['mean']
                _delta_n = _delta/_n
                _delta_n2 = _delta_n*_delta_n
                _term1 = _delta*_delta_n*(_n-1)
                self._duuruva[_v]['mean'] += _delta_n
                self._duuruva[_v]['M4'] += _term1*_delta_n2*(_n*_n-3*_n+3)+6*_delta_n2*self._duuruva[_v]['M2']-4*_delta_n*self._duuruva[_v]['M3']
                self._duuruva[_v]['M3'] += _term1*_delta_n*(_n-2)-3*_delta_n*self._duuruva[_v]['M2']
                self._duuruva[_v]['M2'] += _term1
                self._duuruva[_v]['std'] = self._duuruva[_v]['M2']/(_n-1)
                self._duuruva[_v]['kurtosis'] = (_n*self._duuruva[_v]['M4'])/(self._duuruva[_v]['M2']*self._duuruva[_v]['M2'])-3
                self._duuruva[_v]['skewness'] = torch.sqrt(_n)*self._duuruva[_v]['M3']/(torch.pow(self._duuruva[_v]['M2'],3)*torch.sqrt(self._duuruva[_v]['M2'])) #FIXME check if is right
                if(self._wrapper_duuruva_normalize):
                    c_normal = (c_value - self._duuruva[_v]['mean'])/(self._duuruva[_v]['std'] + cwcn_config.CWCN_DUURUVA_CONFIG.MIN_STD)
                    if(struct_case==1):
                        if(self._is_duuruva_ready_()):
                            c_vect[_b][_v] = c_normal
                        else:
                            c_vect[_b][_v] = torch.Tensor([0]).squeeze(0)
                    elif(struct_case==2):
                        if(self._is_duuruva_ready_()):
                            c_vect[_v] = c_normal
                        else:
                            c_vect[_v] = torch.Tensor([0]).squeeze(0)
                    elif(struct_case==3):
                        if(self._is_duuruva_ready_()):
                            c_vect = c_normal
                        else:
                            c_vect = torch.Tensor([0]).squeeze(0)
        return c_vect