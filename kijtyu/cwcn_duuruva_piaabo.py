# --- --- ---
# cwcn_duuruva_piaabo.py
# --- --- ---
# a mayor TEHDUJCO to python fundation
# --- --- ---
# a mayor TEHDUJCO to the torch fundation
# --- --- ---
import torch
import logging
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
# --- --- ---
import cwcn_config
# --- --- ---
class DUURUVA:
    def __init__(self,_duuruva_vector_size : int,_wrapper_duuruva_normalize, _d_name : str=None):
        self._d_name=_d_name
        self._wrapper_duuruva_std_or_norm=_wrapper_duuruva_normalize
        self._duuruva_cosas = [
            'value',
            'max',
            'min',
            'mean',
            'M4',
            'M3',
            'M2',
            'variance',
            'kurtosis',
            'skewness',
        ]
        self._duuruva_vector_size=_duuruva_vector_size
        self._reset_duuruva_()
        self._plot_instruction = cwcn_config.CWCN_DUURUVA_CONFIG.PLOT_LEVEL
        self._plot_holder = dict([(_dc,[]) for _dc in self._duuruva_cosas])
    def _reset_duuruva_(self):
        self._d_count=0
        self._duuruva=[]
        for _ in range(self._duuruva_vector_size):
            aux_d={}
            aux_d['value'] = torch.Tensor([0]).squeeze(0).to(cwcn_config.device)
            aux_d['diff_1'] = torch.Tensor([0]).squeeze(0).to(cwcn_config.device)
            aux_d['diff_2'] = torch.Tensor([0]).squeeze(0).to(cwcn_config.device)
            aux_d['max'] = torch.Tensor([0]).squeeze(0).to(cwcn_config.device)
            aux_d['min'] = torch.Tensor([0]).squeeze(0).to(cwcn_config.device)
            aux_d['variance'] = torch.Tensor([0]).squeeze(0).to(cwcn_config.device)
            aux_d['mean'] = torch.Tensor([0]).squeeze(0).to(cwcn_config.device)
            aux_d['M2'] = torch.Tensor([0]).squeeze(0).to(cwcn_config.device)
            aux_d['M3'] = torch.Tensor([0]).squeeze(0).to(cwcn_config.device)
            aux_d['M4'] = torch.Tensor([0]).squeeze(0).to(cwcn_config.device)
            aux_d['kurtosis'] = torch.Tensor([0]).squeeze(0).to(cwcn_config.device)
            aux_d['skewness'] = torch.Tensor([0]).squeeze(0).to(cwcn_config.device)
            self._duuruva.append(aux_d)
    def _is_duuruva_ready_(self):
        return cwcn_config.CWCN_DUURUVA_CONFIG.DUURUVA_READY_COUNT<=self._d_count
    def _duuruva_value_wrapper_(self,c_vect,_batch_size=None): #FIXME ugly method
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
                    c_value = c_vect[_b][_v].detach().clone()
                elif(struct_case==2):
                    c_value = c_vect[_v].detach().clone()
                elif(struct_case==3):
                    c_value = c_vect.detach().clone()
                # --- --- --- --- --- --- --- --- --- --- a mayor TEHDUJCO to the WIKI
                # --- --- --- --- https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
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
                self._duuruva[_v]['variance'] = self._duuruva[_v]['M2']/(_n-1)
                self._duuruva[_v]['kurtosis'] = (_n*self._duuruva[_v]['M4'])/(self._duuruva[_v]['M2']*self._duuruva[_v]['M2'])-3
                self._duuruva[_v]['skewness'] = torch.sqrt(_n)*self._duuruva[_v]['M3']/(torch.pow(self._duuruva[_v]['M2'],3)*torch.sqrt(self._duuruva[_v]['M2'])) #FIXME check if is right
                if('norm'):
                    c_standar = (c_value - self._duuruva[_v]['mean'])/(torch.sqrt(self._duuruva[_v]['variance']) + cwcn_config.CWCN_DUURUVA_CONFIG.MIN_STD)
                elif('std'):
                    c_standar = (c_value)/(torch.sqrt(self._duuruva[_v]['variance']) + cwcn_config.CWCN_DUURUVA_CONFIG.MIN_STD)
                elif('not'):
                    c_standar = c_value
                else:
                    assert(False), "wrong wrapper_duuruva_std_or_norm configuration"
                # --- --- --- --- --- 
                if(struct_case==1):
                    if(self._is_duuruva_ready_()):
                        c_vect[_b][_v] = c_standar
                    else:
                        c_vect[_b][_v] = torch.Tensor([0]).squeeze(0)
                elif(struct_case==2):
                    if(self._is_duuruva_ready_()):
                        c_vect[_v] = c_standar
                    else:
                        c_vect[_v] = torch.Tensor([0]).squeeze(0)
                elif(struct_case==3):
                    if(self._is_duuruva_ready_()):
                        c_vect = c_standar
                    else:
                        c_vect = torch.Tensor([0]).squeeze(0)
                # logging.deep_logging("waka {}:struct_case:{} / is_duuruva_ready: {} / mean:{} / variance:{} / normal:{} / standar:{}".format(self._d_name,struct_case,self._is_duuruva_ready_(),self._duuruva[_v]['mean'],self._duuruva[_v]['variance'],c_value,c_standar))
                if(self._plot_instruction is not None):
                    for _dc in self._duuruva_cosas:
                        self._plot_holder[_dc].append(self._duuruva[_v][_dc].clone())
        return c_vect
    def _plot_duuruva_(self):
        class Labeloffset():
            def __init__(self,  ax, label="", axis="y"):
                self.axis = {"y":ax.yaxis, "x":ax.xaxis}[axis]
                self.label=label
                ax.callbacks.connect(axis+'lim_changed', self.update)
                ax.figure.canvas.draw()
                self.update(None)
            def update(self, lim):
                fmt = self.axis.get_major_formatter()
                self.axis.offsetText.set_visible(False)
                self.axis.set_label_text(self.label + " "+ fmt.get_offset() )
        # --- ---
        color_pallete=['red','white','yellow','blue']
        # --- ---
        fig, ax = plt.subplots(1, 1)
        ax.set_title("{}".format(self._d_name),color=(1,1,1),**{'fontname':'DejaVu Sans'})
        # Plot data
        if self._plot_instruction.lower()=='all': 
            c_yielder=(_pl for _pl in self._duuruva_cosas)
            ax.legend(self._duuruva_cosas)
            ax.set_title("{} // {}".format(ax.get_title(),self._duuruva_cosas),color=(1,1,1),**{'fontname':'DejaVu Sans'})
        else:
            c_yielder=(_pl for _pl in self._plot_instruction.split(','))
            ax.legend(self._plot_instruction)
            ax.set_title("{} // {}".format(ax.get_title(),self._plot_instruction),color=(1,1,1),**{'fontname':'DejaVu Sans'})
        ax.yaxis.offsetText.set_visible(False)
        ax.set_facecolor((0,0,0))
        ax.xaxis.label.set_color('white')
        ax.spines['bottom'].set_color('white')
        ax.spines['top'].set_color('white')
        ax.spines['right'].set_color('white')
        ax.spines['left'].set_color('white')
        aux_twinx=[ax.twinx()]
        for _c,_i in enumerate(c_yielder):
            # --- --- 
            d_vect=self._plot_holder[_i]
            # --- --- 
            # aux_twinx.append(ax)
            aux_twinx[_c].yaxis.offsetText.set_visible(False)
            aux_twinx[_c].yaxis.label.set_color(color_pallete[_c])
            aux_twinx[_c].yaxis.set_label_coords(1.025+.1*_c,0.5)
            aux_twinx[_c].spines.left.set_position(("axes", 1.+.1*_c))
            aux_twinx[_c].spines['left'].set_color(color_pallete[_c])
            aux_twinx[_c].tick_params(colors=color_pallete[_c],which='both')
            aux_twinx[_c].plot(d_vect, linewidth=0.3,color=color_pallete[_c])
            # aux_twinx[_c].tick_params(axis='y')
            # aux_twinx[_c].set_ylabel(_i)
            formatter = mticker.ScalarFormatter(useMathText=True)
            formatter.set_powerlimits((-3,2))
            aux_twinx[_c].yaxis.set_major_formatter(formatter)
            lo = Labeloffset(aux_twinx[_c], label=_i, axis="y")
            aux_twinx.append(aux_twinx[_c].twinx())
            # --- --- 
            # print(_i,d_vects[_i].shape)
            # input()
            # --- --- 
        fig.canvas.manager.full_screen_toggle()
        fig.patch.set_facecolor((0,0,0))
        fig.tight_layout()
        # --- ---
