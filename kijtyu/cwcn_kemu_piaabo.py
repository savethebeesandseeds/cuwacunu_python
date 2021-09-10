# --- --- ---
import os
import sys
# --- --- ---
import torch
# --- --- --- ---
from matplotlib import pyplot as plt
# --- --- ---
import cwcn_config
# --- --- ---
def kemu_to_tensor(asset):
    return torch.FloatTensor([asset]).squeeze(0).to(cwcn_config.device)
def kemu_assert_dir(__path):
    if not os.path.exists(__path):
        os.makedirs(__path)
    return __path
def kemu_normalize(x):
    x -= x.mean()
    x /= (x.std() + 1e-8)
    return x
def kemu_pretty_print_object(d, indent=0, set_ident=26):
    sys.stdout.write("\n")
    for key, value in d.items():
        sys.stdout.write('\t' * indent + "{}{}{}".format(cwcn_config.CWCN_COLORS.HEADER,str(key),cwcn_config.CWCN_COLORS.REGULAR))
        sys.stdout.write('.' * abs(set_ident-len(str(key))))
        if isinstance(value, dict):
            kemu_pretty_print_object(value, indent+1)
        else:
            sys.stdout.write('\t' * (indent+1) + "{}{}{}\n".format(cwcn_config.CWCN_COLORS.GREEN,str(value),cwcn_config.CWCN_COLORS.REGULAR))
    sys.stdout.flush()
def kemu_plot_queue_item(c_queue,itm):
        # --- ---
        d_vects=c_queue._dict_vectorize_queue_(_type='array')
        # --- ---
        fig, ax = plt.subplots(1, 1)
        fig.canvas.manager.full_screen_toggle()
        fig.patch.set_facecolor((0,0,0))
        ax.set_title("{} - {} - {}".format(ax.get_title(),cwcn_config.CWCN_CONFIG().AHPA_ID,itm),color=(1,1,1))
        ax.set_facecolor((0,0,0))
        ax.tick_params(colors='white',which='both')
        ax.spines['bottom'].set_color('white')
        ax.spines['top'].set_color('white')
        ax.spines['right'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        # Plot data
        aux_twinx={}
        for _c,_i in enumerate(itm.split(',')):
            # aux_twinx[_i]=ax.twinx()
            # aux_twinx[_i].spines.right.set_position(("axes", 1.+0.5*_c))
            ax.plot(d_vects[_i], linewidth=0.3, label=_i)
            ax.set_ylabel(_i)
            ax.legend(itm.split(','))
            # print(_i,d_vects[_i].shape)
            # input()
        # --- ---
