# --- --- ---
# cwcn_kemu_piaabo
# --- --- ---
import os
import sys
# --- --- ---
import torch
import numpy as np
import logging
# --- --- --- ---
import argparse
import matplotlib.ticker as mticker
from matplotlib import pyplot as plt
# --- --- ---
import cwcn_config
# --- --- ---
def make_parser(): #FIXME not implemented
    parser = argparse.ArgumentParser(description='PyTorch Cuwacunu')
    # parser.add_argument('--data', type=str, default='SST',
    #                         help='Data corpus: [SST, TREC, IMDB]')
    # parser.add_argument('--model', type=str, default='LSTM',
    #                         help='type of recurrent net [LSTM, GRU]')
    # parser.add_argument('--emsize', type=int, default=300,
    #                         help='size of word embeddings [Uses pretrained on 50, 100, 200, 300]')
    # parser.add_argument('--hidden', type=int, default=500,
    #                         help='number of hidden units for the RNN encoder')
    # parser.add_argument('--nlayers', type=int, default=2,
    #                         help='number of layers of the RNN encoder')
    # parser.add_argument('--lr', type=float, default=1e-3,
    #                         help='initial learning rate')
    # parser.add_argument('--clip', type=float, default=5,
    #                         help='gradient clipping')
    # parser.add_argument('--epochs', type=int, default=10,
    #                         help='upper epoch limit')
    # parser.add_argument('--batch_size', type=int, default=32, metavar='N',
    #                         help='batch size')
    # parser.add_argument('--drop', type=float, default=0,
    #                         help='dropout')
    # parser.add_argument('--bi', action='store_true',
    #                         help='[USE] bidirectional encoder')
    # parser.add_argument('--cuda', action='store_false',
    #                     help='[DONT] use CUDA')
    # parser.add_argument('--fine', action='store_true', 
    #                     help='use fine grained labels in SST')
    return parser
# --- --- --- --- 
def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cwcn_config.device=='cuda':
        torch.cuda.manual_seed_all(seed)
# --- --- --- --- 
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
# # def kemu_plot_queue_item_backup(c_queue,itm):
# #         # --- ---
# #         d_vects=c_queue._dict_vectorize_queue_(_type='array')
# #         # --- ---
# #         fig, ax = plt.subplots(1, 1)
# #         fig.canvas.manager.full_screen_toggle()
# #         fig.patch.set_facecolor((0,0,0))
# #         ax.set_title("{} - {} - {}".format(ax.get_title(),cwcn_config.CWCN_CONFIG().AHPA_ID,itm),color=(1,1,1))
# #         ax.set_facecolor((0,0,0))
# #         ax.tick_params(colors='white',which='both')
# #         ax.spines['bottom'].set_color('white')
# #         ax.spines['top'].set_color('white')
# #         ax.spines['right'].set_color('white')
# #         ax.spines['left'].set_color('white')
# #         ax.xaxis.label.set_color('white')
# #         ax.yaxis.label.set_color('white')
# #         # Plot data
# #         aux_twinx={}
# #         for _c,_i in enumerate(itm.split(',')):
# #             # aux_twinx[_i]=ax.twinx()
# #             # aux_twinx[_i].spines.right.set_position(("axes", 1.+0.5*_c))
# #             ax.plot(d_vects[_i], linewidth=0.3, label=_i)
# #             ax.set_ylabel(_i)
# #             ax.legend(itm.split(','))
# #             # print(_i,d_vects[_i].shape)
# #             # input()
# #         # --- ---

def kemu_plot_queue_item(c_queue,itm):
    try:
        d_vects=c_queue._dict_vectorize_queue_(_type='array')
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
        color_pallete=cwcn_config.CWCN_OPTIONS.COLOR_PALLETE
        # --- ---
        fig, ax = plt.subplots(1, 1)
        # ax.set_title("{}".format(),color=(1,1,1),**{'fontname':'DejaVu Sans'})
        # Plot data
        c_yielder=(_pl for _pl in itm.split(','))
        ax.legend(itm)
        ax.set_title("{} // {}".format(ax.get_title(),itm),color=(1,1,1),**{'fontname':'DejaVu Sans'})
        # --- ---
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
            if(":" not in _i):
                d_vect=d_vects[_i]
                if(d_vect.shape[0]==1): 
                    d_vect=d_vects[_i][0]
            else:
                d_vect=[_d[int(_i.split(':')[1])] for _d in d_vects[_i.split(':')[0]]]
                # d_vect=d_vects[_i.split(':')[0]][_i.split(':')[1]]
            # --- --- 
            # print("{} : {}".format(_i,len(d_vect)))
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
    except Exception as e:
        logging.error("error on kemu_plot queue item : {}".format(e))