# --- --- --- 
import torch
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