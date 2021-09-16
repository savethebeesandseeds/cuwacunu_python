# --- --- --- 
# cwcn_tsinuu_piaabo.py
# --- --- --- 
from pickle import FALSE
import torch
import torch.nn as nn
import logging

from torch.nn.modules import linear
# --- --- --- 
import cwcn_config
# --- --- --- 
# MEWAAJACURONE=0 #FIXME (desconocer) add to dictionary
# --- --- --- 
import torch.nn.functional as F
import math
# --- --- --- 
# based on https://github.com/pytorch/pytorch && https://github.com/mttk/rnn-classifier
# --- --- --- 
RNNS = ['LSTM', 'GRU']
class RECURRENT_ENCODER(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, nlayers=1, dropout=0.,
                bidirectional=True, rnn_type='GRU'):
        super(RECURRENT_ENCODER, self).__init__()
        self.bidirectional = bidirectional
        assert rnn_type in RNNS, 'Use one of the following: {}'.format(str(RNNS))
        rnn_cell = getattr(nn, rnn_type) # fetch constructor from torch.nn, cleaner than if
        self.rnn = rnn_cell(embedding_dim, hidden_dim, nlayers, 
            dropout=dropout, bidirectional=bidirectional,
            batch_first=True)
    def forward(self, input, hidden=None):
        return self.rnn(input, hidden)
# --- --- --- 
class LINEAR_ENCODER_CONVERTER(nn.Module):
    def __init__(self, INFORMATION_SIZE, SEQUENCE_SIZE, HIDDEN_BELLY_SIZE):
        super(LINEAR_ENCODER_CONVERTER, self).__init__()
        self.sequence_scale = 1. / math.sqrt(SEQUENCE_SIZE)
        # --- --- --- --- 
        # hipersensible_tit_fabric= lambda : ('hipersencible',1)
        # lowersensible_tit_fabric= lambda : ('lowersencible',0)
        # rand_tit_permutator=lambda x : hipersensible_tit_fabric() if not x % cwcn_config.CWCN_CONFIG().HIPERSENSIBLE_MODULE else lowersensible_tit_fabric()
        # self.maginal_sensible_tits=[[rand_tit_permutator(_+__*HIDDEN_BELLY_SIZE) for _ in range(HIDDEN_BELLY_SIZE)] for __ in range(HIDDEN_BELLY_SIZE)]
        # --- --- --- --- 
        self.maginal_lowersensible_belly_prime_a=nn.parameter.Parameter(torch.rand(INFORMATION_SIZE,1).to(cwcn_config.device),requires_grad=True)
        self.maginal_lowersensible_belly_prime_b=nn.parameter.Parameter(torch.rand(INFORMATION_SIZE,HIDDEN_BELLY_SIZE).to(cwcn_config.device),requires_grad=True)
        self.maginal_belly=nn.parameter.Parameter(torch.rand(HIDDEN_BELLY_SIZE,HIDDEN_BELLY_SIZE).to(cwcn_config.device),requires_grad=True)
        # --- --- --- --- 
        # --- --- --- --- --- --- --- --- --- 
    def forward(self, batched_information_sequence):
        #       # [N] : batch size
        #       # [L] : sequence size
        #       # [H] : information size
        #       # [B] : belly size
        #       # batched_information_sequence : [NxLxH]
        #       # maginal_lowersensible_belly_prime_a : [Hx1]
        #       # maginal_lowersensible_belly_prime_b : [HxB]

        # Here we assume q_dim == k_dim (dot product attention)
        # logging.warning("-- -- --")
        # logging.warning("batched_information_sequence : {}".format(batched_information_sequence.shape))
        # logging.warning("self.maginal_lowersensible_belly_prime_a : {}".format(self.maginal_lowersensible_belly_prime_a))
        # # logging.warning("self.maginal_lowersensible_belly_prime_b : {}".format(self.maginal_lowersensible_belly_prime_b))
        # logging.warning("-- -- --")
        # --- --- 
        x=torch.matmul(
            batched_information_sequence.transpose(1,2),
            batched_information_sequence,
        ) # [Nx(LxH)^T] * [NxLxH] -> [NxHxH]
        # logging.warning("x (1) : {}".format(x.shape))
        x=torch.matmul(
            x,
            self.maginal_lowersensible_belly_prime_a,
        ) # [NxHxH] * [Hx1] -> [NxHx1]
        x=x.squeeze(-1) # [NxHx1] -> [NxH]
        # logging.warning("x (2) : {}".format(x.shape))
        x=x.mul_(self.sequence_scale) # [NxH] -> [NxH]
        # logging.warning("x (3) : {}".format(x.shape))
        x=torch.matmul(
            x,
            self.maginal_lowersensible_belly_prime_b,
        ) # [NxH] * [HxB] -> [NxB]
        # logging.warning("x (4) : {}".format(x.shape))
        x=torch.matmul(
            x,
            self.maginal_belly,
        ) # [NxB] * [BxB] -> [NxB]
        # logging.warning("x (5) : {}".format(x.shape))

        # # batched_information_sequence = batched_information_sequence.unsqueeze(1) # [BxQ] -> [Bx1xQ]
        # # batched_recurrent_abseq = batched_recurrent_abseq.transpose(1,2) # [BxTxR] -> [BxRxT]
        # # batched_values = batched_values # [BxTxV] -> [BxTxV] # enable if rnn uses batch_first=False
        # # # batched_recurrent_abseq = batched_recurrent_abseq.transpose(0,1).transpose(1,2) # [TxBxR] -> [BxRxT] # enable if rnn uses batch_first=False
        # # # batched_values = batched_values.transpose(0,1) # [TxBxV] -> [BxTxV] # enable if rnn uses batch_first=False
        # # # --- --- 
        # # batched_energy = torch.bmm(batched_information_sequence, batched_recurrent_abseq) # [Bx1xQ]x[BxRxT] -> [Bx1xT]
        # # batched_energy = F.softmax(batched_energy.mul_(self.sequence_scale), dim=2) # scale, normalize
        # batched_linear_combination = torch.bmm(batched_energy, batched_values).squeeze(1) #[Bx1xT]x[BxTxV] -> [BxV]
        # return batched_energy, batched_linear_combination
        return x
# --- --- --- 
# --- --- --- --- --- --- --- 
# --- --- --- --- --- --- --- 
class TSINUU_as_marginal_method(nn.Module):
    def __init__(
            self, 
            ALLIU_SIZE, 
            UWAABO_SIZE, 
            MUNAAJPI_SIZE,
            RECURRENT_TYPE,
            RECURRENT_SEQ_SIZE, 
            RECURRENT_HIDEN_SIZE, 
            RECURRENT_N_LAYERS, 
            HIDDEN_BELLY_SIZE,
            UWAABO_HIDDEN_SIZE, 
            MUNAAJPI_HIDDEN_SIZE): # alliu is state, uwabao is action
        # --- --- --- 
        super(TSINUU_as_marginal_method, self).__init__()
        # --- --- --- 
        # self.rr_1=nn.RNN(
        #     input_size=ALLIU_SIZE,
        #     hidden_size=RECURRENT_HIDEN_SIZE,
        #     num_layers=RECURRENT_N_LAYERS,
        #     nonlinearity='tanh',
        #     bias=True,
        #     batch_first=True,
        #     dropout=0.0,
        #     bidirectional=False,
        # )
        # self.rr_state=None # rrn state
        # --- --- --- 
        # self.flat_1=nn.Flatten()
        # --- --- --- --- --- --- --- --- --- 
        self.drop_flag = cwcn_config.CWCN_CONFIG().dropout_flag
        self.drop_factor = cwcn_config.CWCN_CONFIG().dropout_prob
        self.RECURRENT_SEQ_SIZE=RECURRENT_SEQ_SIZE #FIXME now needed, not so cool!
        # --- --- --- --- --- --- --- --- --- 
        BIDIRECTIONAL=False
        self.encoder = RECURRENT_ENCODER(
            ALLIU_SIZE, 
            RECURRENT_HIDEN_SIZE, 
            nlayers=RECURRENT_N_LAYERS, 
            dropout=0.0,#self.drop_factor,
            bidirectional=BIDIRECTIONAL, 
            rnn_type=RECURRENT_TYPE,#/GRU/LSTM
        )
        # --- --- --- --- --- --- --- --- --- 
        self.attention = LINEAR_ENCODER_CONVERTER(
            INFORMATION_SIZE=RECURRENT_HIDEN_SIZE if not BIDIRECTIONAL else 2*RECURRENT_HIDEN_SIZE,
            SEQUENCE_SIZE=RECURRENT_SEQ_SIZE,
            HIDDEN_BELLY_SIZE=HIDDEN_BELLY_SIZE,
        )
        # --- --- --- --- --- --- --- --- --- 
        # --- --- --- --- --- --- --- --- --- 
        self.ul_1=nn.Linear(HIDDEN_BELLY_SIZE+ALLIU_SIZE, UWAABO_HIDDEN_SIZE)
        self.ul_2=nn.Linear(UWAABO_HIDDEN_SIZE, UWAABO_HIDDEN_SIZE)
        self.ul_3=nn.Linear(UWAABO_HIDDEN_SIZE, UWAABO_HIDDEN_SIZE)
        self.ul_4=nn.Linear(UWAABO_HIDDEN_SIZE, UWAABO_HIDDEN_SIZE)
        self.ul_5=nn.Linear(UWAABO_HIDDEN_SIZE, UWAABO_HIDDEN_SIZE)
        self.ul_6=nn.Linear(UWAABO_HIDDEN_SIZE, UWAABO_HIDDEN_SIZE)
        self.ul_7=nn.Linear(UWAABO_HIDDEN_SIZE, UWAABO_HIDDEN_SIZE)
        self.ul_8=nn.Linear(UWAABO_HIDDEN_SIZE, UWAABO_HIDDEN_SIZE)
        self.ul_9=nn.Linear(UWAABO_HIDDEN_SIZE, UWAABO_HIDDEN_SIZE)
        self.ul_10=nn.Linear(UWAABO_HIDDEN_SIZE, UWAABO_HIDDEN_SIZE)
        self.ul_11=nn.Linear(UWAABO_HIDDEN_SIZE, UWAABO_HIDDEN_SIZE)
        self.ul_12=nn.Linear(UWAABO_HIDDEN_SIZE, UWAABO_HIDDEN_SIZE)
        self.ul_13=nn.Linear(UWAABO_HIDDEN_SIZE, UWAABO_HIDDEN_SIZE)
        self.ul_14=nn.Linear(UWAABO_HIDDEN_SIZE, UWAABO_HIDDEN_SIZE)
        self.ul_15=nn.Linear(UWAABO_HIDDEN_SIZE, UWAABO_HIDDEN_SIZE)
        self.ul_16=nn.Linear(UWAABO_HIDDEN_SIZE, UWAABO_HIDDEN_SIZE)
        self.ul_17=nn.Linear(UWAABO_HIDDEN_SIZE, UWAABO_HIDDEN_SIZE)
        self.ul_18=nn.Linear(UWAABO_HIDDEN_SIZE, UWAABO_HIDDEN_SIZE)
        self.ul_19=nn.Linear(UWAABO_HIDDEN_SIZE, UWAABO_HIDDEN_SIZE)
        self.ul_20=nn.Linear(UWAABO_HIDDEN_SIZE, UWAABO_HIDDEN_SIZE)
        # --- 
        self.ul_p1=nn.Linear(UWAABO_HIDDEN_SIZE, UWAABO_SIZE)
        # --- ---
        self.ml_1=nn.Linear(HIDDEN_BELLY_SIZE+ALLIU_SIZE, MUNAAJPI_HIDDEN_SIZE)
        self.ml_2=nn.Linear(MUNAAJPI_HIDDEN_SIZE, MUNAAJPI_HIDDEN_SIZE)
        self.ml_3=nn.Linear(MUNAAJPI_HIDDEN_SIZE, MUNAAJPI_HIDDEN_SIZE)
        self.ml_4=nn.Linear(MUNAAJPI_HIDDEN_SIZE, MUNAAJPI_HIDDEN_SIZE)
        self.ml_5=nn.Linear(MUNAAJPI_HIDDEN_SIZE, MUNAAJPI_SIZE)
        # --- ---
        size = 0
        for p in self.parameters():
            size += p.nelement()
        logging.tsinuu_logging('Total param size: {}'.format(size))
        # --- ---
    def _foward_munaajpi_(self,_alliu,_attention):
        x=torch.cat((_alliu,_attention),1)
        x=nn.Softsign()(self.ml_1(x))
        if(self.training and self.drop_flag):
            x=nn.functional.dropout(x, p=self.drop_factor)
        x=nn.Softsign()(self.ml_2(x))
        if(self.training and self.drop_flag):
            x=nn.functional.dropout(x, p=self.drop_factor)
        x=nn.Softsign()(self.ml_3(x))
        if(self.training and self.drop_flag):
            x=nn.functional.dropout(x, p=self.drop_factor)
        x=nn.Softsign()(self.ml_4(x))
        if(self.training and self.drop_flag):
            x=nn.functional.dropout(x, p=self.drop_factor)
        value=self.ml_5(x)
        return value
    def _foward_uwaabo_(self,_alliu,_attention):
        # --- --- 
        x=torch.cat((_alliu,_attention),1)
        x=nn.Softsign()(self.ul_1(x))
        if(self.training and self.drop_flag):
            x=nn.functional.dropout(x, p=self.drop_factor)
        x=nn.Softsign()(self.ul_2(x))
        if(self.training and self.drop_flag):
            x=nn.functional.dropout(x, p=self.drop_factor)
        x=nn.Softsign()(self.ul_3(x))
        if(self.training and self.drop_flag):
            x=nn.functional.dropout(x, p=self.drop_factor)
        x=nn.Softsign()(self.ul_4(x))
        # # if(self.training and self.drop_flag):
        # #     x=nn.functional.dropout(x, p=self.drop_factor)
        # # x=nn.Softsign()(self.ul_5(x))
        # # if(self.training and self.drop_flag):
        # #     x=nn.functional.dropout(x, p=self.drop_factor)
        # x=nn.Softsign()(self.ul_6(x))
        # x=nn.Softsign()(self.ul_7(x))
        # x=nn.Softsign()(self.ul_8(x))
        # x=nn.Softsign()(self.ul_9(x))
        # x=nn.Softsign()(self.ul_10(x))
        # x=nn.Softsign()(self.ul_11(x))
        # x=nn.Softsign()(self.ul_12(x))
        # x=nn.Softsign()(self.ul_13(x))
        # x=nn.Softsign()(self.ul_14(x))
        # x=nn.Softsign()(self.ul_15(x))
        # x=nn.Softsign()(self.ul_16(x))
        # x=nn.Softsign()(self.ul_17(x))
        # x=nn.Softsign()(self.ul_18(x))
        # x=nn.Softsign()(self.ul_19(x))
        # x=nn.Softsign()(self.ul_20(x))

        # ADDING INFINITE CONTEXT MARGINAL LOGITS LAYER

        if(cwcn_config.CONTEXT_MARGINALIZED_TSINUU):
            y=torch.exp(nn.Softsign()(self.ul_p1(x)))
        else:
            y=nn.Softmax(-1)(self.ul_p1(x))
        return y
    def forward(self, input):
        try:
            # --- --- 
            alliu_state=input[:,0 if cwcn_config.CWCN_UJCAMEI_CAJTUCU_CONFIG.TIME_DECREMENTAL_SEQUENCE else -1,:]
            # --- --- Encoder
            x, hidden = self.encoder(
                input=input, 
                hidden=None
            )
            # --- --- Linear Decoder
            linear_combination = self.attention(batched_information_sequence=x) 
            # --- --- Munaajpi
            value=self._foward_munaajpi_(alliu_state, linear_combination)
            # --- --- Uwaabo
            uwaabo_state=self._foward_uwaabo_(alliu_state, linear_combination)
            # --- --- Tsane
            # dist  = torch.distributions.Categorical(uwaabo=uwaabo)
            if(cwcn_config.CONTEXT_MARGINALIZED_TSINUU):
                uwaabo_state=\
                    (uwaabo_state-uwaabo_state.min()+uwaabo_state.std()/cwcn_config.CWCN_UJCAMEI_CAJTUCU_CONFIG.TSANE_MIN_PROB_MARGIN)\
                        /(uwaabo_state.max()-uwaabo_state.min())
            tsane_dist=torch.distributions.Categorical(
                probs=uwaabo_state
            )
        except Exception as e:
            logging.error("ERROR:",e)
            raise e
        certainty = tsane_dist.probs
        return tsane_dist, value, certainty
# --- --- --- 