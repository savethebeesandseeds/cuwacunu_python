from pickle import FALSE
import torch
import torch.nn as nn
import logging
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
class Encoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, nlayers=1, dropout=0.,
                bidirectional=True, rnn_type='GRU'):
        super(Encoder, self).__init__()
        self.bidirectional = bidirectional
        assert rnn_type in RNNS, 'Use one of the following: {}'.format(str(RNNS))
        rnn_cell = getattr(nn, rnn_type) # fetch constructor from torch.nn, cleaner than if
        self.rnn = rnn_cell(embedding_dim, hidden_dim, nlayers, 
                            dropout=dropout, bidirectional=bidirectional,
                            batch_first=True)
    def forward(self, input, hidden=None):
        return self.rnn(input, hidden)
# --- --- --- 
class Attention(nn.Module):
    def __init__(self, query_dim):
        super(Attention, self).__init__()
        self.scale = 1. / math.sqrt(query_dim)
    def forward(self, query, keys, values):
        # Query = [BxQ]
        # Keys = [TxBxK]
        # Values = [TxBxV]
        # Outputs = a:[TxB], lin_comb:[BxV]
        # Here we assume q_dim == k_dim (dot product attention)
        # print("-- -- --")
        # print("query : {}".format(query.shape))
        # print("keys : {}".format(keys.shape))
        # print("values : {}".format(values.shape))
        # print("-- -- --")
        # --- --- 
        query = query.unsqueeze(1) # [BxQ] -> [Bx1xQ]
        keys = keys.transpose(1,2) # [BxTxK] -> [BxKxT]
        values = values # [BxTxV] -> [BxTxV] # enable if rnn uses batch_first=False
        # keys = keys.transpose(0,1).transpose(1,2) # [TxBxK] -> [BxKxT] # enable if rnn uses batch_first=False
        # values = values.transpose(0,1) # [TxBxV] -> [BxTxV] # enable if rnn uses batch_first=False
        # --- --- 
        energy = torch.bmm(query, keys) # [Bx1xQ]x[BxKxT] -> [Bx1xT]
        energy = F.softmax(energy.mul_(self.scale), dim=2) # scale, normalize
        linear_combination = torch.bmm(energy, values).squeeze(1) #[Bx1xT]x[BxTxV] -> [BxV]
        return energy, linear_combination
# --- --- --- 
# --- --- --- --- --- --- --- 
# --- --- --- --- --- --- --- 
class TSINUU_ACTOR_CRITIC(nn.Module):
    def __init__(
            self, 
            ALLIU_SIZE, 
            UWAABO_SIZE, 
            RECURRENT_TYPE,
            RECURRENT_SEQ_SIZE, 
            RECURRENT_HIDEN_SIZE, 
            RECURRENT_N_LAYERS, 
            UWAABO_HIDDEN_SIZE, 
            MUNAAJPI_HIDDEN_SIZE): # alliu is state, uwabao is action
        # --- --- --- 
        super(TSINUU_ACTOR_CRITIC, self).__init__()
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
        self.RECURRENT_SEQ_SIZE=RECURRENT_SEQ_SIZE #FIXME not needed cool!
        # --- --- --- --- --- --- --- --- --- 
        BIDIRECTIONAL=False
        self.encoder = Encoder(
            ALLIU_SIZE, 
            RECURRENT_HIDEN_SIZE, 
            nlayers=RECURRENT_N_LAYERS, 
            dropout=0.,
            bidirectional=BIDIRECTIONAL, 
            rnn_type=RECURRENT_TYPE,#/GRU/LSTM
        )
        self.attention = Attention(
            query_dim=RECURRENT_HIDEN_SIZE if not BIDIRECTIONAL else 2*RECURRENT_HIDEN_SIZE
        )
        # --- --- --- --- --- --- --- --- --- 
        # --- --- --- --- --- --- --- --- --- 
        self.ul_1=nn.Linear(RECURRENT_HIDEN_SIZE, UWAABO_HIDDEN_SIZE)
        self.ul_2=nn.Linear(UWAABO_HIDDEN_SIZE, UWAABO_HIDDEN_SIZE)
        self.ul_3=nn.Linear(UWAABO_HIDDEN_SIZE, UWAABO_HIDDEN_SIZE)
        self.ul_4=nn.Linear(UWAABO_HIDDEN_SIZE, UWAABO_HIDDEN_SIZE)
        # --- 
        self.ul_p1=nn.Linear(UWAABO_HIDDEN_SIZE, UWAABO_HIDDEN_SIZE)
        self.ul_p2=nn.Linear(UWAABO_HIDDEN_SIZE, UWAABO_SIZE)
        # --- 
        # self.ul_s1=nn.Linear(UWAABO_HIDDEN_SIZE, UWAABO_HIDDEN_SIZE)
        # self.ul_s2=nn.Linear(UWAABO_HIDDEN_SIZE, UWAABO_SIZE)
        # --- ---
        self.ml_1=nn.Linear(RECURRENT_HIDEN_SIZE, MUNAAJPI_HIDDEN_SIZE)
        self.ml_2=nn.Linear(MUNAAJPI_HIDDEN_SIZE, MUNAAJPI_HIDDEN_SIZE)
        self.ml_3=nn.Linear(MUNAAJPI_HIDDEN_SIZE, MUNAAJPI_HIDDEN_SIZE)
        self.ml_4=nn.Linear(MUNAAJPI_HIDDEN_SIZE, MUNAAJPI_HIDDEN_SIZE)
        self.ml_5=nn.Linear(MUNAAJPI_HIDDEN_SIZE, 1)
        # --- ---
        # self.iil_1=nn.Linear(ALLIU_SIZE+UWAABO_SIZE, IITEPI_HIDDEN_SIZE)
        # self.iil_2=nn.Linear(IITEPI_HIDDEN_SIZE, IITEPI_HIDDEN_SIZE)
        # self.iil_3=nn.Linear(IITEPI_HIDDEN_SIZE, ALLIU_SIZE)
        # --- ---
        # self.mjl_1=nn.Linear(ALLIU_SIZE, hidden_size)
        # self.mjl_2=nn.Linear(ALLIU_SIZE, hidden_size)
        # self.mjl_3=nn.Linear(ALLIU_SIZE, hidden_size)
        # --- ---
        size = 0
        for p in self.parameters():
            size += p.nelement()
        logging.tsinuu_logging('Total param size: {}'.format(size))
        # --- ---
        # self.min_sigma=torch.FloatTensor([0.001]).squeeze(0)
        # self.sigma_gain=torch.FloatTensor([2.0]).squeeze(0)
        # self.log_sigma=nn.Parameter(torch.ones(UWAABO_SIZE) * (sigma+self.min_sigma))
    def _foward_encoder_(self,x):
        outputs, hidden = self.encoder(x)
        if isinstance(hidden, tuple): # LSTM
            hidden = hidden[1] # take the cell state    
        if self.encoder.bidirectional: # need to concat the last 2 hidden layers
            hidden = torch.cat([hidden[-1], hidden[-2]], dim=1)
        else:
            hidden = hidden[-1]
        # max across T?
        # Other options (work worse on a few tests):
        # linear_combination, _ = torch.max(outputs, 0)
        # linear_combination = torch.mean(outputs, 0)
        energy, linear_combination = \
            self.attention(
                query=hidden, 
                keys=outputs, 
                values=outputs) 
        return energy, linear_combination
    def _foward_munaajpi_(self,x):
        if(self.drop_flag):nn.Dropout(self.drop_factor)
        x=nn.Softsign()(self.ml_1(x))
        if(self.drop_flag):nn.Dropout(self.drop_factor)
        x=nn.Softsign()(self.ml_2(x))
        if(self.drop_flag):nn.Dropout(self.drop_factor)
        x=nn.Softsign()(self.ml_3(x))
        if(self.drop_flag):nn.Dropout(self.drop_factor)
        x=nn.Softsign()(self.ml_4(x))
        x=self.ml_4(x)
        value=self.ml_5(x)
        return value
    def _foward_uwaabo_(self,x):
        x=self.ul_1(x)
        x=nn.Softsign()(x)
        x=nn.Softsign()(self.ul_2(x))
        x=nn.Softsign()(self.ul_3(x))
        x=nn.Softsign()(self.ul_4(x))
        # --- --- 
        probs=nn.Softsign()(self.ul_p1(x))
        probs=nn.Softmax(-1)(self.ul_p2(probs))
        # sigma = self.log_sigma.exp().expand_as(probs) + self.min_sigma
        # sigma=nn.Softsign()(self.ul_s1(x))
        # sigma=nn.Softmax(-1)(self.ul_s2(sigma))*self.sigma_gain + self.min_sigma
        # # print(x.shape)
        try:
            dist  = torch.distributions.Categorical(probs)
        except Exception as e:
            print("ERROR:",e)
            print("probs:",probs)
            raise e
        # out = torch.Categorical(probs)
        # out = torch.multinomial(dist.sample(), 1)
        return dist
    # def _foward_iitepi_(self,x,a): # state, actions
    #     x=torch.cat((x,a),0)
    #     x=self.iil_1(x)
    #     x=nn.Softsign()(x)
    #     x=self.iil_2(x)
    #     x=nn.Softsign()(x)
    #     x=self.iil_3(x)
    #     n_state=nn.Softsign()(x)
    #     return n_state # next_state
    def forward(self, x):
        # # x,self.rr_state=self.rr_1(x)
        # # # x=self.flat_1(x)
        # # x=self.mean(x)
        # print("x : {}".format(x.shape))
        energy, linear_combination=self._foward_encoder_(x)
        # print("energy : {}".format(energy.shape))
        # print("linear_combination : {}".format(linear_combination.shape))
        # input()
        value=self._foward_munaajpi_(linear_combination)
        dist=self._foward_uwaabo_(linear_combination)
        return dist, value, energy
# --- --- --- 