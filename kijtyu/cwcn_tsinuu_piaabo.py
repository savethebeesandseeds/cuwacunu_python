# --- --- --- 
# cwcn_tsinuu_piaabo.py
# --- --- --- 
import math
import torch
import logging
# --- --- --- 
import cwcn_config
# --- --- --- 
# --- --- --- 
# MEWAAJACURONE=0 #FIXME (desconocer) add to dictionary
# --- --- --- 
# based on https://github.com/pytorch/pytorch && https://github.com/mttk/rnn-classifier
# --- --- --- 
RNNS = ['LSTM', 'GRU']
class RECURRENT_ENCODER(torch.nn.Module):
    def __init__(self, embedding_dim, hidden_dim, nlayers=1, dropout=0.,
                bidirectional=True, rnn_type='GRU'):
        super(RECURRENT_ENCODER, self).__init__()
        self.bidirectional = bidirectional
        assert rnn_type in RNNS, 'Use one of the following: {}'.format(str(RNNS))
        rnn_cell = getattr(torch.nn, rnn_type) # fetch constructor from torch.nn, cleaner than if
        self.rnn = rnn_cell(embedding_dim, hidden_dim, nlayers, 
            dropout=dropout, bidirectional=bidirectional,
            batch_first=True)
    def forward(self, input, hidden=None):
        return self.rnn(input, hidden)
# --- --- --- 
class LINEAR_ENCODER_CONVERTER(torch.nn.Module):
    def __init__(self, INFORMATION_SIZE, SEQUENCE_SIZE, HIDDEN_BELLY_SIZE):
        super(LINEAR_ENCODER_CONVERTER, self).__init__()
        self.sequence_scale = 1. / math.sqrt(SEQUENCE_SIZE)
        # --- --- --- --- 
        # hipersensible_tit_fabric= lambda : ('hipersencible',1)
        # lowersensible_tit_fabric= lambda : ('lowersencible',0)
        # rand_tit_permutator=lambda x : hipersensible_tit_fabric() if not x % cwcn_config.CWCN_CONFIG().HIPERSENSIBLE_MODULE else lowersensible_tit_fabric()
        # self.maginal_sensible_tits=[[rand_tit_permutator(_+__*HIDDEN_BELLY_SIZE) for _ in range(HIDDEN_BELLY_SIZE)] for __ in range(HIDDEN_BELLY_SIZE)]
        # --- --- --- --- 
        self.maginal_lowersensible_belly_prime_a=torch.nn.parameter.Parameter(torch.rand(INFORMATION_SIZE,1).to(cwcn_config.device),requires_grad=True)
        self.maginal_lowersensible_belly_prime_b=torch.nn.parameter.Parameter(torch.rand(INFORMATION_SIZE,HIDDEN_BELLY_SIZE).to(cwcn_config.device),requires_grad=True)
        self.maginal_belly=torch.nn.parameter.Parameter(torch.rand(HIDDEN_BELLY_SIZE,HIDDEN_BELLY_SIZE).to(cwcn_config.device),requires_grad=True)
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
class TSINUU_as_marginal_method(torch.nn.Module):
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
            FORECAST_HIDDEN_SIZE,
            FORECAST_N_HORIZONS,
            UWAABO_HIDDEN_SIZE, 
            MUNAAJPI_HIDDEN_SIZE): # alliu is state, uwabao is action
        # --- --- --- 
        super(TSINUU_as_marginal_method, self).__init__()
        # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ENCODER (train by FORECAST)
        # --- --- --- --- --- --- --- --- --- 
        self.drop_flag = cwcn_config.CWCN_CONFIG().dropout_flag
        self.drop_factor = cwcn_config.CWCN_CONFIG().dropout_prob
        self.RECURRENT_SEQ_SIZE=RECURRENT_SEQ_SIZE #FIXME now needed, not so cool!
        self.IS_RECURRENT_ENCODER_BIDIRECTIONAL=False
        # --- --- --- --- --- --- --- --- --- 
        self.encoder = RECURRENT_ENCODER(
            ALLIU_SIZE, 
            RECURRENT_HIDEN_SIZE, 
            nlayers=RECURRENT_N_LAYERS, 
            dropout= self.drop_factor if self.drop_flag else 0.0,
            bidirectional=self.IS_RECURRENT_ENCODER_BIDIRECTIONAL, 
            rnn_type=RECURRENT_TYPE,#/GRU/LSTM
        )
        # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ATTENTION (train by FORECAST)
        # --- --- --- --- --- --- --- --- --- 
        # --- --- --- --- --- --- --- --- --- 
        self.attention = LINEAR_ENCODER_CONVERTER(
            INFORMATION_SIZE=RECURRENT_HIDEN_SIZE if not self.IS_RECURRENT_ENCODER_BIDIRECTIONAL else 2*RECURRENT_HIDEN_SIZE,
            SEQUENCE_SIZE=RECURRENT_SEQ_SIZE,
            HIDDEN_BELLY_SIZE=HIDDEN_BELLY_SIZE,
        )
        # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- FORECAST
        # --- --- --- --- --- --- --- --- --- 
        self.forecast_scale=torch.tensor(4).squeeze(0).to(cwcn_config.device)
        # --- --- --- --- --- --- --- --- --- 
        self.forecast_1 = torch.nn.Linear(HIDDEN_BELLY_SIZE,FORECAST_HIDDEN_SIZE)
        self.forecast_2 = torch.nn.Linear(FORECAST_HIDDEN_SIZE,FORECAST_HIDDEN_SIZE)
        self.forecast_3 = torch.nn.Linear(FORECAST_HIDDEN_SIZE,FORECAST_HIDDEN_SIZE)
        self.forecast_4 = torch.nn.Linear(FORECAST_HIDDEN_SIZE,FORECAST_HIDDEN_SIZE)
        self.forecast_5 = torch.nn.Linear(FORECAST_HIDDEN_SIZE,FORECAST_HIDDEN_SIZE)
        # --- --- --- --- --- --- --- --- --- 
        self.forecast_p1 = torch.nn.Linear(FORECAST_HIDDEN_SIZE,FORECAST_N_HORIZONS)
        # --- --- --- --- --- --- --- --- --- 
        self.forescast_parameters = []
        self.forescast_parameters.extend(self.encoder.parameters())
        self.forescast_parameters.extend(self.attention.parameters())
        self.forescast_parameters.extend(self.forecast_1.parameters())
        self.forescast_parameters.extend(self.forecast_2.parameters())
        self.forescast_parameters.extend(self.forecast_3.parameters())
        self.forescast_parameters.extend(self.forecast_4.parameters())
        self.forescast_parameters.extend(self.forecast_5.parameters())
        self.forescast_parameters.extend(self.forecast_p1.parameters())
        # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- UWAABO (train with RL)
        # --- --- --- --- --- --- --- --- --- 
        # --- --- --- --- --- --- --- --- --- 
        self.ul_1=torch.nn.Linear(FORECAST_N_HORIZONS+ALLIU_SIZE, UWAABO_HIDDEN_SIZE)
        self.ul_2=torch.nn.Linear(UWAABO_HIDDEN_SIZE, UWAABO_HIDDEN_SIZE)
        self.ul_3=torch.nn.Linear(UWAABO_HIDDEN_SIZE, UWAABO_HIDDEN_SIZE)
        self.ul_4=torch.nn.Linear(UWAABO_HIDDEN_SIZE, UWAABO_HIDDEN_SIZE)
        self.ul_5=torch.nn.Linear(UWAABO_HIDDEN_SIZE, UWAABO_HIDDEN_SIZE)
        # # self.ul_6=torch.nn.Linear(UWAABO_HIDDEN_SIZE, UWAABO_HIDDEN_SIZE)
        # # self.ul_7=torch.nn.Linear(UWAABO_HIDDEN_SIZE, UWAABO_HIDDEN_SIZE)
        # # self.ul_8=torch.nn.Linear(UWAABO_HIDDEN_SIZE, UWAABO_HIDDEN_SIZE)
        # # self.ul_9=torch.nn.Linear(UWAABO_HIDDEN_SIZE, UWAABO_HIDDEN_SIZE)
        # # self.ul_10=torch.nn.Linear(UWAABO_HIDDEN_SIZE, UWAABO_HIDDEN_SIZE)
        # # self.ul_11=torch.nn.Linear(UWAABO_HIDDEN_SIZE, UWAABO_HIDDEN_SIZE)
        # # self.ul_12=torch.nn.Linear(UWAABO_HIDDEN_SIZE, UWAABO_HIDDEN_SIZE)
        # # self.ul_13=torch.nn.Linear(UWAABO_HIDDEN_SIZE, UWAABO_HIDDEN_SIZE)
        # # self.ul_14=torch.nn.Linear(UWAABO_HIDDEN_SIZE, UWAABO_HIDDEN_SIZE)
        # # self.ul_15=torch.nn.Linear(UWAABO_HIDDEN_SIZE, UWAABO_HIDDEN_SIZE)
        # # self.ul_16=torch.nn.Linear(UWAABO_HIDDEN_SIZE, UWAABO_HIDDEN_SIZE)
        # # self.ul_17=torch.nn.Linear(UWAABO_HIDDEN_SIZE, UWAABO_HIDDEN_SIZE)
        # # self.ul_18=torch.nn.Linear(UWAABO_HIDDEN_SIZE, UWAABO_HIDDEN_SIZE)
        # # self.ul_19=torch.nn.Linear(UWAABO_HIDDEN_SIZE, UWAABO_HIDDEN_SIZE)
        # # self.ul_20=torch.nn.Linear(UWAABO_HIDDEN_SIZE, UWAABO_HIDDEN_SIZE)
        # --- --- --- --- --- --- --- --- --- 
        self.ul_p1=torch.nn.Linear(UWAABO_HIDDEN_SIZE, UWAABO_SIZE)
        # --- --- --- --- --- --- --- --- --- 
        self.rl_parameters=[]
        self.rl_parameters.extend(self.ul_1.parameters())
        self.rl_parameters.extend(self.ul_2.parameters())
        self.rl_parameters.extend(self.ul_3.parameters())
        self.rl_parameters.extend(self.ul_4.parameters())
        self.rl_parameters.extend(self.ul_5.parameters())
        self.rl_parameters.extend(self.ul_p1.parameters())
        # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- MUNAAJPI (train with RL)
        # --- --- --- --- --- --- --- --- --- 
        # --- --- --- --- --- --- --- --- --- 
        self.ml_1=torch.nn.Linear(FORECAST_N_HORIZONS+ALLIU_SIZE, MUNAAJPI_HIDDEN_SIZE)
        self.ml_2=torch.nn.Linear(MUNAAJPI_HIDDEN_SIZE, MUNAAJPI_HIDDEN_SIZE)
        self.ml_3=torch.nn.Linear(MUNAAJPI_HIDDEN_SIZE, MUNAAJPI_HIDDEN_SIZE)
        self.ml_4=torch.nn.Linear(MUNAAJPI_HIDDEN_SIZE, MUNAAJPI_HIDDEN_SIZE)
        self.ml_5=torch.nn.Linear(MUNAAJPI_HIDDEN_SIZE, MUNAAJPI_HIDDEN_SIZE)
        # --- --- --- --- --- --- --- --- --- 
        self.ml_p1=torch.nn.Linear(MUNAAJPI_HIDDEN_SIZE, MUNAAJPI_SIZE)
        # --- --- --- --- --- --- --- --- --- 
        self.rl_parameters.extend(self.ml_1.parameters())
        self.rl_parameters.extend(self.ml_2.parameters())
        self.rl_parameters.extend(self.ml_3.parameters())
        self.rl_parameters.extend(self.ml_4.parameters())
        self.rl_parameters.extend(self.ml_5.parameters())
        # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- 
        # --- --- --- --- --- --- --- --- --- 
        # --- --- --- --- --- --- --- --- --- 
        size = 0
        for p in self.parameters():
            size += p.nelement()
        logging.tsinuu_logging('[Total   ] params size: {}'.format(size))
        # --- --- --- --- --- --- --- --- --- 
        size = 0
        logging.tsinuu_logging('[Forecast] params size: {}'.format(size))
        for p in self.forescast_parameters:
            size += p.nelement()
        # --- --- --- --- --- --- --- --- --- 
        size = 0
        logging.tsinuu_logging('[RL      ] params size: {}'.format(size))
        for p in self.rl_parameters:
            size += p.nelement()
        # --- --- --- --- --- --- --- --- --- 
        # --- --- --- --- --- --- --- --- --- 
    def _foward_forecast_(self,_attention):
        # if(self.training and self.drop_flag):
        #     x=torch.nn.functional.dropout(x, p=self.drop_factor)
        x=_attention
        x=torch.nn.Softsign()(self.forecast_1(x))
        if(self.training and self.drop_flag):
            x=torch.nn.functional.dropout(x, p=self.drop_factor)
        x=torch.nn.Softsign()(self.forecast_2(x))
        if(self.training and self.drop_flag):
            x=torch.nn.functional.dropout(x, p=self.drop_factor)
        x=torch.nn.Softsign()(self.forecast_3(x))
        x=torch.nn.Softsign()(self.forecast_4(x))
        x=torch.nn.Softsign()(self.forecast_5(x))
        forecast_out=torch.multiply(self.forecast_scale,torch.nn.Softsign()(self.forecast_p1(x)))
        return forecast_out
    def _foward_munaajpi_(self,_alliu,_forecast_uwaabo):
        # if(self.training and self.drop_flag):
        #     x=torch.nn.functional.dropout(x, p=self.drop_factor)
        x=torch.cat((_alliu,_forecast_uwaabo),1)
        if(self.training and self.drop_flag):
            x=torch.nn.functional.dropout(x, p=self.drop_factor)
        x=torch.nn.Softsign()(self.ml_1(x))
        if(self.training and self.drop_flag):
            x=torch.nn.functional.dropout(x, p=self.drop_factor)
        x=torch.nn.Softsign()(self.ml_2(x))
        x=torch.nn.Softsign()(self.ml_3(x))
        x=torch.nn.Softsign()(self.ml_4(x))
        x=torch.nn.Softsign()(self.ml_5(x))
        munaajpi_value=self.ml_p1(x)
        return munaajpi_value
    def _foward_uwaabo_(self,_alliu,_forecast_uwaabo):
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # _price=self.alliu_to_key(_alliu,'price')
        x=torch.cat((_alliu,_forecast_uwaabo),1)
        if(self.training and self.drop_flag):
            x=torch.nn.functional.dropout(x, p=self.drop_factor)
        x=torch.nn.SELU()(self.ul_1(x))
        if(self.training and self.drop_flag):
            x=torch.nn.functional.dropout(x, p=self.drop_factor)
        x=torch.nn.SELU()(self.ul_2(x))
        x=torch.nn.SELU()(self.ul_3(x))
        x=torch.nn.SELU()(self.ul_4(x))
        x=torch.nn.SELU()(self.ul_5(x))
        # # x=torch.nn.SELU()(self.ul_6(x))
        # # x=torch.nn.SELU()(self.ul_7(x))
        # # x=torch.nn.SELU()(self.ul_8(x))
        # # x=torch.nn.SELU()(self.ul_9(x))
        # # x=torch.nn.SELU()(self.ul_10(x))
        # # x=torch.nn.SELU()(self.ul_11(x))
        # # x=torch.nn.SELU()(self.ul_12(x))
        # # x=torch.nn.SELU()(self.ul_13(x))
        # # x=torch.nn.SELU()(self.ul_14(x))
        # # x=torch.nn.SELU()(self.ul_15(x))
        # # x=torch.nn.SELU()(self.ul_16(x))
        # # x=torch.nn.SELU()(self.ul_17(x))
        # # x=torch.nn.Softsign()(self.ul_18(x))
        # # x=torch.nn.Softsign()(self.ul_19(x))
        # # x=torch.nn.Softsign()(self.ul_20(x))
        # ADDING INFINITE HIPERSENCIBLE CONTEXT MARGINAL LOGITS LAYER, # what?
        if(cwcn_config.CONTEXT_MARGINALIZED_TSINUU):
            # y=torch.exp(torch.nn.Softsign()(self.ul_p1(x)))
            y=torch.nn.Softsign()(self.ul_p1(x))
        else:
            assert(False), "please configure cwcn_config.CONTEXT_MARGINALIZED_TSINUU" #FIXME what?
            y=torch.nn.Softmax(-1)(self.ul_p1(x))
        return y
    def forward(self, model_input):
        try:
            # --- --- 
            batched_alliu_state=model_input[:,0 if cwcn_config.CWCN_UJCAMEI_CAJTUCU_CONFIG.TIME_DECREMENTAL_SEQUENCE else -1,:]
            # --- --- ENCODER
            x_encoded, hidden = self.encoder(
                input=model_input, 
                hidden=None
            )
            # --- --- LINEAR DECODER
            batched_linear_combination = self.attention(batched_information_sequence=x_encoded) 
            # --- --- FORECAST
            batched_forecast_combination = self._foward_forecast_(batched_linear_combination)
            batched_forecast_combination_copy = batched_forecast_combination.clone()
            batched_forecast_combination_copy=batched_forecast_combination_copy.detach()
            # --- --- MUNAAJPI
            munaajpi_value=self._foward_munaajpi_(batched_alliu_state, batched_forecast_combination_copy)
            # --- --- UWAABO
            uwaabo_state=self._foward_uwaabo_(batched_alliu_state, batched_forecast_combination_copy)
            # ... doe snot work unsigl uwaabo aprox alliu #FIXME is a great idea output aprox input
            # --- --- TSANE
            # dist  = torch.distributions.Categorical(uwaabo=uwaabo)
            if(cwcn_config.CONTEXT_MARGINALIZED_TSINUU):
                uwaabo_state=\
                    (uwaabo_state-uwaabo_state.min()+uwaabo_state.std()/cwcn_config.CWCN_UJCAMEI_CAJTUCU_CONFIG.TSANE_MIN_PROB_MARGIN)\
                        /(uwaabo_state.max()-uwaabo_state.min())
            tsane_composed_distribution=torch.distributions.Categorical(
                probs=uwaabo_state
            )
        except Exception as e:
            logging.error("ERROR ON CONTEXT MAGINALIZED TSINUU MODEL FORWARD:",e)
            raise e #FIXME better exception handdler
        tsane_certainty = tsane_composed_distribution.probs
        return tsane_composed_distribution, munaajpi_value, tsane_certainty, batched_forecast_combination
# --- --- --- 