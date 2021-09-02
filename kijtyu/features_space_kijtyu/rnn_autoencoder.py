import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import logging
import pandas as pd
# --- --- --- 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] :: %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

# --- --- --- 
os.environ['CWCN_HOME_FOLDER']='/home/waajacu/work/cuwacunu.waajacu/cuwacunu_python'
os.environ['CWCN_INSTRUMENT']='coin_Cardano'
os.environ['CWCN_INSTRUMENT_HEAD']='Close,Open'
os.environ['CWCN_SEQUENCE_SIZE']='32'
os.environ['CWCN_RNN_HIDDEN_SIZE']='12'
os.environ['CWCN_FEATURES_SIZE']='16'
os.environ['CWCN_BATCH_SIZE']='3'
os.environ['CWCN_INPUT_SIZE']=str(len(os.environ['CWCN_INSTRUMENT_HEAD'].split(',')))
os.environ['CWCN_TRAIN_DATA_FOLDER']=os.environ['CWCN_HOME_FOLDER']+'/cripto_historic_kaggle'
# --- --- --- 
class RECURRENT_AUTOENCODER_KIJTYU(nn.Module):
    def __init__(self):
        super(RECURRENT_AUTOENCODER_KIJTYU, self).__init__()
        self.rnn1=nn.RNN(
            input_size=int(os.environ['CWCN_INPUT_SIZE']),
            hidden_size=int(os.environ['CWCN_RNN_HIDDEN_SIZE']),
            num_layers=2,
            nonlinearity='tanh',
            bias=False,
            batch_first=True,
            dropout=0.0,
            bidirectional=False
        )
        self.latent_space_layer1=torch.nn.Linear(
            in_features=int(os.environ['CWCN_RNN_HIDDEN_SIZE']), 
            out_features=int(os.environ['CWCN_FEATURES_SIZE']), 
            bias=False, 
            device=None, 
            dtype=None
        )
        self.feature_activation=torch.nn.Softsign()
        self.latent_space_layer2=torch.nn.Linear(
            in_features=int(os.environ['CWCN_FEATURES_SIZE']), 
            out_features=int(os.environ['CWCN_RNN_HIDDEN_SIZE']), 
            bias=False, 
            device=None, 
            dtype=None
        )
        self.rnn2=nn.RNN(
            input_size=int(os.environ['CWCN_RNN_HIDDEN_SIZE']),
            hidden_size=int(os.environ['CWCN_INPUT_SIZE']),
            num_layers=2,
            nonlinearity='tanh',
            bias=False,
            batch_first=True,
            dropout=0.0,
            bidirectional=False
        )
        self.w_h1_state=None
        self.w_h2_state=None
        self.init_all_parameters(torch.nn.init.normal_, mean=0., std=1) # self.init_all_parameters(torch.nn.init.constant_, 1.) # 
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=1.)
        clr = self.cyclical_lr(stepsize=20, min_lr=0.02, max_lr=0.5)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, [clr])
    def init_all_parameters(self, init_func, *params, **kwargs):
        for p in self.parameters():
            init_func(p, *params, **kwargs)
    def forward(self, x):
        # print("x(0)(input) : {}".format(x.shape))
        # x,self.w_h1_state=self.rnn1(x) #FIXME add hidden state
        # print("x(1)(rnn1) : {}".format(x.shape))
        # x=self.latent_space_layer1(x)
        # print("x(2)(latent1) : {}".format(x.shape))
        # x=self.feature_activation(x)
        # print("x(4)(feature_activation) : {}".format(x.shape))
        # features=x.mean(1)
        # print("x(meaned1) : {}".format(features.shape))
        # x=self.latent_space_layer2(x)
        # print("x(5)(latent2) : {}".format(x.shape))
        # x,self.w_h2_state=self.rnn2(x)
        # print("x(-1)(rnn2) : {}".format(x.shape))
        # return x,features
    def predict(self,_alliu):
        uwaabo,features=self(_alliu)
        return uwaabo,features
    def cyclical_lr(self,stepsize, min_lr=3e-4, max_lr=3e-3):
        # Scaler: we can adapt this if we do not want the triangular CLR
        scaler = lambda x: 1.
        # Lambda function to calculate the LR
        lr_lambda = lambda it: min_lr + (max_lr - min_lr) * relative(it, stepsize)
        # Additional function to see where on the cycle we are
        def relative(it, stepsize):
            cycle = np.floor(1 + it / (2 * stepsize))
            x = abs(it / stepsize - 2 * cycle + 1)
            return max(0, (1 - x)) * scaler(cycle)
        return lr_lambda
    def train(self,data_kijtyu):
        for d__ in data_kijtyu.__yield_batch__():
            self.optimizer.zero_grad()
            outputs,_ = self.predict(d__)
            loss = self.criterion(outputs, d__) # AutoEncoder is trained by matching the self
            logging.info("loss: {}".format(loss))
            loss.backward()
            self.scheduler.step()
            self.optimizer.step()
# --- --- --- 
class DATA_KIJTYU:
    def __init__(self):
        logging.info("Loading instrument historic data : "+os.environ['CWCN_TRAIN_DATA_FOLDER'])
        self.__exe_discriminator='csv'
        self.__folder=os.environ['CWCN_TRAIN_DATA_FOLDER']
        self.__data={}
        self.__reset_yields__()
        self.__load_folder__()
        self.__load_data__(os.environ['CWCN_INSTRUMENT'])
    def __reset_yields__(self):
        self.__c_index=[-int(os.environ['CWCN_SEQUENCE_SIZE']),0]
    def __load_folder__(self):
        __files_list=[os.path.join(self.__folder,_) for _ in os.listdir(self.__folder) if os.path.isfile(os.path.join(self.__folder,_)) and _.split('.')[-1]==self.__exe_discriminator]
        self.__files_dict=dict([(_.split('.')[-2].split('/')[-1],_) for _ in __files_list])
        self.__aviable_instruments_list=list(self.__files_dict.keys())
    def __load_data__(self,_idc):
        try:
            self.__data[_idc]=pd.read_csv(self.__files_dict[_idc])
            self.__instrument_data=self.__data[os.environ['CWCN_INSTRUMENT']][os.environ['CWCN_INSTRUMENT_HEAD'].split(',')]
            logging.info("Loaded <{}> data <{}>".format(self.__exe_discriminator,_idc))
            # logging.info(self.__data[_idc].head())
        except Exception as e:
            logging.error("Problem loading data <{}> : {}".format(_idc,e))
        return self.__data[_idc]
    def __yield_point__(self):
        while self.__c_index[1]<=self.__instrument_data.shape[0]-int(os.environ['CWCN_SEQUENCE_SIZE']):
            self.__c_index[0]+=int(os.environ['CWCN_SEQUENCE_SIZE'])
            self.__c_index[1]+=int(os.environ['CWCN_SEQUENCE_SIZE'])
            yield torch.FloatTensor(self.__instrument_data.iloc[self.__c_index[0]:self.__c_index[1]].values)
    def __yield_batch__(self): # yields ['batch_size','sequence_size','input_size]
        while(self.__c_index[1]<=self.__instrument_data.shape[0]-(int(os.environ['CWCN_SEQUENCE_SIZE']))*int(os.environ['CWCN_BATCH_SIZE'])):
            yield torch.stack([next(self.__yield_point__()) for _ in range(int(os.environ['CWCN_BATCH_SIZE']))])
# --- --- ---
data_kijtyu=DATA_KIJTYU()
# --- --- ---
wrae_kijtyu=RECURRENT_AUTOENCODER_KIJTYU()
# --- --- --- 
wrae_kijtyu.train(data_kijtyu)
# --- --- --- 
# print(len([_ for _ in data_kijtyu.__yield_batch__()]))
# print(91*int(os.environ['CWCN_SEQUENCE_SIZE'])*int(os.environ['CWCN_BATCH_SIZE']))
# print(91*int(os.environ['CWCN_SEQUENCE_SIZE'])*int(os.environ['CWCN_BATCH_SIZE'])-1374)
# print(next(data_kijtyu.__yield_point__()))
# print(next(data_kijtyu.__yield_batch__()))
# c_batch=next(data_kijtyu.__yield_batch__())
# print(c_batch)
# print(c_batch.shape)
# logging.info("Testing ... ")
# x=torch.FloatTensor([[[1.,1.]]])
# logging.info("input: {}".format(x))
# logging.info("input: {} -> {}".format(x,wrae_kijtyu.predict(x)))
# logging.info("input: {} -> {}".format(x.shape,wrae_kijtyu.predict(x).shape))

