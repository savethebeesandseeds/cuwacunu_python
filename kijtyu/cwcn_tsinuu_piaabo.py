import torch
import torch.nn as nn
from torch.distributions import Normal
# --- --- ---
import cwcn_config
# --- --- --
# MEWAAJACURONE=0 #FIXME (desconocer) add to dictionary
# --- --- ---
class TSINUU_ACTOR_CRITIC(nn.Module):
    def __init__(self, alliu_size, uwaabo_size, UWAABO_HIDDEN_SIZE, MUNAAJPI_HIDDEN_SIZE, sigma=0.0): # alliu is state, uwabao is action
        # --- --- --- 
        super(TSINUU_ACTOR_CRITIC, self).__init__()
        # --- --- --- 
        self.ul_1=nn.Linear(alliu_size, UWAABO_HIDDEN_SIZE)
        self.ul_2=nn.Linear(UWAABO_HIDDEN_SIZE, UWAABO_HIDDEN_SIZE)
        self.ul_3=nn.Linear(UWAABO_HIDDEN_SIZE, UWAABO_HIDDEN_SIZE)
        self.ul_4=nn.Linear(UWAABO_HIDDEN_SIZE, UWAABO_HIDDEN_SIZE)
        # --- 
        self.ul_m1=nn.Linear(UWAABO_HIDDEN_SIZE, UWAABO_HIDDEN_SIZE)
        self.ul_m2=nn.Linear(UWAABO_HIDDEN_SIZE, uwaabo_size)
        # --- 
        self.ul_s1=nn.Linear(UWAABO_HIDDEN_SIZE, UWAABO_HIDDEN_SIZE)
        self.ul_s2=nn.Linear(UWAABO_HIDDEN_SIZE, uwaabo_size)
        # --- ---
        self.ml_1=nn.Linear(alliu_size, MUNAAJPI_HIDDEN_SIZE)
        self.ml_2=nn.Linear(MUNAAJPI_HIDDEN_SIZE, MUNAAJPI_HIDDEN_SIZE)
        self.ml_3=nn.Linear(MUNAAJPI_HIDDEN_SIZE, MUNAAJPI_HIDDEN_SIZE)
        self.ml_4=nn.Linear(MUNAAJPI_HIDDEN_SIZE, 1)
        # --- ---
        # self.iil_1=nn.Linear(alliu_size+uwaabo_size, IITEPI_HIDDEN_SIZE)
        # self.iil_2=nn.Linear(IITEPI_HIDDEN_SIZE, IITEPI_HIDDEN_SIZE)
        # self.iil_3=nn.Linear(IITEPI_HIDDEN_SIZE, alliu_size)
        # --- ---
        # self.mjl_1=nn.Linear(alliu_size, hidden_size)
        # self.mjl_2=nn.Linear(alliu_size, hidden_size)
        # self.mjl_3=nn.Linear(alliu_size, hidden_size)
        # --- ---
        self.min_sigma=torch.FloatTensor([0.001]).squeeze(0)
        self.sigma_gain=torch.FloatTensor([2.0]).squeeze(0)
        # self.log_sigma=nn.Parameter(torch.ones(uwaabo_size) * (sigma+self.min_sigma))
    def _foward_munaajpi_(self,x):
        x=nn.Softsign()(self.ml_1(x))
        x=nn.Softsign()(self.ml_2(x))
        x==self.ml_3(x)
        value=self.ml_4(x)
        return value
    def _foward_uwaabo_(self,x):
        x=self.ul_1(x)
        x=nn.Softsign()(x)
        x=nn.Softsign()(self.ul_2(x))
        x=nn.Softsign()(self.ul_3(x))
        x=nn.Softsign()(self.ul_4(x))
        # --- --- 
        mu=nn.Softsign()(self.ul_m1(x))
        mu=nn.Softsign()(self.ul_m2(mu))
        # sigma = self.log_sigma.exp().expand_as(mu) + self.min_sigma
        sigma=nn.Softsign()(self.ul_s1(x))
        sigma=nn.Softmax(-1)(self.ul_s2(sigma))*self.sigma_gain + self.min_sigma
        # # print(x.shape)
        try:
            dist  = ...Categorical...Normal(mu, sigma)
        except Exception as e:
            print("ERROR:",e)
            print("mu:",mu)
            print("sigma:",sigma)
            print(mu.shape,sigma.shape)
            raise e
        # out = torch.Categorical(probs)
        # out = torch.multinomial(dist.sample(), 1)
        return dist
    def _foward_iitepi_(self,x,a): # state, actions
        x=torch.cat((x,a),0)
        x=self.iil_1(x)
        x=nn.Softsign()(x)
        x=self.iil_2(x)
        x=nn.Softsign()(x)
        x=self.iil_3(x)
        n_state=nn.Softsign()(x)
        return n_state # next_state
    def forward(self, x):
        value=self._foward_munaajpi_(x)
        dist=self._foward_uwaabo_(x)
        return dist, value