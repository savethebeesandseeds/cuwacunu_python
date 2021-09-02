import os
import pandas as pd
import logging
# --- --- ---
os.environ['CWCN_ALL_COINS']='coin_Dogecoin, coin_Cardano, coin_XRP, coin_WrappedBitcoin, coin_Uniswap, coin_USDCoin, coin_Tron, coin_Tether, coin_Stellar, coin_Solana, coin_Polkadot, coin_NEM, coin_Monero, coin_Litecoin, coin_Iota, coin_Ethereum, coin_EOS, coin_CryptocomCoin, coin_Cosmos, coin_ChainLink, coin_Bitcoin, coin_BinanceCoin, coin_Aave'
os.environ['CWCN_COIN']='coin_Dogecoin'
os.environ['CWCN_COIN']='coin_Cardano'
os.environ['CWCN_COIN']='coin_XRP'
os.environ['CWCN_COIN']='coin_WrappedBitcoin'
os.environ['CWCN_COIN']='coin_Uniswap'
os.environ['CWCN_COIN']='coin_USDCoin'
os.environ['CWCN_COIN']='coin_Tron'
os.environ['CWCN_COIN']='coin_Tether'
os.environ['CWCN_COIN']='coin_Stellar'
os.environ['CWCN_COIN']='coin_Solana'
os.environ['CWCN_COIN']='coin_Polkadot'
os.environ['CWCN_COIN']='coin_NEM'
os.environ['CWCN_COIN']='coin_Monero'
os.environ['CWCN_COIN']='coin_Litecoin'
os.environ['CWCN_COIN']='coin_Iota'
os.environ['CWCN_COIN']='coin_Ethereum'
os.environ['CWCN_COIN']='coin_EOS'
os.environ['CWCN_COIN']='coin_CryptocomCoin'
os.environ['CWCN_COIN']='coin_Cosmos'
os.environ['CWCN_COIN']='coin_ChainLink'
os.environ['CWCN_COIN']='coin_Bitcoin'
os.environ['CWCN_COIN']='coin_BinanceCoin'
os.environ['CWCN_COIN']='coin_Aave' 
os.environ['CWCN_WLOT_FOLDER']='./gauss_dumps'
os.environ['CWCN_TRAIN_DATA_FOLDER']='/home/waajacu/work/cuwacunu.waajacu/cuwacunu_python/cripto_historic_kaggle'
# --- --- ---
training_iter = 100
learning_rate = 0.1
# --- --- 
c_horizon=50
c_horizon_delta=0.5
c_iterations=0xFFFF
c_backlash=-0.85
# --- --- ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] :: %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
# --- --- ---
def assert_folder(_f_path):
    if(not os.path.isdir(_f_path)):
        os.mkdir(_f_path)
# --- --- ---
class DATA_KIJTYU:
    def __init__(self,_folder):
        logging.info("Loading data from : "+_folder)
        self.__exe_discriminator='csv'
        self.__folder=_folder
        self._load_folder_()
        self.__data={}
        self.__data_size={}
        self.__c_idc=None
    def _load_folder_(self):
        __files_list=[os.path.join(self.__folder,_) for _ in os.listdir(self.__folder) if os.path.isfile(os.path.join(self.__folder,_)) and _.split('.')[-1]==self.__exe_discriminator]
        self.__files_dict=dict([(_.split('.')[-2].split('/')[-1],_) for _ in __files_list])
        self.__items_list=list(self.__files_dict.keys())
        # logging.info(self.__items_list)
    def _c_data_size_(self):
        return self.__data_size[self.__c_idc]
    def _load_data_(self,_idc):
        try:
            self.__c_idc=_idc
            self.__data[_idc]=pd.read_csv(self.__files_dict[_idc])
            logging.info("Loaded <{}> data <{}>".format(self.__exe_discriminator,_idc))
            if(pd.DataFrame(self.__data[_idc]['Close']).isnull().any().any()):
                logging.warning("Data <{}> has None values".format(_idc))
            # logging.info(self.__data[_idc].head())
            logging.info(self.__data[_idc]['Close'])
        except Exception as e:
            logging.error("Problem loading data <{}> : {}".format(_idc,e))
        self.__data_size[_idc]=len(self.__data[_idc])
        return self.__data[_idc]
# --- --- ---
import math
import torch
import gpytorch
import matplotlib.pyplot as plt
# this is for running the notebook in our testing framework
import os
import imageio

# --- --- ---

# Training data is 100 points in [0,1] inclusive regularly spaced
# train_x = torch.linspace(0, 1, 10)
# True function is sin(2*pi*x) with Gaussian noise

# train_y = torch.pow(torch.sin(train_x) * (36 * math.pi),0.1)*torch.tanh(train_x * (16 * math.pi)) + torch.randn(train_x.size()) * math.sqrt(0.0004)

# We will use the simplest form of GP model, exact inference
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        # self.mean_module = gpytorch.means.LinearMean(1)
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GAUSSIAN_WIKIMYEI:
    def __init__(self):
        # initialize likelihood and model
        self.model = None
    def _set_jk_optimizer_(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)  # Includes GaussianLikelihood parameters
    def _set_knowledge_base_(self,jkimyei_x,jkimyei_y):
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.model = ExactGPModel(jkimyei_x, jkimyei_y, self.likelihood)
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        self._set_jk_optimizer_()
    def _jkimyei_(self,jkimyei_x,jkimyei_y):
        # Exact GPModel
        # if(self.model is None):
        self._set_knowledge_base_(jkimyei_x,jkimyei_y)
        # Use the adam optimizer
        # "Loss" for GPs - the marginal log likelihood
        # Find optimal model hyperparameters
        self.model.train()
        self.likelihood.train()
        for i_ in range(training_iter):
            # Zero gradients from previous iteration
            self.optimizer.zero_grad()
            # Output from model
            output = self.model(jkimyei_x)
            # Calc loss and backprop gradients
            loss = -self.mll(output, jkimyei_y)
            loss.backward()
            if(i_%50==0):
                logging.info('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                    i_, training_iter, loss.item(),
                    self.model.covar_module.base_kernel.lengthscale.item(),
                    self.model.likelihood.noise.item()
                ))
            self.optimizer.step()
    def _uwaabo_hash_(self,uwaabo_x):
        # f_preds = model(uwaabo_x)
        # y_preds = likelihood(model(uwaabo_x))
        # f_mean = f_preds.mean
        # f_var = f_preds.variance
        # f_covar = f_preds.covariance_matrix
        # f_samples = f_preds.sample(sample_shape=torch.Size(1000,))
        # Get into evaluation (predictive posterior) mode
        self.model.eval()
        self.likelihood.eval()
        # Test points are regularly spaced along [0,1]
        # Make predictions by feeding model through likelihood
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self.likelihood(self.model(uwaabo_x))
        return observed_pred
    
class GAUSSIAN_WLOTER:
    def __init__(self):
        self.itm=os.environ['CWCN_COIN']
        self.out_wlot_folder_itm=os.path.join(os.environ['CWCN_WLOT_FOLDER'],os.environ['CWCN_COIN'])
        self.itm_ctx=0
        # --- --- ---
    def __purge_wlot_folder__(self):
        assert_folder(self.out_wlot_folder_itm)
        logging.warning("Purging [folder] : <{}>".format(self.out_wlot_folder_itm))
        for f_ in os.listdir(self.out_wlot_folder_itm):
            ff_=os.path.join(self.out_wlot_folder_itm,f_)
            logging.warning("\t - Purging [file] : <{}>".format(ff_))
            os.remove(ff_)
    def __wlot_gif__(self):
        imags=[]
        for f_ in os.listdir(self.out_wlot_folder_itm):
            ff_=os.path.join(self.out_wlot_folder_itm,f_)
            imags.append(imageio.imread(ff_))
            logging.info("\t - Wppending \t{}\t[GIF] : <{}>".format(self.itm,ff_))
        gf_path=os.path.join(self.out_wlot_folder_itm,"{}.{}".format(self.itm,"gif"))
        imageio.mimsave(gf_path,imags, format='GIF', duration=1)
    def __wlot_graph__(self,uwaabo_x,uwaabo_y,jkimyei_x,jkimyei_y,truth_x,truth_y):
        with torch.no_grad():
            # Initialize plot
            f, ax = plt.subplots(1, 1)
            # f.canvas.manager.full_screen_toggle()
            f.patch.set_facecolor((0,0,0))
            ax.set_title("{} - {}".format(ax.get_title(),os.environ['CWCN_COIN']),color=(1,1,1))
            ax.set_facecolor((0,0,0))
            ax.spines['bottom'].set_color('white')
            ax.spines['top'].set_color('white')
            ax.spines['right'].set_color('white')
            ax.spines['left'].set_color('white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.tick_params(colors='white',which='both')
            # Get upper and lower confidence bounds
            lower, upper = uwaabo_y.confidence_region()
            # Plot future data
            ax.plot(truth_x.numpy(), truth_y.numpy(), 'g', linewidth=0.3)
            # ax.plot(truth_2_x.numpy(), truth_2_y.numpy(), 'y')
            # Plot predictive means as blue line
            ax.plot(uwaabo_x.numpy(), uwaabo_y.mean.numpy(), 'b', linewidth=1.0)
            # Plot training data as black stars
            ax.plot(jkimyei_x.numpy(), jkimyei_y.numpy(), 'w', linewidth=0.8)
            # Shade between the lower and upper confidence bounds
            ax.fill_between(uwaabo_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.2)
            # ax.set_ylim([-3, 3])
            # ax.legend(['Kijtiyu Alliu, Uwaabo Mean, Unknown Alliu, Uwaabo Confidence'])
            # # plt.show()
            figname=os.path.join(self.out_wlot_folder_itm,"{}.{}.png".format(self.itm,self.itm_ctx))
            plt.savefig(figname, dpi=500, facecolor='w', edgecolor='w',
                orientation='portrait', format=None, transparent=False, 
                bbox_inches=None, pad_inches=0.1,metadata=None)
        self.itm_ctx+=1
# --- --- ---


# --- --- ---
if __name__=="__main__":
    # --- --- ---
    data_kijtyu=DATA_KIJTYU(_folder=os.environ['CWCN_TRAIN_DATA_FOLDER'])
    # --- --- ---
    for c_coin in os.environ['CWCN_ALL_COINS'].split(','):
        os.environ['CWCN_COIN']=c_coin.strip()
        # --- --- ---
        working_dataframe=data_kijtyu._load_data_(os.environ['CWCN_COIN'])
        # --- --- ---

        # --- --- ---
        truth_x=torch.linspace(0, data_kijtyu._c_data_size_(), data_kijtyu._c_data_size_())
        truth_y=torch.FloatTensor(list(working_dataframe['Close'][:]))
        # ---
        if(True):
            # --- --- ---
            gw=GAUSSIAN_WLOTER()
            gw.__purge_wlot_folder__()
            # --- --- ---
            gwk=GAUSSIAN_WIKIMYEI()
            _idx_=0
            while(int((1+c_backlash+c_horizon_delta+_idx_)*c_horizon)<data_kijtyu._c_data_size_()):
                if(c_iterations<_idx_):
                    break
                # --- 
                jkimyei_x=truth_x[int(((c_backlash if _idx_!=0 else 0)+_idx_)*c_horizon): int((1+c_backlash+_idx_)*c_horizon)]
                jkimyei_y=truth_y[int(((c_backlash if _idx_!=0 else 0)+_idx_)*c_horizon): int((1+c_backlash+_idx_)*c_horizon)]
                # --- 
                uwaabo_x=truth_x[int(((c_backlash if _idx_!=0 else 0)+_idx_)*c_horizon): int((1+c_backlash+c_horizon_delta+_idx_)*c_horizon)]
                # --- 
                gwk._set_knowledge_base_(jkimyei_x,jkimyei_y)
                gwk._jkimyei_(jkimyei_x,jkimyei_y)
                uwaabo_y=gwk._uwaabo_hash_(uwaabo_x)
                # print(uwaabo_y.sample())
                # --- 
                gw.__wlot_graph__(uwaabo_x,uwaabo_y,jkimyei_x,jkimyei_y,truth_x,truth_y)
                # --- --- ---
                _idx_+=1
            try:
                gw.__wlot_gif__()
            except:
                logging.error("[Unable to build gif]")
