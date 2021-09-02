import os
import pandas as pd
import logging
# --- --- ---
os.environ['CWCN_COIN']='coin_Cardano'
os.environ['CWCN_TRAIN_DATA_FOLDER']='/home/waajacu/work/cuwacunu.waajacu/cuwacunu_python/cripto_historic_kaggle'
# --- --- ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] :: %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
# --- --- ---
class DATA_KIJTYU:
    def __init__(self,_folder):
        logging.info("Loading data from : "+_folder)
        self.__exe_discriminator='csv'
        self.__folder=_folder
        self.__load_folder__()
        self.__data={}
    def __load_folder__(self):
        __files_list=[os.path.join(self.__folder,_) for _ in os.listdir(self.__folder) if os.path.isfile(os.path.join(self.__folder,_)) and _.split('.')[-1]==self.__exe_discriminator]
        self.__files_dict=dict([(_.split('.')[-2].split('/')[-1],_) for _ in __files_list])
        self.__items_list=list(self.__files_dict.keys())
        # logging.info(self.__items_list)
    def __load_data__(self,_idc):
        try:
            self.__data[_idc]=pd.read_csv(self.__files_dict[_idc])
            logging.info("Loaded <{}> data <{}>".format(self.__exe_discriminator,_idc))
            # logging.info(self.__data[_idc].head())
            # logging.info(self.__data[_idc]['Close'])
        except Exception as e:
            logging.error("Problem loading data <{}> : {}".format(_idc,e))
        return self.__data[_idc]
# --- --- ---
data_kijtyu=DATA_KIJTYU(_folder=os.environ['CWCN_TRAIN_DATA_FOLDER'])
working_dataframe=data_kijtyu.__load_data__(os.environ['CWCN_COIN'])
# --- --- ---
c_horizon=100
c_horizon_delta=6
# --- --- ---

import torch
import gpytorch
from matplotlib import pyplot as plt

data_dim = 1
training_iterations = 60
features_dim = 5

# --- --- ---
class LargeFeatureExtractor(torch.nn.Sequential):
    def __init__(self):
        super(LargeFeatureExtractor, self).__init__()
        self.add_module('linear1', torch.nn.Linear(data_dim, 100))
        self.add_module('relu1', torch.nn.ReLU())
        self.add_module('linear2', torch.nn.Linear(100, 50))
        self.add_module('relu2', torch.nn.ReLU())
        self.add_module('linear3', torch.nn.Linear(50, 50))
        self.add_module('relu3', torch.nn.ReLU())
        self.add_module('linear4', torch.nn.Linear(50, features_dim))
# --- --- ---
class GPRegressionModel(gpytorch.models.ExactGP):
        def __init__(self, jkimyei_x, jkimyei_y, likelihood,feature_extractor):
            super(GPRegressionModel, self).__init__(jkimyei_x, jkimyei_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.GridInterpolationKernel(
                gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=2)),
                num_dims=features_dim, grid_size=100
            )
            self.feature_extractor = feature_extractor
        def forward(self, x):
            # We're first putting our data through a deep net (feature extractor)
            # We're also scaling the features so that they're nice values
            projected_x = self.feature_extractor(x)
            projected_x = projected_x - projected_x.min(0)[0]
            projected_x = 2 * (projected_x / projected_x.max(0)[0]) - 1

            mean_x = self.mean_module(projected_x)
            covar_x = self.covar_module(projected_x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
# --- --- ---
# --- --- ---
class GAUSSIAN_DEEP_EXACT_WIKIMYEI:
    def __init__(self):
        self.feature_extractor = LargeFeatureExtractor()
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        if torch.cuda.is_available():
            self.feature_extractor = self.feature_extractor.cuda()
            self.likelihood = self.likelihood.cuda()
        # --- --- ---
    def __predict__(self,uwaabo_x):
        self.model.eval()
        self.likelihood.eval()
        return self.model(uwaabo_x)
    def __train__(self,jkimyei_x,jkimyei_y):
        self.model = GPRegressionModel(jkimyei_x, jkimyei_y, self.likelihood, self.feature_extractor)
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        # Find optimal model hyperparameters
        self.model.train()
        self.likelihood.train()
        # "Loss" for GPs - the marginal log likelihood
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        # Use the adam optimizer
        self.optimizer = torch.optim.Adam([
            {'params': self.feature_extractor.parameters()},
            {'params': self.model.covar_module.parameters()},
            {'params': self.model.mean_module.parameters()},
            {'params': self.likelihood.parameters()},
        ], lr=0.02)
        # --- --- ---
        for i in range(training_iterations):
            # Zero backprop gradients
            self.optimizer.zero_grad()
            # Get output from model
            output = self.model(jkimyei_x)
            # Calc loss and backprop derivatives
            loss = -self.mll(output, jkimyei_y)
            loss.backward()
            self.optimizer.step()
            logging.info('Test [{}] : MAE: {}'.format(i,torch.mean(torch.abs(output.mean - jkimyei_y))))
# --- --- ---
import imageio
import matplotlib.pyplot as plt
class GAUSSIAN_WLOTER:
    def __init__(self):
        self.itm=os.environ['CWCN_COIN']
        self.out_wlot_folder_itm='./gauss_gifs'
        self.itm_ctx=0
        # --- --- ---
    def __purge_wlot_folder__(self):
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
            logging.info("\t - Wppending [GIF] : <{}>".format(ff_))
        gf_path="./{}.{}".format(self.itm,"gif")
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
            # Plot future data
            ax.plot(truth_x.numpy(), truth_y.numpy(), 'g', linewidth=0.3)
            # Plot training data as black stars
            ax.plot(jkimyei_x.numpy(), jkimyei_y.numpy(), 'w', linewidth=0.8)
            # ax.plot(truth_2_x.numpy(), truth_2_y.numpy(), 'y')
            # Plot predictive means as blue line
            ax.plot(uwaabo_x.numpy(), uwaabo_y.mean.numpy(), 'b', linewidth=1.0)
            # ax.plot(uwaabo_x.numpy(), uwaabo_y.numpy(), 'b', linewidth=1.0)
            # Shade between the lower and upper confidence bounds
            lower, upper = uwaabo_y.confidence_region()
            ax.fill_between(uwaabo_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.2)
            # ax.set_ylim([-3, 3])
            # ax.legend(['Kijtiyu Alliu', 'Uwaabo Mean', 'Unknown Alliu', 'Uwaabo Confidence'])
            # # plt.show()
            figname=os.path.join(self.out_wlot_folder_itm,"{}.{}.png".format(self.itm,self.itm_ctx))
            plt.savefig(figname, dpi=500, facecolor='w', edgecolor='w',
                orientation='portrait', format=None, transparent=False, 
                bbox_inches=None, pad_inches=0.1,metadata=None)
        self.itm_ctx+=1
# --- --- ---

# --- --- ---

# --- --- ---
# ---

if __name__=="__main__":
    # --- --- ---
    truth_x=torch.linspace(0, 2*c_horizon, 2*c_horizon)#.unsqueeze(-1)
    truth_y=torch.FloatTensor(list(working_dataframe['Close'][-2*c_horizon:]))#.unsqueeze(-1)
    # --- --- ---
    gw=GAUSSIAN_WLOTER()
    gw.__purge_wlot_folder__()
    # --- --- ---
    gdew=GAUSSIAN_DEEP_EXACT_WIKIMYEI()
    for _idx_ in range(10):
        # ---
        jkimyei_x=torch.linspace(_idx_*c_horizon_delta, c_horizon+_idx_*c_horizon_delta, c_horizon)#.unsqueeze(-1)
        jkimyei_y=torch.FloatTensor(list(working_dataframe['Close'][-2*c_horizon+_idx_*c_horizon_delta:-c_horizon+_idx_*c_horizon_delta]))#.unsqueeze(-1)
        # ---
        uwaabo_x=torch.linspace(c_horizon+(_idx_-1)*c_horizon_delta, c_horizon+(_idx_+1)*c_horizon_delta, 2*c_horizon_delta)#.unsqueeze(-1)
        # ---
        logging.info("truth_x : {}".format(truth_x.shape))
        logging.info("truth_y : {}".format(truth_y.shape))
        logging.info("jkimyei_x : {}".format(jkimyei_x.shape))
        logging.info("jkimyei_y : {}".format(jkimyei_y.shape))
        logging.info("uwaabo_x : {}".format(uwaabo_x.shape))
        gdew.__train__(jkimyei_x,jkimyei_y)
        uwaabo_y=gdew.__predict__(uwaabo_x)
        # --- --- ---
        gw.__wlot_graph__(uwaabo_x,uwaabo_y,jkimyei_x,jkimyei_y,truth_x,truth_y)
    gw.__wlot_gif__()
