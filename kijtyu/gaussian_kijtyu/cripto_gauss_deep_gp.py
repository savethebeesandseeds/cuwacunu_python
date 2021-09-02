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
import math
import torch
import gpytorch
import numpy as np
import tqdm
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution
from gpytorch.distributions import MultivariateNormal
from gpytorch.models import ApproximateGP, GP
from gpytorch.mlls import VariationalELBO, AddedLossTerm
from gpytorch.likelihoods import GaussianLikelihood

from gpytorch.models.deep_gps import DeepGPLayer, DeepGP
from gpytorch.mlls import DeepApproximateMLL


class ToyDeepGPHiddenLayer(DeepGPLayer):
    def __init__(self, input_dims, output_dims, num_inducing=128, mean_type='constant'):
        if output_dims is None:
            inducing_points = torch.randn(num_inducing, input_dims)
            batch_shape = torch.Size([])
        else:
            inducing_points = torch.randn(output_dims, num_inducing, input_dims)
            batch_shape = torch.Size([output_dims])
        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=num_inducing,
            batch_shape=batch_shape
        )
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )
        super(ToyDeepGPHiddenLayer, self).__init__(variational_strategy, input_dims, output_dims)
        if mean_type == 'constant':
            self.mean_module = ConstantMean(batch_shape=batch_shape)
        else:
            self.mean_module = LinearMean(input_dims)
        self.covar_module = ScaleKernel(
            RBFKernel(batch_shape=batch_shape, ard_num_dims=input_dims),
            batch_shape=batch_shape, ard_num_dims=None
        )
    def forward(self, x):
        mean_x = self.mean_module(x)
        print("mean_x")
        print(mean_x.shape)
        input()
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)
    def __call__(self, x, *other_inputs, **kwargs):
        """
        Overriding __call__ isn't strictly necessary, but it lets us add concatenation based skip connections
        easily. For example, hidden_layer2(hidden_layer1_outputs, inputs) will pass the concatenation of the first
        hidden layer's outputs and the input data to hidden_layer2.
        """
        if len(other_inputs):
            if isinstance(x, gpytorch.distributions.MultitaskMultivariateNormal):
                x = x.rsample()

            processed_inputs = [
                inp.unsqueeze(0).expand(gpytorch.settings.num_likelihood_samples.value(), *inp.shape)
                for inp in other_inputs
            ]

            x = torch.cat([x] + processed_inputs, dim=-1)

        return super().__call__(x, are_samples=bool(len(other_inputs)))



class DeepGP(DeepGP):
    def __init__(self, input_dim, output_dim):
        hidden_layer = ToyDeepGPHiddenLayer(
            input_dims=input_dim,
            output_dims=output_dim,
            mean_type='linear',
        )
        last_layer = ToyDeepGPHiddenLayer(
            input_dims=hidden_layer.output_dims,
            output_dims=None,
            mean_type='constant',
        )
        super().__init__()
        self.hidden_layer = hidden_layer
        self.last_layer = last_layer
    def forward(self, inputs):
        hidden_rep1 = self.hidden_layer(inputs)
        output = self.last_layer(hidden_rep1)
        return output


# --- --- --- 
class DEEP_GAUSSIAN_WIKIMYEI():
    def __init__(self):
        # --- ---
        self.num_data = 100
        self.num_input_dims = 1
        self.num_output_dims = 1
        self.num_epochs = 10
        self.num_samples = 10
        self.num_mini_batch = self.num_data//10
        # --- ---
        self.model = DeepGP(self.num_input_dims,self.num_output_dims)
        self.likelihood = GaussianLikelihood()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        # --- ---
        self.optimizer = torch.optim.Adam([
            {'params': self.model.parameters()},
        ], lr=0.01)
        self.mll = DeepApproximateMLL(VariationalELBO(self.likelihood, self.model, self.num_data))
        # --- ---
    def __predict__(self, alliu_batch):#, uwaabo_batch):
        # --- ---
        mus = []
        variances = []
        lls = []
        with torch.no_grad():
            preds = self.likelihood(self.model(alliu_batch))
            mus.append(preds.mean)
            variances.append(preds.variance)
            # lls.append(self.model.likelihood.log_marginal(uwaabo_batch, self.model(alliu_batch)))

        return torch.cat(mus, dim=-1), torch.cat(variances, dim=-1),None#, torch.cat(lls, dim=-1)
    def random_train_iter(self,_alliu,_uwaabo):
        # generates random mini-batches until we have covered the full batch
        for _ in range(self.num_data // self.num_mini_batch):
            rand_ids = np.random.randint(0, self.num_data, self.num_mini_batch)
            yield _alliu[rand_ids, :], _uwaabo[rand_ids, :]
    def sequential_train_iter(self,_alliu,_uwaabo):
        # generates random mini-batches until we have covered the full batch
        for _ in range(self.num_data // self.num_mini_batch):
            seq_ids = list(np.linspace((_)*self.num_mini_batch, (1+_)*self.num_mini_batch, self.num_mini_batch))
            yield _alliu[seq_ids, :], _uwaabo[seq_ids, :]
    def __train__(self,_alliu,_uwaabo):
        for i in range(self.num_epochs):
            # Within each iteration, we will go over each minibatch of data
            for alliu_batch, uwaabo_batch in self.random_train_iter(_alliu,_uwaabo):
                with gpytorch.settings.num_likelihood_samples(self.num_samples):
                    self.optimizer.zero_grad()
                    output,_aux,__aux = self.__predict__(alliu_batch)
                    print("uwaabo_batch : {}".format(uwaabo_batch.shape))
                    print("output : {}".format(output.shape))
                    input()
                    loss = -self.mll(output, uwaabo_batch)
                    loss.backward()
                    self.optimizer.step()
    # def __evaluate__(self):
    #     self.model.eval()
    #     predictive_means, predictive_variances, test_lls = self.__predict__(test_loader)
    #     rmse = torch.mean(torch.pow(predictive_means.mean(0) - test_y, 2)).sqrt()
    #     print(f"RMSE: {rmse.item()}, NLL: {-test_lls.mean().item()}")

    
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
            # ax.plot(uwaabo_x.numpy(), uwaabo_y.mean.numpy(), 'b', linewidth=1.0)
            ax.plot(uwaabo_x.numpy(), uwaabo_y.numpy(), 'b', linewidth=1.0)
            # Shade between the lower and upper confidence bounds
            # lower, upper = uwaabo_y.confidence_region()
            # ax.fill_between(uwaabo_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.2)
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
data_kijtyu=DATA_KIJTYU(_folder=os.environ['CWCN_TRAIN_DATA_FOLDER'])
working_dataframe=data_kijtyu.__load_data__(os.environ['CWCN_COIN'])
# --- --- ---
c_horizon=100
c_horizon_delta=6
# --- --- ---
truth_x=torch.linspace(0, 2*c_horizon, 2*c_horizon).unsqueeze(-1)
truth_y=torch.FloatTensor(list(working_dataframe['Close'][-2*c_horizon:])).unsqueeze(-1)
# ---

# --- --- ---
if __name__=="__main__":
    # --- --- ---
    gw=GAUSSIAN_WLOTER()
    gw.__purge_wlot_folder__()
    # --- --- ---
    gwk=DEEP_GAUSSIAN_WIKIMYEI()
    for _idx_ in range(10):
        # ---
        jkimyei_x=torch.linspace(_idx_*c_horizon_delta, c_horizon+_idx_*c_horizon_delta, c_horizon).unsqueeze(-1)
        jkimyei_y=torch.FloatTensor(list(working_dataframe['Close'][-2*c_horizon+_idx_*c_horizon_delta:-c_horizon+_idx_*c_horizon_delta])).unsqueeze(-1)
        # ---
        uwaabo_x=torch.linspace(c_horizon+(_idx_-1)*c_horizon_delta, c_horizon+(_idx_+1)*c_horizon_delta, 2*c_horizon_delta).unsqueeze(-1)
        # ---
        print("truth_x : {}".format(truth_x.shape))
        print("truth_y : {}".format(truth_y.shape))
        print("jkimyei_x : {}".format(jkimyei_x.shape))
        print("jkimyei_y : {}".format(jkimyei_y.shape))
        print("uwaabo_x : {}".format(uwaabo_x.shape))
        input()
        gwk.__train__(jkimyei_x,jkimyei_y)
        uwaabo_y_mu, uwaabo_y_var, _=gwk.__predict__(uwaabo_x)
        # --- --- ---
        gw.__wlot_graph__(uwaabo_x,uwaabo_y_mu,jkimyei_x,jkimyei_y,truth_x,truth_y)
    gw.__wlot_gif__()
