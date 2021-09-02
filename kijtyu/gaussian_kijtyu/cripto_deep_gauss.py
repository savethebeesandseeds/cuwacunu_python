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
            # print(self.__data[_idc].head())
            # print(self.__data[_idc]['Close'])
        except Exception as e:
            logging.error("Problem loading data <{}> : {}".format(_idc,e))
        return self.__data[_idc]
# --- --- ---
import os
import torch
# import tqdm
import math
import gpytorch
from torch.nn import Linear
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution, \
    LMCVariationalStrategy
from gpytorch.distributions import MultivariateNormal
from gpytorch.models.deep_gps import DeepGPLayer, DeepGP
from gpytorch.mlls import DeepApproximateMLL, VariationalELBO
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings("ignore")

smoke_test = ('CI' in os.environ)
# --- --- ---
data_kijtyu=DATA_KIJTYU(_folder=os.environ['CWCN_TRAIN_DATA_FOLDER'])
working_dataframe=data_kijtyu.__load_data__(os.environ['CWCN_COIN'])
# --- --- ---
c_horizon=20

train_x=torch.linspace(0, c_horizon, c_horizon).unsqueeze(-1)
test_x= torch.linspace(c_horizon-c_horizon//2, c_horizon+c_horizon//2, c_horizon).unsqueeze(-1)

# truth_2_x= torch.linspace(0, 2*c_horizon, 2*c_horizon)


# train_y=torch.FloatTensor(list(working_dataframe['Close'][-2*c_horizon:-c_horizon]))
# truth_y=torch.FloatTensor(list(working_dataframe['Close'][-c_horizon:]))
# truth_2_y=torch.FloatTensor(list(working_dataframe['Close'][-2*c_horizon:]))


# train_x = torch.linspace(0, 1, 100)

train_y = torch.stack([
    torch.FloatTensor(list(working_dataframe['Close'][-2*c_horizon:-c_horizon])),
    torch.FloatTensor(list(working_dataframe['Open'][-2*c_horizon:-c_horizon])),
    torch.FloatTensor(list(working_dataframe['High'][-2*c_horizon:-c_horizon])),
    torch.FloatTensor(list(working_dataframe['Low'][-2*c_horizon:-c_horizon]))
], -1)


# --- --- ---
# Here's a simple standard layer

class DGPHiddenLayer(DeepGPLayer):
    def __init__(self, input_dims, output_dims, num_inducing=128, linear_mean=True):
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

        super().__init__(variational_strategy, input_dims, output_dims)
        self.mean_module = ConstantMean() if linear_mean else LinearMean(input_dims)
        self.covar_module = ScaleKernel(
            MaternKernel(nu=2.5, batch_shape=batch_shape, ard_num_dims=input_dims),
            batch_shape=batch_shape, ard_num_dims=None
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)
# --- --- ---
num_tasks = train_y.size(-1)
num_hidden_dgp_dims = 8

class MultitaskDeepGP(DeepGP):
    def __init__(self, train_x_shape):
        hidden_layer_1 = DGPHiddenLayer(
            input_dims=train_x_shape[-1],
            output_dims=num_hidden_dgp_dims,
            linear_mean=True
        )
        hidden_layer_2 = DGPHiddenLayer(
            input_dims=hidden_layer_1.output_dims,
            output_dims=num_hidden_dgp_dims,
            linear_mean=True
        )
        last_layer = DGPHiddenLayer(
            input_dims=hidden_layer_2.output_dims,
            output_dims=num_tasks,
            linear_mean=True
        )

        super().__init__()

        self.hidden_layer_1 = hidden_layer_1
        self.hidden_layer_2 = hidden_layer_2
        self.last_layer = last_layer

        # We're going to use a ultitask likelihood instead of the standard GaussianLikelihood
        self.likelihood = MultitaskGaussianLikelihood(num_tasks=num_tasks)

    def forward(self, inputs):
        hidden_rep1 = self.hidden_layer_1(inputs)
        hidden_rep2 = self.hidden_layer_2(hidden_rep1)
        output = self.last_layer(hidden_rep2)
        return output

    def predict(self, predict_x):
        with torch.no_grad():

            # The output of the model is a multitask MVN, where both the data points
            # and the tasks are jointly distributed
            # To compute the marginal predictive NLL of each data point,
            # we will call `to_data_independent_dist`,
            # which removes the data cross-covariance terms from the distribution.
            preds = model.likelihood(model(predict_x)).to_data_independent_dist()

        return preds.mean.mean(0), preds.variance.mean(0)


model = MultitaskDeepGP(train_x.shape)
# --- --- ---
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
mll = DeepApproximateMLL(VariationalELBO(model.likelihood, model, num_data=train_y.size(0)))

num_epochs = 1 if smoke_test else 200
# epochs_iter = tqdm.notebook.tqdm(range(num_epochs), desc="Epoch")
for i in range(num_epochs):
    optimizer.zero_grad()
    output = model(train_x)
    loss = -mll(output, train_y)
    # epochs_iter.set_postfix(loss=loss.item())
    loss.backward()
    optimizer.step()
# --- --- ---
# Make predictions
model.eval()
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    mean, var = model.predict(test_x)
    lower = mean - 2 * var.sqrt()
    upper = mean + 2 * var.sqrt()

# Plot results
fig, axs = plt.subplots(1,num_tasks)#figsize=(4 * num_tasks, 3)
fig.canvas.manager.full_screen_toggle()
fig.patch.set_facecolor((0,0,0))
for task, ax in enumerate(axs):
    # ax.set_title("{} - {}".format(ax.get_title(),os.environ['CWCN_COIN']),color=(1,1,1))
    ax.set_facecolor((0,0,0))
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.tick_params(colors='white',which='both')
    ax.plot(train_x.squeeze(-1).detach().numpy(), train_y[:, task].detach().numpy(), 'w')
    ax.plot(test_x.squeeze(-1).numpy(), mean[:, task].numpy(), 'b')
    ax.fill_between(test_x.squeeze(-1).numpy(), lower[:, task].numpy(), upper[:, task].numpy(), alpha=0.2)
    # ax.set_ylim([-3, 3])
    ax.legend(['Observed Data', 'Mean', 'Confidence'])
    ax.set_title(f'Task {task + 1}')
# fig.tight_layout(h_pad=0.2,w_pad=0.161,rect=(0.034,0.053,0.989,0.949))
plt.tight_layout(pad=0.2,h_pad=0.2,w_pad=0.161,rect=(0.034,0.053,0.989,0.949))
# --- --- ---
plt.show()
# --- --- ---
# --- --- ---
# --- --- ---
# --- --- ---