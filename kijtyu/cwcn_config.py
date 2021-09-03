# --- --- --- --- 
import os
import logging
# --- --- --- --- 
import torch
import time
from datetime import datetime, timedelta
import numpy as np
# --- --- --- --- 
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
# --- --- --- --- 
torch.distributions.Distribution.set_default_validate_args(True)
# --- --- --- --- 
# assert(os.environ['CWCN_CONFIG']==os.path.realpath(__file__)), "[ERROR:] wrong configuration import"
# ... #FIXME assert comulative munaajpi is in place, seems ok, gae takes the account
# --- --- --- --- 
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
# --- --- --- --- 
c_now=datetime.now()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] :: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("../logs/cuwcunu_log_file_{}-{}-{}-{}.log".format(c_now.year,c_now.month,c_now.day,c_now.hour))
    ]
)
DEBUG_LEVELV_NUM = 99 
logging.addLevelName(DEBUG_LEVELV_NUM, "RAY ")
def ray_log(message, *args, **kws):
    logging.Logger._log(logging.root,DEBUG_LEVELV_NUM, message, args, **kws) 
logging.ray_log = ray_log
# --- --- --- --- 
class CWCN_DUURUVA_CONFIG:
    DUURUVA_MAX_COUNT = 100
    READY_COUNT = 10
    MIN_STD = 0.005
    NORMALIZE_IMU = False #FIXME agent learns to get bad imu to lower the mean
    ENABLE_DUURUVA_IMU = False #FIXME agent learns to get bad imu to lower the mean
# --- --- --- --- 
class CWCN_OPTIONS:
    PLOT_FLAG           = False
    RENDER_FLAG         = True
# --- --- --- --- 

# --- --- --- --- 
class CWCN_CONFIG:
    def __init__(self):
        if(os.environ.get('CWCN_CONFIG_SYSTEM') is None or os.environ.get('CWCN_CONFIG_SYSTEM')=='default'):
            self._default_config_()
        elif(os.environ.get('CWCN_CONFIG_SYSTEM') == 'ray_system'):
            self._default_config_()
            self._ray_config_()
        else:
            assert(0x0),"[ERROR:] bad configuration"
    def _default_config_(self):
        # @property #FIXME implement or remove
        # --- --- 
        self.CHECKPOINTS_FOLDER  = os.path.normpath(os.path.join(os.path.realpath(__file__),"../checkpoints"))
        # --- --- 
        self.AHPA_ID              = "MountainCarContinuous-v0"#"Pendulum-v0"#"MountainCarContinuous-v0"
        self.ALLIU_COUNT        = torch.Size([2]).numel() #FIXME is not parametric
        self.TSANE_COUNT        = torch.Size([1]).numel() #FIXME is not parametric
        # fixme add uwaabo
        self.IMU_COUNT          = torch.Size([2]).numel() #FIXME is not parametric
        # --- --- 
        self.TRAINING_STEPS           = 256
        self.TRAINING_EPOCHS          = 64
        self.MINI_BATCH_COUNT     = 64
        self.NUM_TESTS           = 3
        self.VALIDATION_EPOCH         = 10 # how often to test the training # for standalone method
        self.BREAK_TRAIN_EPOCH   = 1000 # max amount of EPOCHS # for standalone method
        self.BREAK_TRAIN_IMU       = 0xFFFFFFFF
        # --- --- 
        self.TEHDUJCO_LEARNING_RATE       = 4e-5
        self.TEHDUJCO_GAMMA               = 0.99
        self.TEHDUJCO_GAE_LAMBDA          = 0.95
        self.TEHDUJCO_IMU_BETA         = 0.1 # takes no effect when duuruva imu is active
        self.TEHDUJCO_ENTROPY_BETA        = 0.001
        self.ReferencesToNoMeButWhoThoseAllWhoMadeRechableTheImplementationOfThisAlgorithm_TEHDUJCO_EPSILON         = 0.2 # tehdujco mean thank you.
        # --- --- 
        # self.IITEPI_BETA         = 0.01
        self.UWAABO_BETA         = 0.01
        self.MUNAAJPI_BETA       = 0.01
        self.IMIBAJCHO_MAX            = 0.5 # loss
        self.IMIBAJCHO_MIN            =-0.5 # loss
        # --- --- 
        # --- ---
        self.UWAABO_HIDDEN_SIZE  = 16
        self.MUNAAJPI_HIDDEN_SIZE= 16
        # self.UWAABO_SHAPE  = (
        #         # Activation            # Layer
        #         (None,                  torch.nn.Linear(torch.Size([   self.ALLIU_COUNT     ]).numel(),torch.Size([   16                ]).numel())),
        #         (torch.nn.Softsign(),   torch.nn.Linear(torch.Size([   16                   ]).numel(),torch.Size([   16                ]).numel())),
        #         (torch.nn.Softsign(),   torch.nn.Linear(torch.Size([   16                   ]).numel(),torch.Size([   16                ]).numel())),
        #         (torch.nn.Softsign(),   torch.nn.Linear(torch.Size([   16                   ]).numel(),torch.Size([   16                ]).numel())),
        #         (torch.nn.Softsign(),   torch.nn.Linear(torch.Size([   16                   ]).numel(),torch.Size([   self.TSANE_COUNT  ]).numel()))
        # )
        # self.MUNAAJPI_SHAPE= (   
        #         # Activation            # Layer
        #         (None,                  torch.nn.Linear(torch.Size([   self.ALLIU_COUNT     ]).numel(),torch.Size([   16                ]).numel())),
        #         (torch.nn.Softsign(),   torch.nn.Linear(torch.Size([   16                   ]).numel(),torch.Size([   16                ]).numel())),
        #         (torch.nn.Softsign(),   torch.nn.Linear(torch.Size([   16                   ]).numel(),torch.Size([   16                ]).numel())),
        #         (torch.nn.Softsign(),   torch.nn.Linear(torch.Size([   16                   ]).numel(),torch.Size([   16                ]).numel())),
        #         (None,                  torch.nn.Linear(torch.Size([   16                   ]).numel(),torch.Size([   self.IMU_COUNT    ]).numel()))
        # )
        # self.IITEPI_HIDDEN_SIZE= torch.Size([32])
        # --- ---
        # --- ---
    def _ray_config_(self):
        # --- --- 
        self.RAY_CHECKPOINTS_FOLDER  = os.path.normpath(os.path.join(os.path.realpath(__file__),"../ray_checkpoints"))
        self.TEHDUJCO_RAY_N_TRAILS        = 5000
        # --- --- 
        self.TEHDUJCO_LEARNING_RATE       = tune.loguniform(1e-5, 1e-1,self.IMU_COUNT)
        self.TEHDUJCO_GAMMA               = tune.loguniform(1e-3, 1,self.IMU_COUNT)
        self.TEHDUJCO_GAE_LAMBDA          = tune.loguniform(1e-3, 1,self.IMU_COUNT)
        self.TEHDUJCO_IMU_BETA         = tune.loguniform(1e-5, 1e-1,self.IMU_COUNT)
        self.TEHDUJCO_ENTROPY_BETA        = tune.loguniform(1e-5, 1e-1,self.IMU_COUNT)
        # --- ---
        self.MUNAAJPI_BETA       = tune.loguniform(1e-5, 1,self.IMU_COUNT)
        self.UWAABO_BETA         = tune.loguniform(1e-5, 1,)
        self.UWAABO_HIDDEN_SIZE  = tune.sample_from(lambda _: 2**np.random.randint(2, 9))
        self.MUNAAJPI_HIDDEN_SIZE= tune.sample_from(lambda _: 2**np.random.randint(2, 9))
        # --- --- 
        # self.IITEPI_HIDDEN_SIZE=32
# --- --- --- --- 
class CWCN_COLORS:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    REGULAR = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
class CWCN_MOVE_CURSOR: #FIXME not in use
    up=r'\x1b[{n}A'
    down=r'\x1b[{n}B'
    right=r'\x1b[{n}C'
    left=r'\x1b[{n}D'
# --- --- --- --- 
# logging.info("Loading configuration file {}".format(os.environ['CWCN_CONFIG']))
logging.info("Loading configuration file {}".format(os.path.realpath(__file__)))
# --- --- --- --- 
