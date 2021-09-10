# --- --- --- --- 
import os
import logging
import random
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
# assert(os.environ['CWCN_CONFIG']==os.path.realpath(__file__)), '[ERROR:] wrong configuration import'
# ... #FIXME assert comulative munaajpi is in place, seems ok, gae takes the account
# --- --- --- --- 
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# --- --- --- --- 
c_now=datetime.now()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] :: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('../logs/cuwcunu_log_file_{}-{}-{}-{}.log'.format(c_now.year,c_now.month,c_now.day,c_now.hour))
    ]
)
DEBUG_LEVELV_NUM = 99 
logging.addLevelName(DEBUG_LEVELV_NUM, 'RAY ')
def ray_log(message, *args, **kws):
    logging.Logger._log(logging.root,DEBUG_LEVELV_NUM, message, args, **kws)
DEBUG_LEVELV_NUM = 1 
logging.addLevelName(DEBUG_LEVELV_NUM, 'DEEPLOGGING ')
def deep_logging(message, *args, **kws):
    logging.Logger._log(logging.root,DEBUG_LEVELV_NUM, message, args, **kws)
logging.deep_logging = deep_logging
DEBUG_LEVELV_NUM = 1 
logging.addLevelName(DEBUG_LEVELV_NUM, 'TSANE')
def tsane_logging(message, *args, **kws):
    logging.Logger._log(logging.root,DEBUG_LEVELV_NUM, message, args, **kws)
logging.tsane_logging = tsane_logging
DEBUG_LEVELV_NUM = 1 
logging.addLevelName(DEBUG_LEVELV_NUM, '¡DANGER!')
def danger_logging(message, *args, **kws):
    logging.Logger._log(logging.root,DEBUG_LEVELV_NUM, "{}{}{}".format(CWCN_COLORS.DANGER,message,CWCN_COLORS.REGULAR), args, **kws)
logging.danger_logging = danger_logging
# --- --- --- --- 
# --- --- --- --- --- --- --- ---  
# --- --- --- --- 
# --- --- --- 
# --- --- 
# ---  
ALLOW_TSANE = True
PAPER_INSTRUMENT = True # < --- --- --- --- FAKE / REAL ; (bool) flag
random.seed()
# ---  
# --- --- 
# --- --- --- 
# --- --- --- --- 
# --- --- --- --- --- --- --- ---  
# --- --- --- --- 
class CWCN_DUURUVA_CONFIG:
    # --- --- --- 
    PLOT_LEVEL = 'mean,variance,value'
    DUURUVA_MAX_COUNT = 100
    DUURUVA_READY_COUNT = 50
    MIN_STD = 0.00001
    # --- --- --- 
    ENABLE_DUURUVA_IMU = False #FIXME agent learns to get bad imu to lower the mean
    # --- --- --- GAE/PPO
    NORMALIZE_IMU = True
    NORMALIZE_RETURNS = True # needed for munaajpi to train, but weird #FIXME
    NORMALIZE_ADVANTAGE = True
    # --- --- --- 
    NORMALIZE_ALLIU_PRICE = True
    NORMALIZE_ALLIU_SIZE = False
    NORMALIZE_ALLIU_SIDE = False
    NORMALIZE_ALLIU_PRICE_DELTA = True
    NORMALIZE_ALLIU_TIME_DELTA = True
    # --- --- --- 
# --- --- --- --- 
class CWCN_OPTIONS:
    PLOT_FLAG           = False
    RENDER_FLAG         = True
# --- --- --- --- 
class CWCN_INSTRUMENT_CONFIG:
    # --- --- --- 
    EXCHANGE = 'POLONIEX'
    SYMBOL = 'ADAUSDTPERP' #'BTCUSDTPERP'
    CURRENCY = 'USDT'
    # --- --- --- 
    MULTIPLER = 10
    CONTRACT_VALUE = 10 #ADA
    DELTA_COMMISSION = -0.1 # default is negative
    # --- --- --- 
# --- --- --- --- 
class CWCN_FARM_CONFIG:
    FARM_FOLDER = '../data_farm/FARM'
    FARM_SYMBOLS = [
        'BTCUSDTPERP',
        'ETHUSDTPERP',
        'BSVUSDTPERP',
        'BCHUSDTPERP',
        'YFIUSDTPERP',
        'UNIUSDTPERP',
        'LINKUSDTPERP',
        'TRXUSDTPERP',
        'XRPUSDTPERP',
        'XMRUSDTPERP',
        'LTCUSDTPERP',
        'DOTUSDTPERP',
        'DOGEUSDTPERP',
        'FILUSDTPERP',
        'BNBUSDTPERP',
        '1000SHIBUSDTPERP',
        'BTTUSDTPERP',
        'ADAUSDTPERP',
        'SOLUSDTPERP',
        'LUNAUSDTPERP',
        'ICPUSDTPERP',
        ]
    FARM_DATA_EXTENSION = '.poloniex_ticker_data'
# --- --- --- --- 
class CWCN_SIMULATION_CONFIG:
    # --- --- --- 
    INITIAL_WALLET={
        "availableBalance": 100.0,
        "realizedPnl": 0,
        "marginBalance": 100.0,
        "accountEquity":0.0,
        "positionMargin": 0,
        "orderMargin": 0,
        "frozenFunds": 0,
        "currency":CWCN_INSTRUMENT_CONFIG.CURRENCY,
    }
    CLOSE_ORDER_PROB = 0.5
    # --- --- --- 
    DATA_FOLDER = CWCN_FARM_CONFIG.FARM_FOLDER
    DATA_EXTENSION = CWCN_FARM_CONFIG.FARM_DATA_EXTENSION
    GET_HISTORY_LEN = 100 # amount of data pull when get_trade_history is call
    SKIP_N_DATA = 0
    # --- --- --- 
# --- --- --- --- 
class CWCN_UJCAMEI_CAJTUCU_CONFIG:
    # --- --- --- 
    WEB_SOCKET_SUBS = [
        '/contractAccount/wallet', # account updates
        '/contractMarket/ticker:{}'.format(CWCN_INSTRUMENT_CONFIG.SYMBOL), # price information in real time
        # '/contract/instrument:{}'.format(CWCN_INSTRUMENT_CONFIG.SYMBOL), # aditional market info (mark price)
        # '/contractMarket/execution:{}'.format(CWCN_INSTRUMENT_CONFIG.SYMBOL), # recieve the order feedback (redundant to ticker?)
        # '/contractMarket/level2:{}'.format(CWCN_INSTRUMENT_CONFIG.SYMBOL),
        # '/contractMarket/level2:{}'.format(CWCN_INSTRUMENT_CONFIG.SYMBOL),
    ]
    # --- --- --- 
    TIME_DECREMENTAL_SEQUENCE = False # does the sequence input to the recurrent have in position [0] the most recent time stamp and [-1] the older (default to False, RNN default?)
    TSANE_ACTION_DICT = {
        0:'put',
        1:'pass',
        2:'call'
    }
    ALLIU_LEN = 6
    # --- --- --- 
# --- --- --- --- 
class CWCN_CONFIG:
    def __init__(self):
        if(os.environ.get('CWCN_CONFIG_SYSTEM') is None or os.environ.get('CWCN_CONFIG_SYSTEM')=='default'):
            self._default_config_()
        elif(os.environ.get('CWCN_CONFIG_SYSTEM') == 'ray_system'):
            self._default_config_()
            self._ray_config_()
        else:
            assert(0x0),'[ERROR:] bad configuration'
    def _default_config_(self):
        # @property #FIXME implement or remove
        # --- --- 
        self.UJCAMEI_ALLIU_SEQUENCE_SIZE = 10 # size of alliu recurrent input buffer
        # --- --- 
        self.HIPER_PROFILE_BUFFER_COUNT = 3 # amount of trayetories queue in hold
        # --- --- 
        self.CHECKPOINTS_FOLDER  = os.path.normpath(os.path.join(os.path.realpath(__file__),'../checkpoints'))
        # --- --- 
        self.AHPA_ID            = 'MountainCarContinuous-v0'#'Pendulum-v0'#'MountainCarContinuous-v0'
        self.ALLIU_COUNT        = torch.Size([2]).numel() #FIXME is not parametric
        self.TSANE_COUNT        = torch.Size([1]).numel() #FIXME is not parametric
        # fixme add uwaabo
        self.IMU_COUNT          = torch.Size([2]).numel() #FIXME is not parametric
        # --- --- 
        self.TRAINING_EPOCHS     = 64 # amount of mapps from HIPER_PROFILE_BUFFER_COUNT
        self.AHDO_STEPS          = 256
        self.MINI_BATCH_COUNT    = 32 # lower due to training by .mean()
        self.NUM_TESTS           = 1
        self.VALIDATION_EPOCH    = 10 # how often to test the training # for standalone method
        self.BREAK_TRAIN_EPOCH   = 1000 # max amount of EPOCHS # for standalone method
        self.BREAK_TRAIN_IMU     = 0xFFFFFFFF
        # --- --- 
        self.TEHDUJCO_LEARNING_RATE       = 4e-4
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
        self.RAY_CHECKPOINTS_FOLDER  = os.path.normpath(os.path.join(os.path.realpath(__file__),'../ray_checkpoints'))
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
    GROSS = '\033[1m'
    DANGER = '\033[41m'
    UNDERLINE = '\033[4m'
class CWCN_CURSOR: #FIXME not in use
    UP=r'\x1b[{n}A'
    DOWN=r'\x1b[{n}B'
    LEFT=r'\x1b[{n}D'
    RIGHT=r'\x1b[{n}C'
    CLEAR_LINE=r'\033[K'
    CARRIER_RETURN=r'\r'
# --- --- --- --- 
# logging.info('Loading configuration file {}'.format(os.environ['CWCN_CONFIG']))
logging.info('Loading configuration file {}'.format(os.path.realpath(__file__)))
# --- --- --- --- 
