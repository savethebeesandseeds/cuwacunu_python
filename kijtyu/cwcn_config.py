# --- --- --- --- 
import os
import logging
import random
import torch
import time
from datetime import datetime, timedelta
import numpy as np
from ray import tune
# --- --- --- --- 
import cwcn_kemu_piaabo
# --- --- --- --- 
torch.distributions.Distribution.set_default_validate_args(True)
# --- --- --- --- 
c_load_file=r"/home/waajacu/work/cuwacunu.waajacu/cuwacunu_python/kijtyu/checkpoints/MountainCarContinuous-v0_best_+0.011.dat"
c_load_file=r"/home/waajacu/work/cuwacunu.waajacu/cuwacunu_python/kijtyu/checkpoints/MountainCarContinuous-v0_best_+0.000.dat"
c_load_file=r'/home/waajacu/work/cuwacunu.waajacu/cuwacunu_python/kijtyu/checkpoints/alwayssaving.UjcameiCajtucu-v0.wkymodel'
c_load_file=None
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
# --- --- 
DEBUG_LEVELV_NUM_RAY = 4
logging.addLevelName(DEBUG_LEVELV_NUM_RAY, 'RAY ')
def ray_logging(message, *args, **kws):
    if(logging.root.level>=DEBUG_LEVELV_NUM_RAY):
        logging.Logger._log(logging.root,DEBUG_LEVELV_NUM_RAY, message, args, **kws)
DEBUG_LEVELV_NUM_DEEP = 30000 
logging.ray_logging = ray_logging
# --- --- 
# --- --- 
logging.addLevelName(DEBUG_LEVELV_NUM_DEEP, 'DEEPLOGGING ')
def deep_logging(message, *args, **kws):
    if(logging.root.level>=DEBUG_LEVELV_NUM_DEEP):
        logging.Logger._log(logging.root,DEBUG_LEVELV_NUM_DEEP, message, args, **kws)
logging.deep_logging = deep_logging
DEBUG_LEVELV_NUM_TSANE = 2
# --- --- 
logging.addLevelName(DEBUG_LEVELV_NUM_TSANE, 'TSANE')
def tsane_logging(message, *args, **kws):
    if(logging.root.level>=DEBUG_LEVELV_NUM_TSANE):
        logging.Logger._log(logging.root,DEBUG_LEVELV_NUM_TSANE, message, args, **kws)
logging.tsane_logging = tsane_logging
DEBUG_LEVELV_NUM_DANGER = 1 
# --- --- 
logging.addLevelName(DEBUG_LEVELV_NUM_DANGER, '¡DANGER!')
def danger_logging(message, *args, **kws):
    if(logging.root.level>=DEBUG_LEVELV_NUM_DANGER):
        logging.Logger._log(logging.root,DEBUG_LEVELV_NUM_DANGER, "{}{}{}".format(CWCN_COLORS.DANGER,message,CWCN_COLORS.REGULAR), args, **kws)
logging.danger_logging = danger_logging
DEBUG_LEVELV_NUM_UJCAMEI = 200000 
# --- --- 
logging.addLevelName(DEBUG_LEVELV_NUM_UJCAMEI, 'UJCAMEI')
def ujcamei_logging(message, *args, **kws):
    if(logging.root.level>=DEBUG_LEVELV_NUM_UJCAMEI):
        logging.Logger._log(logging.root,DEBUG_LEVELV_NUM_UJCAMEI, message, args, **kws)
logging.ujcamei_logging = ujcamei_logging
DEBUG_LEVELV_NUM_ORDERS = 200001
# --- --- 
logging.addLevelName(DEBUG_LEVELV_NUM_ORDERS, 'ORDER')
def orders_logging(message, *args, **kws):
    if(logging.root.level>=DEBUG_LEVELV_NUM_ORDERS):
        logging.Logger._log(logging.root,DEBUG_LEVELV_NUM_ORDERS, message, args, **kws)
logging.orders_logging = orders_logging
DEBUG_LEVELV_NUM_JKIMYEI = 200001
# --- --- 
logging.addLevelName(DEBUG_LEVELV_NUM_JKIMYEI, 'JKIMYEI')
def jkimyei_logging(message, *args, **kws):
    if(logging.root.level>=DEBUG_LEVELV_NUM_JKIMYEI):
        logging.Logger._log(logging.root,DEBUG_LEVELV_NUM_JKIMYEI, message, args, **kws)
logging.jkimyei_logging = jkimyei_logging
DEBUG_LEVELV_NUM_WIKIMYEI = 200001
# --- --- 
logging.addLevelName(DEBUG_LEVELV_NUM_WIKIMYEI, 'WIKIMYEI')
def wikimyei_logging(message, *args, **kws):
    if(logging.root.level>=DEBUG_LEVELV_NUM_WIKIMYEI):
        logging.Logger._log(logging.root,DEBUG_LEVELV_NUM_WIKIMYEI, message, args, **kws)
logging.wikimyei_logging = wikimyei_logging
DEBUG_LEVELV_NUM_TSINUU = 200001
# --- --- 
logging.addLevelName(DEBUG_LEVELV_NUM_TSINUU, 'TSINUU')
def tsinuu_logging(message, *args, **kws):
    if(logging.root.level>=DEBUG_LEVELV_NUM_TSINUU):
        logging.Logger._log(logging.root,DEBUG_LEVELV_NUM_TSINUU, message, args, **kws)
logging.tsinuu_logging = tsinuu_logging
# --- --- 

# --- --- --- --- 
# --- --- --- --- --- --- --- ---  
# --- --- --- --- 
# --- --- --- 
# --- --- 
# ---  
ALLOW_TSANE = True
PAPER_INSTRUMENT = True # < --- --- --- --- FAKE / REAL ; (bool) flag
cwcn_kemu_piaabo.seed_everything(seed=min(8191,torch.seed()//300000000000))
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
    NORMALIZE_IMU = True # write sometihng
    NORMALIZE_RETURNS = True # needed for munaajpi to train, but weird #FIXME
    NORMALIZE_ADVANTAGE = True # recommended by experts
    # --- --- --- 
    NORMALIZE_ALLIU_PRICE = True
    NORMALIZE_ALLIU_SIZE = False
    NORMALIZE_ALLIU_SIDE = False
    NORMALIZE_ALLIU_PRICE_DELTA = True
    NORMALIZE_ALLIU_TIME_DELTA = True
    # --- --- --- 
# --- --- --- --- 
class CWCN_OPTIONS:
    PLOT_INTERVAL       = 5
    PLOT_FLAG           = True
    RENDER_FLAG         = False
# --- --- --- --- 
class CWCN_INSTRUMENT_CONFIG:
    # --- --- --- 
    EXCHANGE = 'POLONIEX'
    SYMBOL = 'BTCUSDTPERP' #'ADAUSDTPERP'/'BTCUSDTPERP'
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
        "unrealisedPNL":0,
        "marginBalance": 100.0,
        "accountEquity":0.0,
        "positionMargin": 0,
        "orderMargin": 0,
        "frozenFunds": 0,
        "currency":CWCN_INSTRUMENT_CONFIG.CURRENCY,
    }
    CLOSE_ORDER_PROB = 1.0
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
    ALLIU_COUNT = 6
    TSANE_COUNT = len(list(TSANE_ACTION_DICT.keys()))
    IMU_COUNT   = 1
    UJCAMEI_ALLIU_SEQUENCE_SIZE = 64 # size of alliu recurrent input buffer
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
        self.AHPA_ID             = 'UjcameiCajtucu-v0'#'MountainCarContinuous-v0'#'Pendulum-v0'#'MountainCarContinuous-v0'
        # --- --- 
        self.CHECKPOINTS_FOLDER  = os.path.normpath(os.path.join(os.path.realpath(__file__),'../checkpoints'))
        # --- --- 
        self.ALWAYS_SAVING_MODEL = True
        self.ALWAYS_SAVING_MODEL_PATH = os.path.join(self.CHECKPOINTS_FOLDER,'alwayssaving.{}.wkymodel'.format(self.AHPA_ID))
        # --- --- 
        # --- --- 
        self.HIPER_PROFILE_BUFFER_COUNT = 1 # amount of trayetories queue in hold, default to 1 means train only on the last episode
        # --- --- 
        # fixme add uwaabo
        # --- --- 
        self.TRAINING_EPOCHS     = 42 # amount of mapps from HIPER_PROFILE_BUFFER_COUNT
        self.AHDO_STEPS          = 256
        self.MINI_BATCH_COUNT    = 32 # lower due to training by .mean()
        self.NUM_TESTS           = 1
        self.VALIDATION_EPOCH    = 100 # how often to test the training # for standalone method
        self.BREAK_TRAIN_EPOCH   = 1000 # max amount of EPOCHS # for standalone method
        self.BREAK_TRAIN_IMU     = 0xFFFFFFFF
        # --- --- 
        self.dropout_flag = True
        self.dropout_prob = 0.05
        self.TEHDUJCO_LEARNING_RATE       = 4e-4
        self.TEHDUJCO_GAMMA               = 0.99
        self.TEHDUJCO_GAE_LAMBDA          = 0.95
        self.TEHDUJCO_IMU_BETA            = 0.1 # takes no effect when duuruva imu is active
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
        self.RECURRENT_TYPE = 'GRU'
        self.RECURRENT_HIDEN_SIZE = 32
        self.RECURRENT_N_LAYERS = 3 #FIXME grow
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
        #         (None,                  torch.nn.Linear(torch.Size([   16                   ]).numel(),torch.Size([   CWCN_UJCAMEI_CAJTUCU_CONFIG.IMU_COUNT    ]).numel()))
        # )
        # self.IITEPI_HIDDEN_SIZE= torch.Size([32])
        # --- ---
        # --- ---
    def _ray_config_(self):
        # --- ---
        self.ray_prtnt_flag = True
        self.export_flag = True
        self.close_at_finish =True
        # --- --- 
        self.RAY_CHECKPOINTS_FOLDER  = os.path.normpath(os.path.join(os.path.realpath(__file__),'../ray_checkpoints'))
        self.RAY_N_TRAILS        = 5000
        # --- --- 
        self.TEHDUJCO_LEARNING_RATE       = tune.loguniform(1e-5, 1e-3,CWCN_UJCAMEI_CAJTUCU_CONFIG.IMU_COUNT)
        self.TEHDUJCO_GAMMA               = tune.loguniform(1e-3, 1,CWCN_UJCAMEI_CAJTUCU_CONFIG.IMU_COUNT)
        self.TEHDUJCO_GAE_LAMBDA          = tune.loguniform(1e-3, 1,CWCN_UJCAMEI_CAJTUCU_CONFIG.IMU_COUNT)
        self.TEHDUJCO_IMU_BETA         = tune.loguniform(1e-5, 1e-1,CWCN_UJCAMEI_CAJTUCU_CONFIG.IMU_COUNT)
        self.TEHDUJCO_ENTROPY_BETA        = tune.loguniform(1e-5, 1e-1,CWCN_UJCAMEI_CAJTUCU_CONFIG.IMU_COUNT)
        # --- ---
        self.UJCAMEI_ALLIU_SEQUENCE_SIZE = tune.sample_from(lambda _: 2**np.random.randint(2, 9))
        # --- ---
        self.MUNAAJPI_BETA       = tune.loguniform(1e-5, 1,CWCN_UJCAMEI_CAJTUCU_CONFIG.IMU_COUNT)
        self.UWAABO_BETA         = tune.loguniform(1e-5, 1,)
        self.RECURRENT_HIDEN_SIZE= tune.sample_from(lambda _: 2**np.random.randint(2, 9))
        self.RECURRENT_N_LAYERS  = tune.sample_from(lambda _: 2**np.random.randint(1, 4))
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
    UP='\033[A'
    DOWN='\033[B'
    LEFT='\033[D'
    RIGHT='\033[C'
    CLEAR_LINE='\033[K'
    CARRIER_RETURN='\r'
# --- --- --- --- 
# logging.info('Loading configuration file {}'.format(os.environ['CWCN_CONFIG']))
logging.info('Loading configuration file {}'.format(os.path.realpath(__file__)))
# --- --- --- --- 
