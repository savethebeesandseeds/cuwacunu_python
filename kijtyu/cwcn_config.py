# --- --- --- --- 
# cwcn_config
# --- --- --- --- 
import os
import logging
from pickle import TRUE
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
c_load_file=None
c_load_file=r'/home/waajacu/work/cuwacunu.waajacu/cuwacunu_python/kijtyu/checkpoints/alwayssaving.UjcameiCajtucu-v0.wkymodel'
# --- --- --- --- 
# assert(os.environ['CWCN_CONFIG']==os.path.realpath(__file__)), '[ERROR:] wrong configuration import'
# ... #FIXME assert comulative munaajpi is in place, seems ok, gae takes the account
# --- --- --- --- 
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# --- --- --- --- 
c_now=datetime.now()
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] :: %(message)s',#'%(asctime)s [%(levelname)s] :: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('../logs/cuwcunu_log_file_{}-{}-{}-{}.log'.format(c_now.year,c_now.month,c_now.day,c_now.hour))
    ]
)
# --- --- 
DEBUG_LEVELV_NUM_RAY = 1
logging.addLevelName(DEBUG_LEVELV_NUM_RAY, 'RAY ')
def ray_logging(message, *args, **kws):
    if(logging.root.level>=DEBUG_LEVELV_NUM_RAY):
        logging.Logger._log(logging.root,DEBUG_LEVELV_NUM_RAY, message, args, **kws)
logging.ray_logging = ray_logging
# --- --- 
DEBUG_LEVELV_NUM_DEEP = 30000 
logging.addLevelName(DEBUG_LEVELV_NUM_DEEP, 'DEEPLOGGING ')
def deep_logging(message, *args, **kws):
    if(logging.root.level>=DEBUG_LEVELV_NUM_DEEP):
        logging.Logger._log(logging.root,DEBUG_LEVELV_NUM_DEEP, message, args, **kws)
logging.deep_logging = deep_logging
# --- --- 
DEBUG_LEVELV_NUM_TSANE = 3
logging.addLevelName(DEBUG_LEVELV_NUM_TSANE, 'TSANE')
def tsane_logging(message, *args, **kws):
    if(logging.root.level>=DEBUG_LEVELV_NUM_TSANE):
        logging.Logger._log(logging.root,DEBUG_LEVELV_NUM_TSANE, message, args, **kws)
logging.tsane_logging = tsane_logging
# --- --- 
DEBUG_LEVELV_NUM_DANGER = 4
logging.addLevelName(DEBUG_LEVELV_NUM_DANGER, '¡DANGER!')
def danger_logging(message, *args, **kws):
    if(logging.root.level>=DEBUG_LEVELV_NUM_DANGER):
        logging.Logger._log(logging.root,DEBUG_LEVELV_NUM_DANGER, "{}{}{}".format(CWCN_COLORS.DANGER,message,CWCN_COLORS.REGULAR), args, **kws)
logging.danger_logging = danger_logging
# --- --- 
DEBUG_LEVELV_NUM_UJCAMEI = 20008
logging.addLevelName(DEBUG_LEVELV_NUM_UJCAMEI, 'UJCAMEI')
def ujcamei_logging(message, *args, **kws):
    if(logging.root.level>=DEBUG_LEVELV_NUM_UJCAMEI):
        logging.Logger._log(logging.root,DEBUG_LEVELV_NUM_UJCAMEI, message, args, **kws)
logging.ujcamei_logging = ujcamei_logging
# --- --- 
DEBUG_LEVELV_NUM_ORDERS = 200001
logging.addLevelName(DEBUG_LEVELV_NUM_ORDERS, 'ORDER')
def orders_logging(message, *args, **kws):
    if(logging.root.level>=DEBUG_LEVELV_NUM_ORDERS):
        logging.Logger._log(logging.root,DEBUG_LEVELV_NUM_ORDERS, message, args, **kws)
logging.orders_logging = orders_logging
# --- --- 
DEBUG_LEVELV_NUM_JKIMYEI = 200007
logging.addLevelName(DEBUG_LEVELV_NUM_JKIMYEI, 'JKIMYEI')
def jkimyei_logging(message, *args, **kws):
    if(logging.root.level>=DEBUG_LEVELV_NUM_JKIMYEI):
        logging.Logger._log(logging.root,DEBUG_LEVELV_NUM_JKIMYEI, message, args, **kws)
logging.jkimyei_logging = jkimyei_logging
# --- --- 
DEBUG_LEVELV_NUM_WIKIMYEI = 200001
logging.addLevelName(DEBUG_LEVELV_NUM_WIKIMYEI, 'WIKIMYEI')
def wikimyei_logging(message, *args, **kws):
    if(logging.root.level>=DEBUG_LEVELV_NUM_WIKIMYEI):
        logging.Logger._log(logging.root,DEBUG_LEVELV_NUM_WIKIMYEI, message, args, **kws)
logging.wikimyei_logging = wikimyei_logging
# --- --- 
DEBUG_LEVELV_NUM_TSINUU = 200001
logging.addLevelName(DEBUG_LEVELV_NUM_TSINUU, 'TSINUU')
def tsinuu_logging(message, *args, **kws):
    if(logging.root.level>=DEBUG_LEVELV_NUM_TSINUU):
        logging.Logger._log(logging.root,DEBUG_LEVELV_NUM_TSINUU, message, args, **kws)
logging.tsinuu_logging = tsinuu_logging
# --- --- 

# --- --- --- --- 
# --- --- --- --- --- --- --- ---  
PAPER_INSTRUMENT = True #FIXME # < --- --- --- --- FAKE / REAL ; (bool) flag
# --- --- --- --- 
# --- --- --- 
# --- --- 
SYMBOL_INSTRUMENT = 'BTCUSDTPERP' #'BCHUSDTPERP' #'BTCUSDTPERP' #'SINE-100'#'BTCUSDTPERP' #'ADAUSDTPERP'/'BTCUSDTPERP'
# ---  
GREEDY_TSANE_SAMPLE = True
ALLOW_TSANE = True # ALLOW_TSANE=True/PAPER_INSTRUMENT=False -> WARNING!, ALLOW_TSANE=False/PAPER_INSTRUMENT=any -> UNABLE TO TRAIN RL, ALLOW_TSANE=True/PAPER_INSTRUMENT=True -> SAFE!
ALLOW_TRAIN = True #FIXME, find that there are no delays when no train is allowed
# ---  
TRAIN_ON_RL = True # train the uwaabo/munaajpi
TRAIN_ON_FORECAST = False # train the FORECAST; important it is to train FORECAST, but forecast means to know the future, forecast training is only enable while PEPER_INSTUMENT=False from a data source File  
if(TRAIN_ON_FORECAST):assert(ALLOW_TRAIN and PAPER_INSTRUMENT)
assert(not ALLOW_TRAIN ^ (TRAIN_ON_FORECAST or TRAIN_ON_RL)) # something must be trainable
FORECAST_HORIZONS = [3,5,7,11] # in terms of ticks into the future 
assert(all(_>1 for _ in FORECAST_HORIZONS)), "[BAD CONFIGURATION] forecast = 1 makes no scence, and 0 is not allowed"
# ---  
STOP_TO_PLOT_EVERY = 100
STOP_LOSS_USDT = -100
# ---  
# ---  
# ---  
# ---  --- --- --- --- what?
CONTEXT_MARGINALIZED_TSINUU = True # good stuff
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
    DUURUVA_MAX_COUNT = 25
    DUURUVA_READY_COUNT = 25
    MIN_STD = 0.0001
    # --- --- --- 
    ENABLE_DUURUVA_IMU = False #FIXME agent learns to get bad imu to lower the mean
    # --- --- --- GAE/PPO
    NORMALIZE_IMU = False # write sometihng
    NORMALIZE_RETURNS = True # needed for munaajpi to train, but weird #FIXME
    NORMALIZE_ADVANTAGE = True # recommended by experts
    # --- --- --- 
    COMPUTE_ALLIU_PRICE = True # asert True when forecast training
    COMPUTE_ALLIU_SIZE = False
    COMPUTE_ALLIU_SIDE = False
    COMPUTE_ALLIU_PRICE_DELTA = True
    COMPUTE_ALLIU_TIME_DELTA = True
    COMPUTE_ALLIU_CURRENTqTY = False
    # --- --- --- 
# --- --- --- --- 
class CWCN_OPTIONS:
    COLOR_PALLETE=['red','white','yellow','green','purple','blue']
    PLOT_INTERVAL       = STOP_TO_PLOT_EVERY
    PLOT_FLAG           = True
    RENDER_FLAG         = False
    AHDO_PLOT_SETS=['imu,returns,value,price','imu,price,put_certainty,pass_certainty,call_certainty']#alliu:0, forecast_non_uwaabo
    LEARNING_PLOT_SETS=['uwaabo_imibajcho,munaajpi_imibajcho,tsane_imibajcho,forecast_imibajcho']
# --- --- --- --- 
class CWCN_INSTRUMENT_CONFIG:
    # --- --- --- 
    EXCHANGE = 'POLONIEX' #FIXME not used
    SYMBOL = SYMBOL_INSTRUMENT #FIXME functions refer to CWCN_INSTURMENT_CONF insted of symbol_instrumnet directly, 
    CURRENCY = 'USDT'
    LEVERAGE="50"
    # --- --- --- 
    MULTIPLER = 10/int(LEVERAGE)
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
        "availableBalance": 10.0,
        "realisedPnl": 0.0,
        "unrealisedPnl":0.0,
        "marginBalance": 10.0,
        "accountEquity":0.0,
        "positionMargin": 0.0,
        "orderMargin": 0.0,
        "frozenFunds": 0.0,
        "currency":CWCN_INSTRUMENT_CONFIG.CURRENCY,
    }
    CLOSE_ORDER_PROB = 1.0
    # --- --- --- 
    DATA_FOLDER = CWCN_FARM_CONFIG.FARM_FOLDER
    # DATA_FOLDER = '../data_farm/FUNC'

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
    # --- --- --- 
    # --- --- --- --- --- --- 
    CERTAINTY_FACTOR=0.0 # porcentaje de certeza en interválo [0,1]
    AMBIVALENCE_FACTOR=0.0#0.24 # diffenrence between buy and sell for such of be not pass
    # --- --- --- 
    MAX_POSITION_SIZE=2 # max amount of contracts
    TSANE_MIN_PROB_MARGIN=25
    # --- --- --- 
    # --- --- --- --- --- --- 
    # --- --- --- 
    IMU_COUNT   = 1 #FIXed
    TSANE_COUNT = len(list(TSANE_ACTION_DICT.keys()))
    UJCAMEI_ALLIU_SEQUENCE_SIZE = 24 # size of alliu recurrent input buffer
    # --- --- --- 
    UJCAMEI_ACTIVE_BAO = ['price', 'price_delta', 'time_delta']
    ALLIU_COUNT = len(UJCAMEI_ACTIVE_BAO)
    UJCAMEI_BAO = {
        # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        #                # price 
        'price.compute_flag':True,
        'price':(lambda pos, wal, tk_dict, past_tk_dict : tk_dict['price']),
        'price.tensor':(lambda tk_dict:torch.Tensor([tk_dict['price']]).squeeze(0)),
        'price.std_or_norm':'norm', # 'norm'/'std'/'not' : 'norm' is (x-mean)/(std) ; 'std' is x/std ; 'not' when duuruva wrapper returns just but x. ---a: see duuruva for references
        # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        #                # size 
        'size.compute_flag':True,
        'size':(lambda pos, wal, tk_dict, past_tk_dict : tk_dict['size']),
        'size.tensor':(lambda tk_dict:torch.Tensor([tk_dict['size']]).squeeze(0)),
        'size.std_or_norm':'std', # 'norm'/'std'/'not' : 'norm' is (x-mean)/(std) ; 'std' is x/std ; 'not' when duuruva wrapper returns just but x.  ---a: see duuruva for references
        # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        #                # side 
        'side.compute_flag':True,
        'side':(lambda pos, wal, tk_dict, past_tk_dict : -1 if tk_dict['side']=='sell' else +1 if tk_dict['side']=='buy' else 0),
        'side.tensor':(lambda tk_dict:torch.Tensor([-1 if tk_dict['side']=='sell' else +1 if tk_dict['side']=='buy' else 0]).squeeze(0)),
        'side.std_or_norm':'not', # 'norm'/'std'/'not' : 'norm' is (x-mean)/(std) ; 'std' is x/std ; 'not' when duuruva wrapper returns just but x.  ---a: see duuruva for references
        # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        #                # time_delta 
        'time_delta.compute_flag':True,
        'time_delta':(lambda pos, wal, tk_dict, past_tk_dict : tk_dict['ts']-past_tk_dict['ts']),
        'time_delta.tensor':(lambda tk_dict:torch.Tensor([tk_dict['time_delta']]).squeeze(0)),
        'price_delta.std_or_norm':'std', # 'norm'/'std'/'not' : 'norm' is (x-mean)/(std) ; 'std' is x/std ; 'not' when duuruva wrapper returns just but x.   ---a: see duuruva for references
        # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        #                # price_delta 
        'price_delta.compute_flag':True,
        'price_delta':(lambda pos, wal, tk_dict, past_tk_dict : tk_dict['price']-past_tk_dict['price']),
        'price_delta.tensor':(lambda tk_dict: torch.Tensor([tk_dict['price_delta']]).squeeze(0)),
        'time_delta.std_or_norm':'std', # 'norm'/'std'/'not' : 'norm' is (x-mean)/(std) ; 'std' is x/std ; 'not' when duuruva wrapper returns just but x.---a: see duuruva for references
        # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        #                # ts 
        'ts.compute_flag':True,
        'ts':(lambda pos, wal, tk_dict, past_tk_dict : tk_dict['ts']),
        'ts.tensor':(lambda tk_dict: torch.Tensor([tk_dict['ts']]).squeeze(0)),
        'ts.std_or_norm':'not', # 'norm'/'std'/'not' : 'norm' is (x-mean)/(std) ; 'std' is x/std ; 'not' when duuruva wrapper returns just but x.---a: see duuruva for references
        # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        #                # sequence #FIXME what!
        #.'compute_flag':True,
        # 'sequence':(lambda pos, wal, tk_dict, past_tk_dict : tk_dict['sequence']),
        # 'sequence.tensor':(lambda : None),
        # ...'sequence.std_or_norm':'norm', # 'norm'/'std'/'not' : 'norm' is (x-mean)/(std) ; 'std' is x/std ; 'not' when duuruva wrapper returns just but x.---a: see duuruva for references
        # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        #                # currentQty 
        'currentQty.compute_flag':True,
        'currentQty':(lambda pos, wal, tk_dict, past_tk_dict : pos.currentQty),
        'currentQty.tensor':(lambda tk_dict: torch.Tensor([tk_dict['currentyQty']]).squeeze(0)),
        'currentQty.std_or_norm':'norm', # 'norm'/'std'/'not' : 'norm' is (x-mean)/(std) ; 'std' is x/std ; 'not' when duuruva wrapper returns just but x.---a: see duuruva for references
        # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        #FIXME missing some messures for non_uawabo_forecast
    }
    # --- --- --- 
    POSITION_UPDATE_METHODS = {}
    WALLET_UPDATE_METHODS = {
        'stop_loss': (lambda x : x.uc._clear_positions_() if x.unrealisedPnl is not None and x.unrealisedPnl < STOP_LOSS_USDT else None)
    }
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
        self.APPEND_AHDO_QUEUE_PROFILE = 'imu'
        self.HIPER_PROFILE_BUFFER_COUNT = 3 # amount of trayetories queue in hold, default to 1 means train only on the last episode
        # --- --- 
        # --- --- -------- ---- ---- -- -- --- 
        self.TEHDUJCO_LEARNING_RATE       = 4e-5
        self.AHDO_STEPS          = 2**12
        # --- --- 
        # fixme add uwaabo
        # --- --- 
        self.TRAINING_EPOCHS     = 5 # int(1//self.TEHDUJCO_LEARNING_RATE) # amount of mapps from HIPER_PROFILE_BUFFER_COUNT
        self.MINI_BATCH_COUNT    = int(max(self.AHDO_STEPS//128,1)) # lower due to training by .mean()
        self.TEST_STEPS          = int(max(self.AHDO_STEPS,24))
        self.NUM_TESTS           = 1 # maybe wome fansy pants prime
        self.VALIDATION_EPOCH    = 10000 # how often to test the training # for standalone method
        self.BREAK_TRAIN_EPOCH   = 10000 # max amount of EPOCHS # for standalone method
        self.BREAK_TRAIN_IMU     = 0xFFFFFFFF # reward limit that delivers the sign of complete training
        # --- --- 
        self.dropout_flag = False
        self.dropout_prob = 0.15
        # --- --- 
        self.TEHDUJCO_GAMMA               = 0.99
        self.TEHDUJCO_GAE_LAMBDA          = 0.95
        self.TEHDUJCO_IMU_BETA            = 1.0 # takes no effect when duuruva imu is active
        self.TEHDUJCO_ENTROPY_BETA        = 0.01
        self.ReferencesToNoMeButWhoThoseAllWhoMadeRechableTheImplementationOfThisAlgorithm_TEHDUJCO_EPSILON         = 0.2 # tehdujco mean thank you.
        # --- --- 
        # self.IITEPI_BETA         = 0.01
        self.FORECAST_BETA      = 1.0
        self.UWAABO_BETA         = 1.0
        self.MUNAAJPI_BETA       = 1.0
        self.IMIBAJCHO_MAX            = 5000 # loss
        self.IMIBAJCHO_MIN            =-5000 # loss
        # --- --- 
        # --- ---
        self.RECURRENT_TYPE = 'GRU' #['LSTM', 'GRU']
        self.RECURRENT_HIDEN_SIZE = 18
        self.RECURRENT_N_LAYERS = 4 #FIXME grow
        self.UWAABO_HIDDEN_SIZE  = 36
        self.FORECAST_HIDDEN_SIZE = 48
        self.FORECAST_N_HORIZONS = len(FORECAST_HORIZONS)
        self.MUNAAJPI_HIDDEN_SIZE = 16
        self.HIDDEN_BELLY_SIZE = 6
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
        # --- --- BROKEN
        assert(0xFF), 'bproken ray' #broken
        # # # self.ray_prtnt_flag = True
        # # # self.export_flag = True
        # # # self.close_at_finish =True
        # # # # --- --- 
        # # # self.RAY_CHECKPOINTS_FOLDER  = os.path.normpath(os.path.join(os.path.realpath(__file__),'../ray_checkpoints'))
        # # # self.RAY_N_TRAILS        = 5000
        # # # # --- --- 
        # # # self.TEHDUJCO_LEARNING_RATE       = tune.loguniform(1e-5, 1e-3,CWCN_UJCAMEI_CAJTUCU_CONFIG.IMU_COUNT)
        # # # self.TEHDUJCO_GAMMA               = tune.loguniform(1e-3, 1,CWCN_UJCAMEI_CAJTUCU_CONFIG.IMU_COUNT)
        # # # self.TEHDUJCO_GAE_LAMBDA          = tune.loguniform(1e-3, 1,CWCN_UJCAMEI_CAJTUCU_CONFIG.IMU_COUNT)
        # # # self.TEHDUJCO_IMU_BETA         = tune.loguniform(1e-5, 1e-1,CWCN_UJCAMEI_CAJTUCU_CONFIG.IMU_COUNT)
        # # # self.TEHDUJCO_ENTROPY_BETA        = tune.loguniform(1e-5, 1e-1,CWCN_UJCAMEI_CAJTUCU_CONFIG.IMU_COUNT)
        # # # # --- ---
        # # # self.UJCAMEI_ALLIU_SEQUENCE_SIZE = tune.sample_from(lambda _: 2**np.random.randint(2, 9))
        # # # # --- ---
        # # # self.MUNAAJPI_BETA       = tune.loguniform(1e-5, 1,CWCN_UJCAMEI_CAJTUCU_CONFIG.IMU_COUNT)
        # # # self.UWAABO_BETA         = tune.loguniform(1e-5, 1,)
        # # # self.RECURRENT_HIDEN_SIZE= tune.sample_from(lambda _: 2**np.random.randint(2, 9))
        # # # self.RECURRENT_N_LAYERS  = tune.sample_from(lambda _: 2**np.random.randint(1, 4))
        # # # self.UWAABO_HIDDEN_SIZE  = tune.sample_from(lambda _: 2**np.random.randint(2, 9))
        # # # self.MUNAAJPI_HIDDEN_SIZE= tune.sample_from(lambda _: 2**np.random.randint(2, 9))
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
    RED = '\033[31m'
    UNDERLINE = '\033[4m'
    PRICE = '\033[0;32m'
    YELLOW = '\033[1;33m'
    DARKGRAY = '\033[1;30m'
    GRAY = '\033[0;37m'
    WHITE = '\033[1;37m'
class CWCN_CURSOR: #FIXME not in use
    UP='\033[A'
    DOWN='\033[B'
    LEFT='\033[D'
    RIGHT='\033[C'
    CLEAR_LINE='\033[K'
    CARRIER_RETURN='\r'
# --- --- --- --- 
# logging.info('Loading configuration file {}'.format(os.environ['CWCN_CONFIG']))
logging.info('Loading from source the configuration file {}. '.format(os.path.realpath(__file__)))
# --- --- --- --- 
