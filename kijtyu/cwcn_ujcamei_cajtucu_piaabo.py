# --- --- 
# cwcn_ujcamei_cajtucu_piaabo
# --- --- 
import sys
sys.path.append('../communications/')
# --- --- 
import torch
import time
import copy
import asyncio
import logging
# --- --- 
import rcsi_utils
import cwcn_config
import cwcn_duuruva_piaabo
import communications_config
import poloniex_api
import cwcn_simulation_kijtyu
# --- --- 
assert(cwcn_config.CWCN_INSTRUMENT_CONFIG.EXCHANGE=='POLONIEX'), "unrecognized exchange" # this is a valid assertion, till future arrives; waduka nagi pana quimio mismo.
# --- ---
class UJCAMEI_CAJTUCU_INSTRUMENT_REPRESENTATION:
    def __init__(self, _ujcamei_cajtucu):
        self.sequence_size = cwcn_config.CWCN_UJCAMEI_CAJTUCU_CONFIG.UJCAMEI_ALLIU_SEQUENCE_SIZE
        self.new_tick_aviable_on_queue = False
        self.uc=_ujcamei_cajtucu
        self.price_duuruva=cwcn_duuruva_piaabo.DUURUVA(_duuruva_vector_size=0x01, _d_name='price_duuruva'.upper(),
            _standar_or_normalize=cwcn_config.CWCN_DUURUVA_CONFIG.STD_OR_NORM_BAO['price'],
            _wrapper_duuruva_normalize=cwcn_config.CWCN_DUURUVA_CONFIG.NORMALIZE_ALLIU_PRICE)
        self.size_duuruva=cwcn_duuruva_piaabo.DUURUVA(_duuruva_vector_size=0x01, _d_name='size_duuruva'.upper(),
            _standar_or_normalize=cwcn_config.CWCN_DUURUVA_CONFIG.STD_OR_NORM_BAO['size'],
            _wrapper_duuruva_normalize=cwcn_config.CWCN_DUURUVA_CONFIG.NORMALIZE_ALLIU_SIZE) #Z
        self.side_duuruva=cwcn_duuruva_piaabo.DUURUVA(_duuruva_vector_size=0x01, _d_name='side_duuruva'.upper(),
            _standar_or_normalize=cwcn_config.CWCN_DUURUVA_CONFIG.STD_OR_NORM_BAO['side'],
            _wrapper_duuruva_normalize=cwcn_config.CWCN_DUURUVA_CONFIG.NORMALIZE_ALLIU_SIDE) #D
        self.price_delta_duuruva=cwcn_duuruva_piaabo.DUURUVA(_duuruva_vector_size=0x01, _d_name='price_delta_duuruva'.upper(),
            _standar_or_normalize=cwcn_config.CWCN_DUURUVA_CONFIG.STD_OR_NORM_BAO['price_delta'],
            _wrapper_duuruva_normalize=cwcn_config.CWCN_DUURUVA_CONFIG.NORMALIZE_ALLIU_PRICE_DELTA)
        self.time_delta_duuruva=cwcn_duuruva_piaabo.DUURUVA(_duuruva_vector_size=0x01, _d_name='time_delta_duuruva'.upper(),
            _standar_or_normalize=cwcn_config.CWCN_DUURUVA_CONFIG.STD_OR_NORM_BAO['time_delta'],
            _wrapper_duuruva_normalize=cwcn_config.CWCN_DUURUVA_CONFIG.NORMALIZE_ALLIU_TIME_DELTA)
        self.alliu_sequence_tensor=torch.zeros((self.sequence_size,cwcn_config.CWCN_UJCAMEI_CAJTUCU_CONFIG.ALLIU_COUNT)).to(cwcn_config.device)
        self.instrument_queue=[]
        if(cwcn_config.PAPER_INSTRUMENT and cwcn_config.TRAIN_ON_FORECAST):
            self._forecast_non_uwaabo=None
        self._price=None
    def _c_instrument_state_(self):
        if(cwcn_config.CWCN_UJCAMEI_CAJTUCU_CONFIG.TIME_DECREMENTAL_SEQUENCE):
            return self.instrument_queue[0]
        else:
            return self.instrument_queue[-1]
    def _update_alliu_sequence_(self, _tick_tensor):
        # _tick_tensor : [price, price_delta, time_delta, side, size]
        if(cwcn_config.CWCN_UJCAMEI_CAJTUCU_CONFIG.TIME_DECREMENTAL_SEQUENCE):
            self.alliu_sequence_tensor = torch.cat((_tick_tensor.unsqueeze(0),self.alliu_sequence_tensor[:-1])).to(cwcn_config.device)
            self.alliu_sequence_tensor[:+self.sequence_size]
        else:
            self.alliu_sequence_tensor = torch.cat((self.alliu_sequence_tensor[+1:],_tick_tensor.unsqueeze(0))).to(cwcn_config.device)
            self.alliu_sequence_tensor[-self.sequence_size:]
        return self.alliu_sequence_tensor
    def _instrument_queue_healt_(self):
        #FIXME add sequence continuity check
        _healt_flag=True
        if(cwcn_config.CWCN_UJCAMEI_CAJTUCU_CONFIG.TIME_DECREMENTAL_SEQUENCE):
            aux_ts=0xFFFFFFFFFFFFFFFF
            for _iq in self.instrument_queue:
                _healt_flag&=_iq['ts']<aux_ts
        else:
            aux_ts=0x00
            for _iq in self.instrument_queue:
                _healt_flag&=_iq['ts']>aux_ts
        if(not _healt_flag):
            logging.error("[HEALT] Unhealty load detected for UJCAMEI_CAJTUCU_INSTRUMENT_REPRESENTATION")
        self.uc._step_flgs['healt_checked']=True
        return _healt_flag
    def _load_instrument_representation_(self):
        c_max_count=max(self.sequence_size,cwcn_config.CWCN_DUURUVA_CONFIG.DUURUVA_READY_COUNT)
        c_instrument_history=self.uc._echange_instrument.market_instrument.get_trade_history(cwcn_config.CWCN_INSTRUMENT_CONFIG.SYMBOL)
        logging.info("[request] trade history : {}, brought size:{}".format(cwcn_config.CWCN_INSTRUMENT_CONFIG.SYMBOL, len(c_instrument_history)))
        c_instrument_history+=self.instrument_queue
        c_instrument_history=[_d1 for _n1,_d1 in enumerate(c_instrument_history) if _d1['ts'] not in [_d2['ts'] for _d2 in c_instrument_history[_n1+1:]]] # filter duplicates timestamp
        c_instrument_history=sorted(c_instrument_history,key=(lambda x: x['ts']), reverse=cwcn_config.CWCN_UJCAMEI_CAJTUCU_CONFIG.TIME_DECREMENTAL_SEQUENCE)
        # if(cwcn_config.CWCN_UJCAMEI_CAJTUCU_CONFIG.TIME_DECREMENTAL_SEQUENCE):
        #     self.instrument_queue=c_instrument_history[-self.sequence_size:]
        # else:
        #     self.instrument_queue=c_instrument_history[:+self.sequence_size]
        c_ctx=0
        for _iq in c_instrument_history:
            c_ctx+=1
            self._update_instrument_(_iq)
        while(c_ctx<c_max_count): #FIXME check the experiemnt when c_max_count >> 100
            c_ctx+=1
            logging.info("wait for queue to load : {}/{}".format(c_ctx,c_max_count))
            self.uc._wait_for_step_()
    def _update_instrument_(self,_tk_dict=None):
        if(_tk_dict is None and not cwcn_config.PAPER_INSTRUMENT):
            _tk_dict=self.uc._echange_instrument.market_instrument.get_ticker(cwcn_config.CWCN_INSTRUMENT_CONFIG.SYMBOL)
        # logging.info("_update_instrument_")
        # --- --- --- CAST
        _tk_dict['price']=float(_tk_dict['price'])
        _tk_dict['size']=int(_tk_dict['size'])
        _tk_dict['ts']=int(_tk_dict['ts'])
        # --- --- --- 
        if(self.new_tick_aviable_on_queue==True): # ticked arrived too fast for prossesing unit; accomulating tick
            logging.error("Not expected behaviour; double update or error state, inconclusive (appending method need to be FIX!)")
            # # if(cwcn_config.CWCN_UJCAMEI_CAJTUCU_CONFIG.TIME_DECREMENTAL_SEQUENCE):
            # #     c_index=0
            # # else:
            # #     c_index=-1
            # # self.instrument_queue[c_index]['price']=_tk_dict['price']
            # # self.instrument_queue[c_index]['size']+=_tk_dict['size'] #FIXME this is incorrect
            # # self.instrument_queue[c_index]['side']=_tk_dict['side']
            # # self.instrument_queue[c_index]['ts']=_tk_dict['ts']
            # # self.instrument_queue[c_index]['currentQty']=self.uc._uc_position_.currentQty
        self.new_tick_aviable_on_queue=True
        if(len(self.instrument_queue)!=0):
            if(cwcn_config.CWCN_UJCAMEI_CAJTUCU_CONFIG.TIME_DECREMENTAL_SEQUENCE):
                _past_tk_dict=self.instrument_queue[0]
            else:
                _past_tk_dict=self.instrument_queue[-1]
        else:
            _past_tk_dict=copy.deepcopy(_tk_dict)
        # --- --- --- --- --- --- --- --- --- --- --- --- good stuff
        # --- --- --- INTRUMENT QUEUE
        _temp_tk_dict=dict(
            [ cwcn_config.CWCN_UJCAMEI_CAJTUCU_CONFIG.UJCAMEI_BAO[_](
                pos=self.uc._uc_position_, 
                wal=self.uc._uc_wallet, 
                tk_dict=_tk_dict, 
                past_tk_dict=_past_tk_dict, 
                ) for _ in cwcn_config.cwcn_config.CWCN_UJCAMEI_CAJTUCU_CONFIG.UJCAMEI_ACTIVE_BAO
            ]
        )
        # --- --- --- 
        if(cwcn_config.CWCN_UJCAMEI_CAJTUCU_CONFIG.TIME_DECREMENTAL_SEQUENCE):
            self.instrument_queue.insert(0,_temp_tk_dict)
            self.instrument_queue=self.instrument_queue[:+self.sequence_size]
        else:
            self.instrument_queue.append(_temp_tk_dict)
            self.instrument_queue=self.instrument_queue[-self.sequence_size:]
        # --- --- --- --- --- --- --- --- --- --- --- 
        # --- --- --- INSTRUMENT TENSORIZATION
        # #FIXME make it stable, for now it does not detect batches not needs it never ever.
        _temp_tk_dict={
            'price':torch.Tensor([_temp_tk_dict['price']]).squeeze(0).to(cwcn_config.device),
            'size':torch.Tensor([_temp_tk_dict['size']]).squeeze(0).to(cwcn_config.device),
            'side':torch.Tensor([-1 if _temp_tk_dict['side']=='sell' else +1 if _temp_tk_dict['side']=='buy' else 0]).squeeze(0).to(cwcn_config.device),
            'time_delta':torch.Tensor([_temp_tk_dict['ts']-_past_tk_dict['ts']]).squeeze(0).to(cwcn_config.device),
            'price_delta':torch.Tensor([_temp_tk_dict['price']-_past_tk_dict['price']]).squeeze(0).to(cwcn_config.device),
            'currentQty':torch.Tensor([_temp_tk_dict['currentQty']]).squeeze(0).to(cwcn_config.device),
        }
        # --- --- --- INSTRUMENT DUURUVA
        _temp_tk_dict={
            'price':self.price_duuruva._duuruva_value_wrapper_(_temp_tk_dict['price']),
            'size':self.size_duuruva._duuruva_value_wrapper_(_temp_tk_dict['size']),
            'side':self.side_duuruva._duuruva_value_wrapper_(_temp_tk_dict['side']),
            'time_delta':self.time_delta_duuruva._duuruva_value_wrapper_(_temp_tk_dict['time_delta']),
            'price_delta':self.price_delta_duuruva._duuruva_value_wrapper_(_temp_tk_dict['price_delta']),
            'currentQty':_temp_tk_dict['currentQty'],
        }
        if(cwcn_config.PAPER_INSTRUMENT and cwcn_config.TRAIN_ON_FORECAST):
            self._forecast_non_uwaabo=torch.Tensor([_tk_dict['forecast_non_uwaabo']]).squeeze(0).to(cwcn_config.device)/torch.sqrt(self.price_delta_duuruva._duuruva[0]['variance']+ cwcn_config.CWCN_DUURUVA_CONFIG.MIN_STD)
        self._price=_temp_tk_dict['price'] 
        # --- --- --- 
        self._instrument_queue_healt_()
        # _tick_tensor : [price, price_delta, time_delta, side, size]
        # --- --- --- 
        c_tensor=torch.Tensor([
            _temp_tk_dict[_k] for _k in list(_temp_tk_dict.keys())
        ]).to(cwcn_config.device)
        self._update_alliu_sequence_(c_tensor)
        self.new_tick_aviable_on_queue=False
        self.uc._step_flgs['instrument_updated']=True
    def _get_alliu_(self):
        return self.alliu_sequence_tensor
# --- --- --- --- --- --- 
class UJCAMEI_CAJTUCU_ORDER:
    def __init__(self,price,size,side):
        logging.danger_logging("nothing is ready, come one jump to munamunaake (nah, not ye)")
        logging.warning("UJCAMEI CAJTUCU ORDER IS NOT READY")
        self._price=price
        self._size=size
        self._side=side
class UJCAMEI_CAJTUCU_POSITION:
    def __init__(self,_ujcamei_cajtucu):
        self.uc=_ujcamei_cajtucu
        # self.position_size = None #FIXME not used; count, signed -sell and +buy
        self._reset_position_()
    def _reset_position_(self):
        self.realizedPnl = None
        self.unrealizedPnl = None
        self.currentQty = None
        self.isOpen = None
        self.realizedCost = None
        self.unrealizedCost = None
        self._update_position_()
    def _update_position_(self,_dict=None):
        # #   {
        # #     "id": "123", //Position ID
        # #     "symbol": "BTCUSDTPERP",//Symbol
        # #     "autoDeposit": true,//Auto deposit margin or not
        # #     "maintMarginReq": 0.005,//Maintenance margin requirement
        # #     "riskLimit": 200,//Risk limit
        # #     "realLeverage": 1.06,//Leverage of the order
        # #     "crossMode": false,//Cross mode or not
        # #     "delevPercentage": 0.1,//ADL ranking percentile
        # #     "openingTimestamp": 1558433191000,//Open time
        # #     "currentTimestamp": 1558507727807,//Current timestamp
        # #     "currentQty": -20,//Current position
        # #     "currentCost": 0.00266375,//Current position value
        # #     "currentComm": 0.00000271,//Current commission
        # #     "unrealizedCost": 0.00266375,//Unrealized value
        # #     "realizedGrossCost": 0,//Accumulated realized gross profit value
        # #     "realizedCost": 0.00000271,//Current realized position value
        # #     "isOpen": true,//Opened position or not
        # #     "markPrice": 7933.01,//Mark price
        # #     "markValue": 0.00252111,//Mark value
        # #     "posCost": 0.00266375,//Position value
        # #     "posCross": 1.2e-7,//Manually added margin
        # #     "posInit": 0.00266375,//Leverage margin
        # #     "posComm": 0.00000392,//Bankruptcy cost
        # #     "posLoss": 0,//Funding fees paid out
        # #     "posMargin": 0.00266779,//Position margin
        # #     "posMaint": 0.00001724,//Maintenance margin
        # #     "maintMargin": 0.00252516,//Position margin
        # #     "realizedGrossPnl": 0,//Accumulated realized gross profit value
        # #     "realizedPnl": -0.00000253,//Realised profit and loss
        # #     "unrealizedPnl": -0.00014264,//Unrealized profit and loss
        # #     "unrealizedPnlPcnt": -0.0535,//Profit-loss ratio of the position
        # #     "unrealizedRoePcnt": -0.0535,//Rate of return on investment
        # #     "avgEntryPrice": 7508.22,//Average entry price
        # #     "liquidationPrice": 1000000,//Liquidation price
        # #     "bankruptPrice": 1000000,//Bankruptcy price
        # #     "settleCurrency": "XBT"                         //Currency used to clear and settle the trades     
        # # }
        logging.ujcamei_logging("[UPDATE POSITION]")
        # --- --- --- 
        if(_dict is None):
            _dict=self.uc._echange_instrument.trade_instrument.get_position_details(cwcn_config.CWCN_INSTRUMENT_CONFIG.SYMBOL)
        logging.ujcamei_logging("[update position] results in : {}".format(_dict))
        self.realizedPnl    = _dict['realizedPnl']
        self.unrealizedPnl  = _dict['unrealizedPnl']
        self.currentQty     = _dict['currentQty']
        self.isOpen         = _dict['isOpen']
        # --- --- --- 
        self.uc._step_flgs['position_updated']=True
        # self.realizedCost   = _dict['realizedCost']
        # self.unrealizedCost = _dict['unrealizedCost']
        for _k in list(cwcn_config.CWCN_UJCAMEI_CAJTUCU_CONFIG.POSITION_UPDATE_METHODS.keys()):
            cwcn_config.CWCN_UJCAMEI_CAJTUCU_CONFIG.POSITION_UPDATE_METHODS[_k](self)
        self.uc._uc_wallet_._update_wallet_(_dict) # just a partial update
# --- --- --- --- --- --- --- 
class UJCAMEI_CAJTUCU_WALLET:
    def __init__(self,_ujcamei_cajtucu):
        self.uc=_ujcamei_cajtucu
        self.currency=None
        self.orderMargin=None
        self.availableBalance=None
        self.pastAvailableBalance=None
        self.realizedPnl=None
        self.unrealizedPnl=None
        self.pastUnrealizedPnl=None
        self._reset_wallet_()
    def _reset_wallet_(self):
        if(cwcn_config.PAPER_INSTRUMENT):
            self.currency = cwcn_config.CWCN_SIMULATION_CONFIG.INITIAL_WALLET['currency']
            self.orderMargin = None
            self.availableBalance = cwcn_config.CWCN_SIMULATION_CONFIG.INITIAL_WALLET['availableBalance']
            self.realizedPnl = cwcn_config.CWCN_SIMULATION_CONFIG.INITIAL_WALLET['realizedPnl'] #FIXME not used
            self.unrealizedPnl = cwcn_config.CWCN_SIMULATION_CONFIG.INITIAL_WALLET['unrealizedPnl'] #FIXME not used
        else:
            self._update_wallet_()
        self.pastAvailableBalance = self.availableBalance
        self.pastUnrealizedPnl = self.unrealizedPnl
    def _request_info_(self):pass
    def _update_wallet_(self,_dict=None):
        if(_dict is None and not cwcn_config.PAPER_INSTRUMENT):
            _dict=self.uc._echange_instrument.user_instrument.get_account_overview()
        logging.ujcamei_logging("[UPDATE WALLET] {}".format(_dict))
        uc_dict_keys = list(self.__dict__.keys())
        for _k in list(_dict.keys()):
            if(_k == 'currency' and _dict[_k]!=cwcn_config.CWCN_INSTRUMENT_CONFIG.CURRENCY):
                logging.error("[MAYOR WARNING!] wrong currency detected on ujcamei-cajtucu wallet update {}".format(_dict)) #FIXME, close all orders on error
            elif(_k in uc_dict_keys):
                self.__dict__[_k]=_dict[_k]
        self.uc._step_flgs['wallet_been_updated']=True
        for _k in list(cwcn_config.CWCN_UJCAMEI_CAJTUCU_CONFIG.WALLET_UPDATE_METHODS.keys()):
            cwcn_config.CWCN_UJCAMEI_CAJTUCU_CONFIG.WALLET_UPDATE_METHODS[_k](self)
class KUJTIYU_UJCAMEI_CAJTUCU:
    def __init__(self,_wikimyei):
        logging.ujcamei_logging("Initializating UJCAMEI_CAJTUCU module")
        if(cwcn_config.PAPER_INSTRUMENT):
            self._echange_instrument=cwcn_simulation_kijtyu.EXCHANGE_INSTRUMENT(\
                _call_function=self._ujcamei_)
        else:
            self._echange_instrument=poloniex_api.EXCHANGE_INSTRUMENT(\
                _message_wrapper_=self._ujcamei_, 
                _websocket_subs=cwcn_config.CWCN_UJCAMEI_CAJTUCU_CONFIG.WEB_SOCKET_SUBS)
            self._echange_instrument_loop=asyncio.get_event_loop()
        # --- --- 
        self.wk_ujcamei_cajtucu=_wikimyei
        self._reset_step_flags_()
        self._uc_wallet_= UJCAMEI_CAJTUCU_WALLET(self)
        self._uc_position_= UJCAMEI_CAJTUCU_POSITION(self)
        self._uc_instrumet_ = UJCAMEI_CAJTUCU_INSTRUMENT_REPRESENTATION(self)
        self._uc_instrumet_._load_instrument_representation_()
    def _clear_positions_(self):
        logging.warning("EMERGENCY Cancel [clear all positions] {}\n".format(self._uc_position_.__dict__))
        self._echange_instrument.trade_instrument.clear_positions()
        self._uc_wallet_._update_wallet_()
        self._uc_position_._update_position_()
        self._uc_instrumet_._update_instrument_()
        return True
    def _reset_step_flags_(self):
        self._step_flgs = {
            'is_done_checked':False,
            'healt_checked':False,
            'instrument_updated':False,
            'position_updated':False,
            'wallet_been_updated':False,
            'action_taken':False,
            'action_taken':False,
            'action_taken':False,
            'reward_given':False,
        }
    def _ujcamei_raise_(self, msg, aux_msg=None):
        self._uc_instrumet_.new_tick_aviable_on_queue=False
        logging.warning("[UJCAMEI] found unrecognized message comming from websocket : {} - {}".format(msg, aux_msg if aux_msg is not None else '')) #FIXME, close all orders on error
    def _ujcamei_(self,msg):
        # --- --- --- --- 
        try:
            logging.ujcamei_logging('[UJCAMEI] (upcoming message) {}'.format(msg))
            if('type' in list(msg.keys()) and msg['type'] in ['welcome',"ack","pong"]):
                # {
                #   'id': '49bae88d-c26a-4448-9028-8d313d599f63', 'type': 'welcome'/'pong'
                # }
                pass
            elif('type' in list(msg.keys()) and msg['type'] in ['subscribe']):
                # Subscribe note 
                # {
                #     "id":   
                #     "type": "subscribe",
                #     "topic": "/contractMarket/execution:BTCUSDTPERP",
                #     "response": true                              
                # }
                if('response' in list(msg.keys()) and str(msg['response']).lower()!='true'):
                    self._ujcamei_raise_(msg, '[expected message response to be true]')
            elif('topic' in list(msg.keys()) and '/contractMarket/ticker' in msg['topic']):
                # ticker data
                # {
                #     "subject": "ticker",
                #     "topic": "/contractMarket/ticker:BTCUSDTPERP",
                #     "data": {
                #       "symbol": "BTCUSDTPERP", //Market of the symbol
                #       "sequence": 45,//Sequence number which is used to judge the continuity of the pushed messages
                #       "side": "sell",//Transaction side of the last traded taker order
                #       "price": 3600.00,//Filled price
                #       "size": 16,//Filled quantity
                #       "tradeId": <md5?>,    //Order ID
                #       "bestBidSize": 795,      //Best bid size
                #       "bestBidPrice": 3200.00, //Best bid 
                #       "bestAskPrice": 3600.00, //Best ask size
                #       "bestAskSize": 284,      //Best ask
                #       "ts": 1553846081210004941     //Filled time - nanosecond
                #     }
                # }
                if('subject' in list(msg.keys()) and msg['subject'] in ['ticker']):
                    if(msg['data']['symbol']==cwcn_config.CWCN_INSTRUMENT_CONFIG.SYMBOL):
                        aux_dict={
                            'price' : msg['data']['price'],
                            'side'  : msg['data']['side'],
                            'size'  : msg['data']['size'],
                            'ts'    : msg['data']['ts'],
                        }
                        if(cwcn_config.PAPER_INSTRUMENT and cwcn_config.TRAIN_ON_FORECAST):
                            aux_dict['forecast_non_uwaabo'] = msg['data']['forecast_non_uwaabo']
                        self._uc_instrumet_._update_instrument_(msg['data'])
                        self._uc_position_._update_position_() #FIXME fokin slow
                    else:
                        self._ujcamei_raise_(msg,"[unexpected symbol in ticker]")
                else:
                    self._ujcamei_raise_(msg,"[unregocnized ticker update]")
            elif('topic' in list(msg.keys()) and '/contractAccount/wallet' in msg['topic']):
                if('subject' in list(msg.keys()) and msg['subject'] in ['orderMargin.change']):
                    # Order Margin Event
                    # { 
                    #     "topic": "/contractAccount/wallet",
                    #     "subject": "orderMargin.change",
                    #     "channelType": "private",
                    #     "data": {
                    #       "orderMargin": 5923,//Current order margin
                    #       "currency":"USDT",//Currency
                    #       "timestamp": 1553842862614
                    #     }
                    # }
                    self._uc_wallet_._update_wallet_(msg['data'])
                    pass
                elif('subject' in list(msg.keys()) and msg['subject'] in ['availableBalance.change']):
                    # Available Balance Event
                    # { 
                    # "topic": "/contractAccount/wallet",
                    # "subject": "availableBalance.change",
                    # "channelType": "private",
                    # "data": {
                    #     "availableBalance": 5923, //Current available amount
                    #     "currency":"USDT",//Currency
                    #     "timestamp": 1553842862614
                    #   }
                    # }
                    self._uc_wallet_._update_wallet_(msg['data'])
                else:
                    self._ujcamei_raise_(msg, '[unregocnized wallet update]')
            # --- --- --- --- 
            # # # elif('/contractMarket/execution' in msg['topic']): # not in use, find why is not redudant to ticker
            # # #     if(msg['subject'] in ['match']):
            # # #         # {
            # # #         # "topic": "/contractMarket/execution:BTCUSDTPERP",
            # # #         # "subject": "match",
            # # #         # "data": {
            # # #         #     "symbol": "BTCUSDTPERP",       //Symbol
            # # #         #     "sequence": 36,//Sequence number which is used to judge the continuity of the pushed messages  
            # # #         #     "side": "buy",// Side of liquidity taker
            # # #         #     "matchSize": 1,   //Filled quantity
            # # #         #     "size": 1,//unFilled quantity
            # # #         #     "price": 3200.00,// Filled price
            # # #         #     "takerOrderId": "5c9dd00870744d71c43f5e25",  //Taker order ID
            # # #         #     "ts": 1553846281766256031,//Filled time - nanosecond
            # # #         #     "makerOrderId": "5c9d852070744d0976909a0c",  //Maker order ID
            # # #         #     "tradeId": "5c9dd00970744d6f5a3d32fc"        //Transaction ID
            # # #         #     }
            # # #         # }
            # # #         # "data": {
            # # #         #     'makerUserId': '123', 
            # # #         #         'symbol': 'ADAUSDTPERP', 
            # # #         #         'sequence': 288941, 
            # # #         #         'side': 'sell', 
            # # #         #         'size': 10, 
            # # #         #         'price': 2.82175, 
            # # #         #         'takerOrderId': '6136bb245e36c2000646ecfd', 
            # # #         #         'makerOrderId': '6136bb245e36c2000646ecf9', 
            # # #         #         'takerUserId': '14330687', 
            # # #         #         'tradeId': '6136bb2421865f0ea7f60da3', 
            # # #         #         'ts': 1630976773229714931
            # # #         # }
            # # #         aux_dict={
            # # #           'size'  : msg['data']['matchSize'], #FIXME assert is the correct item
            # # #           'price' : msg['data']['price'],
            # # #           'side'  : msg['data']['side'],
            # # #         }
            # # #         pass
            # # #     else:
            # # #         self._ujcamei_raise_(msg, '[unregocnized execution]')
            else:
                self._ujcamei_raise_(msg, '[unregocnized topic]')
        except Exception as e:
            self._ujcamei_raise_(e, "[EXCEPTION ERROR]")
    # --- --- --- --- 
    # --- --- --- --- --- 
    # --- --- --- --- --- --- ENVIROMENT
    # --- --- --- --- --- 
    # --- --- --- --- 
    def _take_action_(self,_tsane : int, _certainty):
        if(torch.is_tensor(_tsane)):
            _action=cwcn_config.CWCN_UJCAMEI_CAJTUCU_CONFIG.TSANE_ACTION_DICT[_tsane.detach().item()]
            _cert=_certainty[_tsane.detach().item()]
        else:
            _action=cwcn_config.CWCN_UJCAMEI_CAJTUCU_CONFIG.TSANE_ACTION_DICT[_tsane]
            _cert=_certainty[_tsane]
        if(abs(self._uc_position_.currentQty)>=cwcn_config.CWCN_UJCAMEI_CAJTUCU_CONFIG.MAX_POSITION_SIZE):
            if(_action=='call' and self._uc_position_.currentQty>0):
                _action='pass'
            elif(_action=='put' and self._uc_position_.currentQty<0):
                _action='pass'
        if(cwcn_config.ALLOW_TSANE or cwcn_config.PAPER_INSTRUMENT):
            if(_action == 'pass' or _cert<=cwcn_config.CWCN_UJCAMEI_CAJTUCU_CONFIG.CERTAINTY_FACTOR):
                logging.tsane_logging(":   · | ·   : pass : {} : ".format(['%.4f' % n for n in _certainty.tolist()]))
                self._step_flgs['action_taken']=True
            elif(_action == 'call'):
                logging.tsane_logging(": | ·   ·   : buy  : {} : ".format(['%.4f' % n for n in _certainty.tolist()]))
                self._echange_instrument.trade_instrument.create_market_order(\
                    symbol=cwcn_config.CWCN_INSTRUMENT_CONFIG.SYMBOL,
                    side='buy',
                    size=1,
                    leverage=cwcn_config.CWCN_INSTRUMENT_CONFIG.LEVERAGE)
                self._step_flgs['action_taken']=True
            elif(_action == 'put'):
                logging.tsane_logging(":   ·   · | : sell : {} : ".format(['%.4f' % n for n in _certainty.tolist()]))
                self._echange_instrument.trade_instrument.create_market_order(\
                    symbol=cwcn_config.CWCN_INSTRUMENT_CONFIG.SYMBOL,
                    side='sell',
                    size=1,
                    leverage=cwcn_config.CWCN_INSTRUMENT_CONFIG.LEVERAGE)
                self._step_flgs['action_taken']=True
            else:
                logging.error("unrecognized tsane : {} : {} \ {}".format(_tsane,_action,_certainty.detach()))
        else:
            logging.warning("Action is not permitted by cuwacunu user : [not done] {}, {}".format(_action, ['%.4f' % n for n in _certainty.tolist()]))
        return _action
    # --- --- --- --- 
    def _get_alliu_(self):
        return self._uc_instrumet_._get_alliu_()
    def _get_reward_(self,_certainty):
        self._step_flgs['reward_given']=True
        try:
            if(not self._step_flgs['wallet_been_updated']):
                logging.error("update wallet before getting reward")
            # return self._uc_wallet_.availableBalance * _certainty.std() #self._uc_wallet_.realizedPnl+
            rward=self._uc_wallet_.availableBalance-self._uc_wallet_.pastAvailableBalance\
                +0.6*(self._uc_wallet_.unrealizedPnl-self._uc_wallet_.pastUnrealizedPnl)
            self._uc_wallet_.pastAvailableBalance=self._uc_wallet_.availableBalance
            self._uc_wallet_.pastUnrealizedPnl=self._uc_wallet_.unrealizedPnl
            # return self._uc_wallet_.availableBalance + 0.7*self._uc_wallet_.unrealizedPnl #self._uc_wallet_.realizedPnl+
        except Exception as e:
            logging.warning("Unable to get REWARD : {}".format(e))
            rward=0.0
        return rward
    def _get_if_is_done_(self):
        self._step_flgs['is_done_checked']=True
        return False #FIXME never done
    def _request_info_(self,_certainty):
        _info={
            # 'README.rc4': rcsi_utils.RCsi_CRYPT("TEHDUJUCO","el miamunaake juega el té de la divinidad por combatir recursos ('piaabo==ayuda') \
            # la busqueda por el sentimiento que es sincero ('adho'==lugar) si los datos ('alliu'==mentira); el método es en whiakaju y las decisiónes \
            # que esquivan sacar la lengua en el juego de vendiciones (''duuruva''=='vendición'); ahora que no existe más el maldito oro, quién podrá \
            # venderse por una vendición; y el mercado bendice a los que le resuelven, resolverlo por corresponderme parte de la arábica hazaña \
            # de lo que está bendito en las ecuaciones; en mi no hay eso bendito que se combate a sí por los recursos y tampoco lo habita mi código: \
            # les juro un ('tsodaho') protejiendo el ('nebajke') del que estando vivo combate en si la gracia sincera al consciente recaudo de dicha y tiempo."),
            'reward_was_value' : self.c_reward,
            'alliu_was_value' : None,
            'certainty_was' : _certainty,
            'put_certainty_was_value' : _certainty[list(cwcn_config.CWCN_UJCAMEI_CAJTUCU_CONFIG.TSANE_ACTION_DICT.values()).index('put')],
            'pass_certainty_was_value' : _certainty[list(cwcn_config.CWCN_UJCAMEI_CAJTUCU_CONFIG.TSANE_ACTION_DICT.values()).index('pass')],
            'call_certainty_was_value' : _certainty[list(cwcn_config.CWCN_UJCAMEI_CAJTUCU_CONFIG.TSANE_ACTION_DICT.values()).index('call')],
            'done_was_value' : self.c_done,
            'price_was_value': self._uc_instrumet_._price,
        }
        if(cwcn_config.PAPER_INSTRUMENT and cwcn_config.TRAIN_ON_FORECAST):
            _info['non_uwaabo_forecast_was_value'] = self._uc_instrumet_._forecast_non_uwaabo
        return _info
    def _wait_for_step_(self):
        if(cwcn_config.PAPER_INSTRUMENT):
            if(not self._echange_instrument.market_instrument.simulation_step_ticker(symbol=cwcn_config.CWCN_INSTRUMENT_CONFIG.SYMBOL)):
                logging.warning("RESETING DATA ADQUISITION, RESTARTING FILE ITERATION")
                self._uc_instrumet_._load_instrument_representation_()
        else:
            while(not self.new_tick_aviable_on_queue):
                self._echange_instrument_loop.run_until_complete(asyncio.sleep(0.05))
            # sys.stdout.write("··· waiting for tick ···")
            # sys.stdout.write(cwcn_config.CWCN_CURSOR.CARRIER_RETURN)
            # sys.stdout.write(cwcn_config.CWCN_CURSOR.CLEAR_LINE)
    def _assert_step_flags_(self):
        self.step_bugger_flag=all([self._step_flgs[_f] for _f in list(self._step_flgs.keys())])
        if(not self.step_bugger_flag):
            logging.error("step is found defetive : {}".format(self._step_flgs))
    def step(self,_tsane, _certainty):
        # ---
        self._reset_step_flags_()
        self._take_action_(_tsane=_tsane, _certainty=_certainty)
        self._wait_for_step_()
        # --- 
        self.c_alliu=self._get_alliu_()
        self.c_done=self._get_if_is_done_()
        self.c_reward=self._get_reward_(_certainty=_certainty)
        self.c_info=self._request_info_(_certainty=_certainty)
        # ---
        self._assert_step_flags_()
        sys.stdout.write("{}{}\t\t\t\t\t\t\t\t\t\t\t\t\t\tbalance : {:.4f} \t unrealizedPnl: {:.4}\n".format(cwcn_config.CWCN_CURSOR.CARRIER_RETURN,cwcn_config.CWCN_CURSOR.UP,self._uc_wallet_.availableBalance, self._uc_wallet_.unrealizedPnl))
        sys.stdout.flush()
        # ---
        return self.c_alliu,self.c_reward,self.c_done,self.c_info
        # ---
    def reset(self): #FIXME reset is not implemented
        if(cwcn_config.PAPER_INSTRUMENT):
            self._echange_instrument.user_instrument._reset_()
            self._echange_instrument.trade_instrument._reset_()
            self._echange_instrument.market_instrument._reset_()
            # --- 
        else:
            logging.warning("reset is not thougth trught on non paper instrument")
            self._update_wallet_(self.uc._echange_instrument.user_instrument.get_account_overview())
        # --- --- 
        self.wk_ujcamei_cajtucu.wk_state._reset_()
        # --- --- 
        c_alliu,_,__,__=self.step(
            _tsane=list(cwcn_config.CWCN_UJCAMEI_CAJTUCU_CONFIG.TSANE_ACTION_DICT.keys())[list(cwcn_config.CWCN_UJCAMEI_CAJTUCU_CONFIG.TSANE_ACTION_DICT.values()).index('pass')],
            _certainty=torch.Tensor([0.0, 0.0, 0.0]).to(cwcn_config.device)
            )
        # --- --- 
        return c_alliu
    def render(self):
        pass
if __name__=="__main__":
    ujcamei_cajtucu=KUJTIYU_UJCAMEI_CAJTUCU()
    print("--- --- --- --- ")
    print(ujcamei_cajtucu._uc_instrumet_.instrument_queue)
    print("--- --- --- --- ")
    print(ujcamei_cajtucu._get_alliu_())
    print("--- --- --- --- ")
    print("--- --- --- --- ")
    print("--- --- --- --- ")
    print("--- --- --- --- ")
    print(ujcamei_cajtucu.step(0))
    print("--- --- --- --- ")
    print(ujcamei_cajtucu.step(1))
    print("--- --- --- --- ")
    print(ujcamei_cajtucu.step(2))
    print("--- --- --- --- ")
    # ujcamei_cajtucu._uc_instrumet_.price_duuruva._plot_duuruva_()
    # ujcamei_cajtucu._uc_instrumet_.size_duuruva._plot_duuruva_()
    # ujcamei_cajtucu._uc_instrumet_.side_duuruva._plot_duuruva_()
    # ujcamei_cajtucu._uc_instrumet_.time_delta_duuruva._plot_duuruva_()
    # ujcamei_cajtucu._uc_instrumet_.price_delta_duuruva._plot_duuruva_()
    # import matplotlib.pyplot as plt
    # plt.show()