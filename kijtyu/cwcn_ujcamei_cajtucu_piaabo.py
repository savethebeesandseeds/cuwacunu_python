# --- --- 
import sys
sys.path.append('../communications/')
# --- --- 
import torch
# --- --- 
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
        self.sequence_size = cwcn_config.CWCN_CONFIG().UJCAMEI_ALLIU_SEQUENCE_SIZE
        self.new_tick_aviable_on_queue = False
        self.uc=_ujcamei_cajtucu
        self.price_duuruva=cwcn_duuruva_piaabo.DUURUVA(_duuruva_vector_size=0x01, _d_name='price_duuruva'.upper(),
            _wrapper_duuruva_normalize=cwcn_config.CWCN_DUURUVA_CONFIG.NORMALIZE_ALLIU_PRICE)
        self.size_duuruva=cwcn_duuruva_piaabo.DUURUVA(_duuruva_vector_size=0x01, _d_name='size_duuruva'.upper(),
            _wrapper_duuruva_normalize=cwcn_config.CWCN_DUURUVA_CONFIG.NORMALIZE_ALLIU_SIZE) #Z
        self.side_duuruva=cwcn_duuruva_piaabo.DUURUVA(_duuruva_vector_size=0x01, _d_name='side_duuruva'.upper(),
            _wrapper_duuruva_normalize=cwcn_config.CWCN_DUURUVA_CONFIG.NORMALIZE_ALLIU_SIDE) #D
        self.price_delta_duuruva=cwcn_duuruva_piaabo.DUURUVA(_duuruva_vector_size=0x01, _d_name='price_delta_duuruva'.upper(),
            _wrapper_duuruva_normalize=cwcn_config.CWCN_DUURUVA_CONFIG.NORMALIZE_ALLIU_PRICE_DELTA)
        self.time_delta_duuruva=cwcn_duuruva_piaabo.DUURUVA(_duuruva_vector_size=0x01, _d_name='time_delta_duuruva'.upper(),
            _wrapper_duuruva_normalize=cwcn_config.CWCN_DUURUVA_CONFIG.NORMALIZE_ALLIU_TIME_DELTA)
        self.alliu_sequence_tensor=torch.zeros((self.sequence_size,cwcn_config.CWCN_UJCAMEI_CAJTUCU_CONFIG.ALLIU_LEN)).to(cwcn_config.device)
        self.instrument_queue=[]
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
        self._step_flgs['healt_checked']=True
        return _healt_flag
    def _load_instrument_representation_(self):
        c_max_count=max(self.sequence_size,cwcn_config.CWCN_DUURUVA_CONFIG.DUURUVA_READY_COUNT)
        c_instrument_history=self.uc._echange_instrument.market.get_trade_history(cwcn_config.CWCN_INSTRUMENT_CONFIG.SYMBOL)
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
    def _update_instrument_(self,_dict):
        # logging.info("_update_instrument_")
        # --- --- --- CAST
        _dict['price']=float(_dict['price'])
        _dict['size']=int(_dict['size'])
        _dict['ts']=int(_dict['ts'])
        # --- --- --- 
        if(self.new_tick_aviable_on_queue==True): # ticked arrived too fast for prossesing unit; accomulating tick
            logging.error("Not expected behaviour; double update, inconclusive (appending method need to be FIX!)")
            if(cwcn_config.CWCN_UJCAMEI_CAJTUCU_CONFIG.TIME_DECREMENTAL_SEQUENCE):
                c_index=0
            else:
                c_index=-1
            self.instrument_queue[c_index]['price']=_dict['price']
            self.instrument_queue[c_index]['size']+=_dict['size'] #FIXME this is incorrect
            self.instrument_queue[c_index]['side']=_dict['side']
            self.instrument_queue[c_index]['ts']=_dict['ts']
            self.instrument_queue[c_index]['currentQty']=self.uc._uc_position_.currentQty
        self.new_tick_aviable_on_queue=True...
        if(len(self.instrument_queue)!=0):
            if(cwcn_config.CWCN_UJCAMEI_CAJTUCU_CONFIG.TIME_DECREMENTAL_SEQUENCE):
                past_dict=self.instrument_queue[0]
            else:
                past_dict=self.instrument_queue[-1]
        else:
            past_dict=copy.deepcopy(_dict)
        # --- --- --- INTRUMENT QUEUE
        aux_dict={
            'price':_dict['price'],
            'size':_dict['size'],
            'side':-1 if _dict['side']=='sell' else +1 if _dict['side']=='buy' else 0,
            'time_delta':_dict['ts']-past_dict['ts'],
            'price_delta':_dict['price']-past_dict['price'],
            'ts':_dict['ts'],
            'sequence':_dict['sequence'],
            'currentQty':self.uc._uc_position_.currentQty,
        }
        if(cwcn_config.CWCN_UJCAMEI_CAJTUCU_CONFIG.TIME_DECREMENTAL_SEQUENCE):
            self.instrument_queue.insert(0,aux_dict)
            self.instrument_queue=self.instrument_queue[:+self.sequence_size]
        else:
            self.instrument_queue.append(aux_dict)
            self.instrument_queue=self.instrument_queue[-self.sequence_size:]
        # --- --- --- INSTRUMENT DUURUVA
        aux_dict={
            'price':torch.Tensor([aux_dict['price']]).squeeze(0).to(cwcn_config.device),
            'size':torch.Tensor([aux_dict['size']]).squeeze(0).to(cwcn_config.device),
            'side':torch.Tensor([-1 if aux_dict['side']=='sell' else +1 if aux_dict['side']=='buy' else 0]).squeeze(0).to(cwcn_config.device),
            'time_delta':torch.Tensor([aux_dict['ts']-past_dict['ts']]).squeeze(0).to(cwcn_config.device),
            'price_delta':torch.Tensor([aux_dict['price']-past_dict['price']]).squeeze(0).to(cwcn_config.device),
            'currentQty':torch.Tensor([aux_dict['currentQty']]).squeeze(0).to(cwcn_config.device),
        }
        aux_dict={
            'price':self.price_duuruva._duuruva_value_wrapper_(aux_dict['price']),
            'size':self.size_duuruva._duuruva_value_wrapper_(aux_dict['size']),
            'side':self.side_duuruva._duuruva_value_wrapper_(aux_dict['side']),
            'time_delta':self.time_delta_duuruva._duuruva_value_wrapper_(aux_dict['time_delta']),
            'price_delta':self.price_delta_duuruva._duuruva_value_wrapper_(aux_dict['price_delta']),
            'currentQty':aux_dict['currentQty'],
        }
        # --- --- --- 
        self._instrument_queue_healt_()
        # _tick_tensor : [price, price_delta, time_delta, side, size]
        # --- --- --- 
        c_tensor=torch.Tensor([
            aux_dict['price'],
            aux_dict['price_delta'],
            aux_dict['time_delta'],
            aux_dict['side'],
            aux_dict['size'],
            aux_dict['currentQty'],
        ]).to(cwcn_config.device)
        self._update_alliu_sequence_(c_tensor)
        self.new_tick_aviable_on_queue=False
        self._step_flgs['instrument_updated']=True

    def _get_alliu_(self):
        return self.alliu_sequence_tensor
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
        self.realisedPnl = None
        self.unrealisedPnl = None
        self.currentQty = None
        self.isOpen = None
        self.realisedCost = None
        self.unrealisedCost = None
        self._update_position_()
    def _update_position_(self):
        #   {
        #     "id": "123", //Position ID
        #     "symbol": "BTCUSDTPERP",//Symbol
        #     "autoDeposit": true,//Auto deposit margin or not
        #     "maintMarginReq": 0.005,//Maintenance margin requirement
        #     "riskLimit": 200,//Risk limit
        #     "realLeverage": 1.06,//Leverage of the order
        #     "crossMode": false,//Cross mode or not
        #     "delevPercentage": 0.1,//ADL ranking percentile
        #     "openingTimestamp": 1558433191000,//Open time
        #     "currentTimestamp": 1558507727807,//Current timestamp
        #     "currentQty": -20,//Current position
        #     "currentCost": 0.00266375,//Current position value
        #     "currentComm": 0.00000271,//Current commission
        #     "unrealisedCost": 0.00266375,//Unrealised value
        #     "realisedGrossCost": 0,//Accumulated realised gross profit value
        #     "realisedCost": 0.00000271,//Current realised position value
        #     "isOpen": true,//Opened position or not
        #     "markPrice": 7933.01,//Mark price
        #     "markValue": 0.00252111,//Mark value
        #     "posCost": 0.00266375,//Position value
        #     "posCross": 1.2e-7,//Manually added margin
        #     "posInit": 0.00266375,//Leverage margin
        #     "posComm": 0.00000392,//Bankruptcy cost
        #     "posLoss": 0,//Funding fees paid out
        #     "posMargin": 0.00266779,//Position margin
        #     "posMaint": 0.00001724,//Maintenance margin
        #     "maintMargin": 0.00252516,//Position margin
        #     "realisedGrossPnl": 0,//Accumulated realised gross profit value
        #     "realisedPnl": -0.00000253,//Realised profit and loss
        #     "unrealisedPnl": -0.00014264,//Unrealised profit and loss
        #     "unrealisedPnlPcnt": -0.0535,//Profit-loss ratio of the position
        #     "unrealisedRoePcnt": -0.0535,//Rate of return on investment
        #     "avgEntryPrice": 7508.22,//Average entry price
        #     "liquidationPrice": 1000000,//Liquidation price
        #     "bankruptPrice": 1000000,//Bankruptcy price
        #     "settleCurrency": "XBT"                         //Currency used to clear and settle the trades     
        # }
        _dict=self.uc._echange_instrument.trade.get_position_details(cwcn_config.CWCN_INSTRUMENT_CONFIG.SYMBOL)
        logging.info("[UPDATE POSITION] {}".format(_dict))
        self.realisedPnl    = _dict['realisedPnl']
        self.unrealisedPnl  = _dict['unrealisedPnl']
        self.currentQty     = _dict['currentQty']
        self.isOpen         = _dict['isOpen']
        self.uc._step_flgs['position_updated']=True

        # self.realisedCost   = _dict['realisedCost']
        # self.unrealisedCost = _dict['unrealisedCost']
class UJCAMEI_CAJTUCU_WALLET:
    def __init__(self,_ujcamei_cajtucu):
        self.uc=_ujcamei_cajtucu
        self.currency = None
        self.orderMargin = None
        self.availableBalance = None
        self.realized_pnl = None #FIXME not used
        self.unrealized_pnl = None #FIXME not used
        self._initialize_wallet_()
    def _initialize_wallet_(self):
        self._update_wallet_(self.uc._echange_instrument.user.get_account_overview())
    def _request_info_(self):pass
    def _update_wallet_(self,_dict):
        uc_dict_keys = list(self.__dict__.keys())
        for _k in list(_dict.keys()):
            if(_k == 'currency' and _dict[_k]!=cwcn_config.CWCN_INSTRUMENT_CONFIG.CURRENCY):
                logging.error("[MAYOR WARNING!] wrong currency detected on ujcamei-cajtucu wallet update {}".format(_dict)) #FIXME, close all orders on error
            elif(_k in uc_dict_keys):
                self.__dict__[_k]=_dict[_k]
        self.uc._step_flgs['wallet_been_updated']=True
        # self.uc._uc_position_._update_position_()
class KUJTIYU_UJCAMEI_CAJTUCU:
    def __init__(self):
        logging.info("Initializating UJCAMEI_CAJTUCU module")
        if(cwcn_config.PAPER_INSTRUMENT):
            self._echange_instrument=cwcn_simulation_kijtyu.EXCHANGE_INSTRUMENT(\
                _call_function=self._ujcamei_)
        else:
            self._echange_instrument=poloniex_api.EXCHANGE_INSTRUMENT(\
                _message_wrapper_=self._ujcamei_, 
                _websocket_subs=cwcn_config.CWCN_UJCAMEI_CAJTUCU_CONFIG.WEB_SOCKET_SUBS)
            self._echange_instrument_loop=asyncio.get_event_loop()
        # --- --- 
        self._uc_wallet_= UJCAMEI_CAJTUCU_WALLET(self)
        self._uc_position_= UJCAMEI_CAJTUCU_POSITION(self)
        self._uc_instrumet = UJCAMEI_CAJTUCU_INSTRUMENT_REPRESENTATION(self)
        self._uc_instrumet._load_instrument_representation_()
    def _ujcamei_raise_(self, msg, aux_msg=None):
        self._uc_instrumet.new_tick_aviable_on_queue=False
        logging.warning("[UJCAMEI] found unrecognized message comming from websocket : {} - {}".format(msg, aux_msg if aux_msg is not None else '')) #FIXME, close all orders on error
    def _ujcamei_(self,msg):
        # --- --- --- --- 
        try:
            logging.info('[UJCAMEI] <upcoming message> {}'.format(msg))
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
                        self._uc_instrumet._update_instrument_(msg['data'])
                        self._uc_position_._update_position_()
                    else:
                        self._ujcamei_raise_(msg,"[unexpected symbol in ticker]")
                else:
                    self._ujcamei_raise_(msg,"[unregocnized ticker]")
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
                    self._uc_position_._update_position_()
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
                    self._uc_position_._update_position_()
                else:
                    self._ujcamei_raise_(msg, '[unregocnized wallet]')
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
    # --- --- --- --- --- --- 
    def take_action(self,_tsane : int):
        _action=cwcn_config.CWCN_UJCAMEI_CAJTUCU_CONFIG.TSANE_ACTION_DICT[_tsane]
        if(cwcn_config.ALLOW_TSANE):
            if(_action == 'pass'):
                logging.tsane_logging(":   . | .   : pass : {} : {}".format(_tsane,_action))
                self._step_flgs['action_taken']=True
            elif(_action == 'call'):
                logging.tsane_logging(": | .   .   : buy  : {} : {}".format(_tsane,_action))
                self._echange_instrument.trade.create_market_order(\
                    symbol=cwcn_config.CWCN_INSTRUMENT_CONFIG.SYMBOL,
                    side='buy',
                    size=1,
                    leverage=0)
                self._step_flgs['action_taken']=True
            elif(_action == 'put'):
                logging.tsane_logging(":   .   . | : sell : {} : {}".format(_tsane,_action))
                self._echange_instrument.trade.create_market_order(\
                    symbol=cwcn_config.CWCN_INSTRUMENT_CONFIG.SYMBOL,
                    side='sell',
                    size=1,
                    leverage=0)
                self._step_flgs['action_taken']=True
            else:
                logging.error("unrecognized tsane : {} : {}".format(_tsane,_action))
        else:
            logging.warning("Action is not permitted by cuwacunu user")
    # --- --- --- --- 
    def _get_alliu_(self):
        return self._uc_instrumet._get_alliu_()
    def _get_reward_(self):
        self._step_flgs['reward_given']=True

        ...
    def _get_if_is_done_(self):
        ...
    def _request_info_(self):
        return {
        'README.rc4': rcsi_utils.RCsi_CRYPT("TEHDUJUCO","los monos juegan a la divinidad por el combate de los recursos ('piaabo==ayuda') \
        la busqueda por el sentimiento que es sincero ('adho'==lugar) si los datos ('alliu'==mentira) y el procedimiento es de un mono \
        que evita juegar a la divinidad con bendiciones (''duuruva''==bendición); una 'vendición' traduce aquel mono que por cualquier cosa --ahora \
        que no existe más dinero-- se vende; y el mercado bendice a los que le resuelven y yo quiero resolverlo a corresponde a mi asignar la práctica \
        de lo que está bendito en las ecuaciones. en mi no hay eso bendito que combate por los recursos y tampoco lo habita mi código: \
        el ('tsodaho') proteje el ('nebajke') del que estando vivo combate de si la sinceridad o gracia, al consciente recaudo de dicha y tiempo."),
            'reward_was_value' : None,
            'alliu_was_value' : None,
            'done_was_value' : None,
        }
    def _wait_for_step_(self):
        if(cwcn_config.PAPER_INSTRUMENT):
            self._echange_instrument.simulation_step_ticker()
        else:
            while(not self.new_tick_aviable_on_queue):
                self._echange_instrument_loop.run_until_complete(asyncio.sleep(0.05))
    def _assert_step_(self):
        self.step_bugger_flag=all([self._step_flgs[_f] for _f in list(self._step_flgs.keys())])
        if(self.step_bugger_flag):
            logging.error(self.step_bugger_flag), "step is found defetive"
        self.step_bugger_flag=False
    def step(self,_tasne):
        if(torch.is_tensor(_tasne)):
            self.take_action(_tasne.detach().numpy())
        else:
            self.take_action(_tasne)
        self._wait_for_step_()
        self._assert_step_()
        return self._get_alliu_(), self._get_reward_(), self._get_if_is_done_(), self._request_info_()
if __name__=="__main__":
    ujcamei_cajtucu=KUJTIYU_UJCAMEI_CAJTUCU()
    print("--- --- --- --- ")
    print(ujcamei_cajtucu._uc_instrumet.instrument_queue)
    print("--- --- --- --- ")
    print(ujcamei_cajtucu._get_alliu_())
    print("--- --- --- --- ")
    print("--- --- --- --- ")
    print("--- --- --- --- ")
    print("--- --- --- --- ")
    ujcamei_cajtucu.step(0)
    print("--- --- --- --- ")
    ujcamei_cajtucu.step(1)
    print("--- --- --- --- ")
    ujcamei_cajtucu.step(2)
    print("--- --- --- --- ")
    # ujcamei_cajtucu._uc_instrumet.price_duuruva._plot_duuruva_()
    # ujcamei_cajtucu._uc_instrumet.size_duuruva._plot_duuruva_()
    # ujcamei_cajtucu._uc_instrumet.side_duuruva._plot_duuruva_()
    # ujcamei_cajtucu._uc_instrumet.time_delta_duuruva._plot_duuruva_()
    # ujcamei_cajtucu._uc_instrumet.price_delta_duuruva._plot_duuruva_()
    # import matplotlib.pyplot as plt
    # plt.show()