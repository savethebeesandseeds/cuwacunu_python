# --- --- --- --- 
# cwcn_simulation_kijtyu
# --- --- --- --- 
# --- --- --- --- no async needed
import ast
import random
from uuid import uuid4
import os
import sys
import logging
# --- --- --- --- 
# --- --- --- --- 
import cwcn_config
# --- --- -- --- 

# --- ---  
# --- --- --- 
# --- --- --- --- 
# --- --- --- --- --- --- CORE
# --- --- --- --- 
# --- --- ---
# --- --- 

# --- --- --- --- EXCHANGE INSTRUMENT

class EXCHANGE_INSTRUMENT:
    def __init__(self,_call_function):
        logging.info("[start:] (simulation) EXCHANGE INSTRUMENT")
        self.market_instrument = MarketClient(self)
        self.trade_instrument = TradeClient(self)
        self.user_instrument = UserClient(self)
        self.call_function = _call_function
        logging.info("[ready:] (simulation) EXCHANGE INSTRUMENT")
    # --- --- --- 
    def _make_all_step_updates_(self):
        self.trade_instrument._update_current_orders_state_pnl_()
        self.trade_instrument._close_open_orders_by_prob_()
        self.user_instrument._update_account_overview_()
        return True
    def _step_updates_(self):
        if(self.market_instrument._load_tick_()):
            self._make_all_step_updates_()
            return True
        else:
            return False
    def _invoke_ujcamei_cajtucu_(self):
        _tick_data={
            "symbol": cwcn_config.CWCN_INSTRUMENT_CONFIG.SYMBOL,
            "price": self.market_instrument._instrument_state['price'],
            "side": self.market_instrument._instrument_state['side'],
            "size": self.market_instrument._instrument_state['size'],
            "ts": self.market_instrument._instrument_state['ts'],
        }
        if(cwcn_config.PAPER_INSTRUMENT and cwcn_config.TRAIN_ON_FORECAST):
            _tick_data['forecast_non_uwaabo']=self.market_instrument._instrument_state['forecast_non_uwaabo']
        # --- --- --- --- --- 
        self.call_function({
            'topic':'/contractAccount/wallet',
            'subject': 'availableBalance.change',
            'data':{
                "availableBalance": self.user_instrument.acc_overview_state['availableBalance'],
                "currency":self.user_instrument.acc_overview_state['currency'],
            }
        })
        self.call_function({
            'topic':'/contractMarket/ticker',
            'subject': 'ticker',
            'data':_tick_data,
        })
# --- --- --- --- MARKET 
class MarketClient:
    def __init__(self, _exchange_instrument):
        self.ei=_exchange_instrument
        self._request_file_path = os.path.join(cwcn_config.CWCN_SIMULATION_CONFIG.DATA_FOLDER,"{}{}".format(cwcn_config.CWCN_INSTRUMENT_CONFIG.SYMBOL,cwcn_config.CWCN_SIMULATION_CONFIG.DATA_EXTENSION))
        if(cwcn_config.PAPER_INSTRUMENT and cwcn_config.TRAIN_ON_FORECAST):
            self._enhance_file_with_forecast_()
        self._load_file_()
        self._reset_()
    def _enhance_file_with_forecast_(self):
        # --- --- --- open
        with open(self._request_file_path) as _f_:
            file_content=_f_.read().split(',\n')
        file_ticks=[]
        # --- --- --- forecast
        for _fc in file_content:
            try:
                file_ticks.append(ast.literal_eval(_fc))
            except:
                logging.warning("Failed to enhance line file content : {}".format(_fc))
        for _idn,_ft in enumerate(file_ticks[:-cwcn_config.FORECAST_HORIZONS]):
            _ft['forecast_non_uwaabo']=0
            for _idh in range(cwcn_config.FORECAST_HORIZONS):
                _ft['forecast_non_uwaabo']+=file_ticks[_idn+_idh]['price']-_ft['price']
            _ft['forecast_non_uwaabo']/=cwcn_config.FORECAST_HORIZONS
        file_ticks=file_ticks[:-cwcn_config.FORECAST_HORIZONS]
        # --- --- --- save
        self._request_file_path = os.path.join(cwcn_config.CWCN_SIMULATION_CONFIG.DATA_FOLDER,"{}.forecast{}".format(cwcn_config.CWCN_INSTRUMENT_CONFIG.SYMBOL,cwcn_config.CWCN_SIMULATION_CONFIG.DATA_EXTENSION))
        with open(self._request_file_path,'w') as _f_:
            for _ft_ in file_ticks:
                _f_.write('{},\n'.format(_ft_))
    def _reset_(self):
        self._instrument_state=None
    def _load_file_(self):
        self._request_file=open(self._request_file_path,'r')
        for _ in range(cwcn_config.CWCN_SIMULATION_CONFIG.SKIP_N_DATA):
            next(self._request_file)
    def get_trade_history(self, symbol):
        """
        List the last cwcn_config.CWCN_SIMULATION_CONFIG.GET_HISTORY_LEN trades for a symbol.
        The most recent tick is placed in possition [0]
        """
        assert(symbol==cwcn_config.CWCN_INSTRUMENT_CONFIG.SYMBOL), "wrong symbol detected on trade history (simulation)"
        c_history = [self._load_tick_() for _ in range(cwcn_config.CWCN_SIMULATION_CONFIG.GET_HISTORY_LEN)]
        c_history.reverse()
        return c_history
    def simulation_step_ticker(self, symbol):
        """
        The real-time ticker includes the last traded price, the last traded size, transaction ID, the side of liquidity taker, the best bid price and size, the best ask price and size as well as the transaction time of the orders.
        These messages can also be obtained through Websocket. The Sequence Number is used to judge whether the messages pushed by Websocket is continuous."""
        assert(symbol==cwcn_config.CWCN_INSTRUMENT_CONFIG.SYMBOL), "wrong symbol detected on get ticker (simulation)"
        if(self.ei._step_updates_()):
            self.ei._invoke_ujcamei_cajtucu_()
            return self._instrument_state
        else:
            return False
    def _load_tick_(self):
        # dont use, use simulation_step_ticker instead
        try:
            c_tick=next(self._request_file).replace(',\n','')
            self._instrument_state=ast.literal_eval(c_tick)
            return self._instrument_state
        except:
            self._request_file.close()
            self._load_file_()
            return False
    def _get_tick_price_(self):
        return self._instrument_state['price']
# --- --- --- TRADE
class TradeClient:
    def __init__(self, _exchange_instrument):
        self.ei=_exchange_instrument
        self._reset_()
    def _reset_(self):
        self._position_overview = None
        self.client_orders = []
    def _close_open_orders_by_prob_(self):
        for _i in range(len(self.client_orders)):
            if(self.client_orders[_i]['isOpen'] and random.random()<cwcn_config.CWCN_SIMULATION_CONFIG.CLOSE_ORDER_PROB):
                self.client_orders[_i]['isOpen'] = False
                assert(self.client_orders[_i]['price'] is None), "unexpected behaviour when closing orders"
                self.client_orders[_i]['price']=self.ei.market_instrument._get_tick_price_()
                logging.orders_logging("[order] is now closed : {}, with price : {}".format(\
                    self.client_orders[_i]['clientOid'],self.client_orders[_i]['price']))
    def _update_current_orders_state_pnl_(self):
        for _i in range(len(self.client_orders)):
            if(not self.client_orders[_i]['isOpen']):
                self.client_orders[_i]['orderPnl']=\
                    (self.ei.market_instrument._get_tick_price_()-self.client_orders[_i]['price'])*\
                        (cwcn_config.CWCN_INSTRUMENT_CONFIG.MULTIPLER)*\
                            self.client_orders[_i]['size']
    def _get_best_match_order_(self,_side):
        c_ords=sorted([(_i,_o) for _i,_o in enumerate(self.client_orders) if _o['side']==_side],reverse=True,key=lambda x:x[1]['orderPnl'])
        return (c_ords[0]) if len(c_ords)>0 else None
    def _process_order_(self,_dict):
        assert(_dict['symbol']==cwcn_config.CWCN_INSTRUMENT_CONFIG.SYMBOL), "wrong symbol detected on process order (simulation)"
        assert(_dict['size']==1), "unexpected order size" #FIXME allow for different sizes
        if(_dict['side']=='buy'):
            c_match_order=self._get_best_match_order_('sell')
        elif(_dict['side']=='sell'):
            c_match_order=self._get_best_match_order_('buy')
        else:
            aux_str="order side not undersootd : {}".format(_dict)
            assert(False), aux_str
        if(c_match_order is None): # user holds no matching inverse order sell->buy, buy->sell
            self.client_orders.append(_dict)
        else:
            self.client_orders.pop(c_match_order[0])
            self.ei.user_instrument._apply_delta_aviableBalance_(c_match_order[1]['orderPnl'])
            if(c_match_order[1]['orderPnl']>=0):
                sys.stdout.write("{}{}{}\t\t\t\t\t\t\t\t\t\t\t\t{}{}\n".format(cwcn_config.CWCN_CURSOR.CARRIER_RETURN,cwcn_config.CWCN_CURSOR.UP,cwcn_config.CWCN_COLORS.GREEN,c_match_order[1]['orderPnl'],cwcn_config.CWCN_COLORS.REGULAR))
            else:
                sys.stdout.write("{}{}{}\t\t\t\t\t\t\t\t\t\t\t\t{}{}\n".format(cwcn_config.CWCN_CURSOR.CARRIER_RETURN,cwcn_config.CWCN_CURSOR.UP,cwcn_config.CWCN_COLORS.RED,c_match_order[1]['orderPnl'],cwcn_config.CWCN_COLORS.REGULAR))
            sys.stdout.flush()
        self.ei.user_instrument._update_account_overview_() #FIXME maybe redundant
    def _get_isOpen_(self):
        return any([_o['isOpen'] for _o in self.client_orders])
    def _get_currentQty_(self):
        c_qty=0.0
        for _o in self.client_orders:
            c_qty+=+_o['size'] if _o['side']=='buy' else -_o['size'] if _o['side']=='sell' else 0
        return c_qty
    def _get_realisedPnl_(self):
        return self.ei.user_instrument.acc_overview_state['realisedPnl']
    def _get_unrealisedPnl_(self):
        c_unrealisedPnl=0.0
        for _o in self.client_orders:
            c_unrealisedPnl+=_o['orderPnl']
        return c_unrealisedPnl
    def get_position_details(self, symbol):
        """
        Get the position details of a specified position."""
        # {
        #     "id": "5ce3cda60c19fc0d4e9ae7cd",                //Position ID
        #     "symbol": "BTCUSDTPERP",                              //Symbol
        #     "autoDeposit": true,                             //Auto deposit margin or not
        #     "maintMarginReq": 0.005,                         //Maintenance margin requirement
        #     "riskLimit": 200,                                //Risk limit
        #     "realLeverage": 1.06,                            //Leverage of the order
        #     "crossMode": false,                              //Cross mode or not
        #     "delevPercentage": 0.1,                          //ADL ranking percentile
        #     "openingTimestamp": 1558433191000,               //Open time
        #     "currentTimestamp": 1558507727807,               //Current timestamp
        #     "currentQty": -20,                               //Current position
        #     "currentCost": 0.00266375,                       //Current position value
        #     "currentComm": 0.00000271,                       //Current commission
        #     "unrealizedCost": 0.00266375,                    //Unrealized value
        #     "realizedGrossCost": 0,                          //Accumulated realized gross profit value
        #     "realizedCost": 0.00000271,                      //Current realized position value
        #     "isOpen": true,                                  //Opened position or not
        #     "markPrice": 7933.01,                            //Mark price
        #     "markValue": 0.00252111,                         //Mark value
        #     "posCost": 0.00266375,                           //Position value
        #     "posCross": 1.2e-7,                              //Manually added margin
        #     "posInit": 0.00266375,                           //Leverage margin
        #     "posComm": 0.00000392,                           //Bankruptcy cost
        #     "posLoss": 0,                                    //Funding fees paid out
        #     "posMargin": 0.00266779,                         //Position margin
        #     "posMaint": 0.00001724,                          //Maintenance margin
        #     "maintMargin": 0.00252516,                       //Position margin
        #     "realizedGrossPnl": 0,                           //Accumulated realized gross profit value
        #     "realisedPnl": -0.00000253,                      //Realised profit and loss
        #     "unrealisedPnl": -0.00014264,                    //Unrealized profit and loss
        #     "unrealisedPnlPcnt": -0.0535,                    //Profit-loss ratio of the position
        #     "unrealizedRoePcnt": -0.0535,                    //Rate of return on investment
        #     "avgEntryPrice": 7508.22,                        //Average entry price
        #     "liquidationPrice": 1000000,                     //Liquidation price
        #     "bankruptPrice": 1000000,                         //Bankruptcy price
        #     "settleCurrency": "XBT"                         //Currency used to clear and settle the trades     
        # }
        assert(symbol==cwcn_config.CWCN_INSTRUMENT_CONFIG.SYMBOL), "wrong symbol detected on get position details (simulation)"
        self.ei._make_all_step_updates_()
        self._position_overview = {
            "symbol":symbol,
            "isOpen":self._get_isOpen_(),
            "currentQty":self._get_currentQty_(),
            "realisedPnl":self._get_realisedPnl_(),
            "unrealisedPnl":self._get_unrealisedPnl_(),
        }
        return self._position_overview

    def create_market_order(self, symbol, side, size, leverage, client_oid=None, **kwargs):
        """
        Place Market Order Functions
        Param	type	Description
        size	Integer	[optional] amount of contract to buy or sell"""
        # {
        #     "clientOid": "5c52e11203aa677f33e493fb",
        #     "reduceOnly": false,
        #     "closeOrder": false,
        #     "forceHold": false,
        #     "hidden": false,
        #     "iceberg": false,
        #     "leverage": 20,
        #     "postOnly": false,
        #     "price": 8000,
        #     "remark": "remark",
        #     "side": "buy",
        #     "size": 20,
        #     "stop": "",
        #     "stopPrice": 0,
        #     "stopPriceType": "",
        #     "symbol": "BTCUSDTPERP",
        #     "timeInForce": "",
        #     "type": "limit",
        #     "visibleSize": 0
        # }
        client_oid = str(client_oid) if client_oid else str(uuid4())
        params = {
            'symbol'   : symbol,
            'side'     : side,
            'size'     : size,
            'leverage' : leverage,
            'clientOid': client_oid,
            'orderPnl' : 0,
            'price'    : None,
            'isOpen'   : True,
        }
        if kwargs:
            params.update(kwargs)
        self.ei.user_instrument._apply_delta_commission_()
        self._process_order_(params)
        return True #FIXME add all the returns

    def cancel_order(self, order_id):
        # to cancel an order is NOT to clear the position
        if(all([_oo['isOpen'] for _oo in self.client_orders if _oo['clientOid'] == order_id])):
            self.ei.user_instrument._apply_delta_commission_() #FIXME may not be needed in clancel procedure
            self.client_orders = [_oo for _oo in self.client_orders if _oo['clientOid'] != order_id]
        return True #FIXME check if order can be canceled

    def clear_positions(self,symbol):
        # to cancel an order is NOT to clear the position, only open orders can be cancel
        for _o in self.client_orders:
            if(_o['isOpen']):
                self.cancel_order(_o["clientOid"])
            else:
                self.create_market_order(
                    symbol=_o['symbol'],
                    side='buy' if _o['side']=='sell' else 'sell',
                    size=_o['size'],
                    leverage=_o['leverage'],
                )
        return True
    
    # def emergency_clear_positions(self):
    #     self.clear_positions(cwcn_config.CWCN_INSTRUMENT_CONFIG.SYMBOL)
    #     for _o in self.client_orders:
    #         if(not _o['isOpen']):
    #             self.create_market_order(\
    #                 symbol=cwcn_config.CWCN_INSTRUMENT_CONFIG.SYMBOL, 
    #                 side='sell' if _o['side']=='buy' else 'buy', 
    #                 size=_o['size'], 
    #                 leverage=cwcn_config.CWCN_INSTRUMENT_CONFIG.LEVERAGE, 
    #                 client_oid=None)
    #     return True

    
# --- --- --- --- USER
class UserClient:
    # "data": {
    #     "accountEquity": 99.8999305281, //Account equity = marginBalance + Unrealized Pnl 
    #     "unrealisedPnl": 0, //Unrealized profit and loss
    #     "marginBalance": 99.8999305281, //Margin balance = positionMargin + orderMargin + frozenFunds + availableBalance
    #     "positionMargin": 0, //Position margin
    #     "orderMargin": 0, //Order margin
    #     "frozenFunds": 0, //Frozen funds 
    #     "availableBalance": 99.8999305281 //Available balance
    #     "currency": "USDT" //currency code
    # }
    def __init__(self, _exchange_instrument):
        self.ei=_exchange_instrument
        self._reset_()
    def _reset_(self):
        self.acc_overview_state=cwcn_config.CWCN_SIMULATION_CONFIG.INITIAL_WALLET
    def _apply_delta_commission_(self):
        self.acc_overview_state['availableBalance']+=cwcn_config.CWCN_INSTRUMENT_CONFIG.DELTA_COMMISSION
    def _apply_delta_aviableBalance_(self,_delta):
        self.acc_overview_state['realisedPnl']+=_delta
        self.acc_overview_state['availableBalance']+=_delta
    def _update_account_unrealisedPnl_(self):
        self.acc_overview_state['unrealisedPnl']=0.0
        for _o in self.ei.trade_instrument.client_orders:
            self.acc_overview_state['unrealisedPnl']+=_o['orderPnl']
    def _update_account_overview_(self,_dict={}):
        self.acc_overview_state.update(_dict)
        self._update_account_unrealisedPnl_()
        self.acc_overview_state['marginBalance']=\
            self.acc_overview_state['positionMargin']+\
            self.acc_overview_state['orderMargin']+\
            self.acc_overview_state['frozenFunds']+\
            self.acc_overview_state['availableBalance']
        self.acc_overview_state['accountEquity']=\
            self.acc_overview_state['marginBalance']+\
            self.acc_overview_state['unrealisedPnl']
        # --- --- --- 
        # --- --- --- 
    def get_account_overview(self, **kwargs):
        self.ei._make_all_step_updates_()
        return self.acc_overview_state
# --- --- --- --- 
if __name__=='__main__':
    # --- --- --- ---
    c_trade_instrument = EXCHANGE_INSTRUMENT(None)
    # --- --- --- ---
        