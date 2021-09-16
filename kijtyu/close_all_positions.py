# --- --- --- 
# close_all_positions.py
# --- --- ---  
import sys
sys.path.append('../kijtyu')
import poloniex_api
import logging
import json
# --- --- ---  
import rcsi_utils
import cwcn_config
# --- --- --- --- --- 
if(__name__=='__main__'):
    # --- --- --- ---
    # SYMBOL = 'BTCUSDTPERP'
    # --- --- --- --- 
    import time
    # --- --- --- --- s
    c_trade_instrument = poloniex_api.EXCHANGE_INSTRUMENT(_is_farm=False)
    # time.sleep(30)
    # c_trade_instrument._market_methods_()
    # logging.info(json.dumps(c_trade_instrument.market_instrument.get_trade_history(cwcn_config.CWCN_INSTRUMENT_CONFIG.SYMBOL),indent=4))
    # asyncio.run(c_trade_instrument._ws_methods_())
    # asyncio.run(c_trade_instrument._ticker_data_farm_())
    # c_trade_instrument._user_methods_()
    
    logging.info("clear_positions:")
    c_trade_instrument.trade_instrument.clear_positions()

    # logging.info(json.dumps(c_trade_instrument.user_instrument.get_account_overview(),indent=4))
    # logging.info("get_position_details:")
    # logging.info(json.dumps(c_trade_instrument.trade_instrument.get_position_details(cwcn_config.CWCN_INSTRUMENT_CONFIG.SYMBOL),indent=4))
    # logging.info("get_all_positions:")
    # logging.info(json.dumps(c_trade_instrument.trade_instrument.get_all_positions(),indent=4))

    # logging.info("get_ticker:")
    # logging.info(json.dumps(c_trade_instrument.market_instrument.get_ticker(cwcn_config.CWCN_INSTRUMENT_CONFIG.SYMBOL),indent=4))
