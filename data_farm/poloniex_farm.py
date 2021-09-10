# --- --- 
import sys
sys.path.append('../communications/')
# --- --- 
import asyncio
# --- --- 
import poloniex_api
import cwcn_config
# --- --- 
if __name__=='__main__':
    # --- --- --- ---
    # SYMBOL = 'BTCUSDTPERP'
    # --- --- --- --- 
    # --- --- --- --- 
    c_trade_instrument = poloniex_api.EXCHANGE_INSTRUMENT(_is_farm=True)
    asyncio.run(c_trade_instrument._ticker_data_farm_(cwcn_config.CWCN_FARM_CONFIG.FARM_FOLDER))
    # asyncio.run(c_trade_instrument._ws_methods_())
    # c_trade_instrument._user_methods_()
    # c_trade_instrument._market_methods_()