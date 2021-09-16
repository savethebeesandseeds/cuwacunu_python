# function farm
# SCRIPT GENERATOR of a general TK_DICT price wave
# {'symbol': 'ADAUSDTPERP', 'sequence': 1623911812775, 'side': 'buy', 'size': 6, 'price': 2.91952, 'bestBidSize': 600, 'bestBidPrice': '2.9158', 'bestAskPrice': '2.91952', 'tradeId': '613381a021865f0ea7dca98c', 'ts': 1630765440905645910, 'bestAskSize': 5},
import numpy as np
if __name__=='__main__':
    _len=10000
    _freq=100
    _lambda=(lambda x:np.sin(2*np.pi*_freq*x))
    _symbol='SINE-{}'.format(_freq)
    _ext='poloniex_ticker_data'
    _file_path='./FUNC/{}.{}'.format(_symbol,_ext)
    with open(_file_path,'w') as _file_:
        for _l in reversed(range(_len)):
            c_dict = {'symbol': _symbol, 
            'sequence': _l, 
            'side': 'buy', 
            'size': 1, 
            'price': _lambda(_l/_len), 
            'bestBidSize': 0, 
            'bestBidPrice': 0, 
            'bestAskPrice': 0, 
            'tradeId': 'nan', 
            'ts': _l, 
            'bestAskSize': 0,}
            _file_.write('{},\n'.format(c_dict))

