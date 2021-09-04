# --- --- --- --- 
# --- --- --- --- 
import json
import requests
import hmac
import hashlib
import base64
import time
# --- --- --- --- 
import ssl
import certifi
from uuid import uuid4
import websockets
from urllib.parse import urljoin
# --- --- --- --- 
import asyncio
import os
# --- --- --- --- 
import rcsi_utils
import communications_config
# --- --- -- --- 

# --- ---  
# --- --- --- 
# --- --- --- --- 
# --- --- --- --- --- --- CORE
# --- --- --- --- 
# --- --- ---
# --- --- 

# --- --- --- --- TRADE INSTRUMENT

class TRADE_INSTRUMENT:
    def __init__(self):
        # --- --- --- 
        self.ws_client = WsClient(\
            self._on_message_, 
            key=rcsi_utils.RCsi_CRYPT(communications_config.PLX_FUT_ADA_CONFIG.RCsi_KEY,communications_config.PLX_FUT_ADA_CONFIG.API_KEY), 
            secret=rcsi_utils.RCsi_CRYPT(communications_config.PLX_FUT_ADA_CONFIG.RCsi_KEY,communications_config.PLX_FUT_ADA_CONFIG.API_SECRET), 
            passphrase=rcsi_utils.RCsi_CRYPT(communications_config.PLX_FUT_ADA_CONFIG.RCsi_KEY,communications_config.PLX_FUT_ADA_CONFIG.API_PASS)
        )
        # --- --- --- 
        self.rest_client = RestClient(
            key=rcsi_utils.RCsi_CRYPT(communications_config.PLX_FUT_ADA_CONFIG.RCsi_KEY,communications_config.PLX_FUT_ADA_CONFIG.API_KEY), 
            secret=rcsi_utils.RCsi_CRYPT(communications_config.PLX_FUT_ADA_CONFIG.RCsi_KEY,communications_config.PLX_FUT_ADA_CONFIG.API_SECRET), 
            passphrase=rcsi_utils.RCsi_CRYPT(communications_config.PLX_FUT_ADA_CONFIG.RCsi_KEY,communications_config.PLX_FUT_ADA_CONFIG.API_PASS)
        )
        self.market = self.rest_client.market_api()
        # self.trade = self.rest_client.trade_api()
        self.user = self.rest_client.user_api()
    
    def _on_message_(self,msg):
        if msg['topic'] == f'/contract/instrument:{communications_config.PLX_FUT_ADA_CONFIG.SYMBOL}':
            print(f'Get {communications_config.PLX_FUT_ADA_CONFIG.SYMBOL} Index Price: {msg["data"]} : unix time : {time.time()}')
        elif msg['topic'] == f'/contractMarket/execution:{communications_config.PLX_FUT_ADA_CONFIG.SYMBOL}':
            print(f'Last Execution: {msg["data"]} : unix time : {time.time()}')
        elif msg['topic'] == f'/contractMarket/level2:{communications_config.PLX_FUT_ADA_CONFIG.SYMBOL}':
            print(f'Get {communications_config.PLX_FUT_ADA_CONFIG.SYMBOL} Level 2 :{msg["data"]} : unix time : {time.time()}')
    async def _ws_methods_(self):
        await self.ws_client.connect()
        # await self.ws_client.subscribe(f'/contract/instrument:{communications_config.PLX_FUT_ADA_CONFIG.SYMBOL}')
        # await self.ws_client.subscribe(f'/contractMarket/execution:{communications_config.PLX_FUT_ADA_CONFIG.SYMBOL}')
        # await self.ws_client.subscribe(f'/contractMarket/level2:{communications_config.PLX_FUT_ADA_CONFIG.SYMBOL}')
        await self._send_rest_request('GET', '/api/v1/account-overview', auth=self._private)
        await asyncio.sleep(30)
        await self.ws_client.disconnect()
    def _market_methods_(self):
        # Fetch MarketData
        # server_time = self.market.get_server_timestamp()
        # print("[server_time:] {}".format(json.dumps(server_time,sort_keys=True,indent=4)))
        # l3_depth = self.market.get_l3_order_book(communications_config.PLX_FUT_ADA_CONFIG.SYMBOL)
        # print("[l3_depth:] {}".format(json.dumps(l3_depth,sort_keys=True,indent=4)))
        # l2_depth = self.market.get_l2_order_book(communications_config.PLX_FUT_ADA_CONFIG.SYMBOL)
        # print("[l2_depth:] {}".format(json.dumps(l2_depth,sort_keys=True,indent=4)))
        # klines = self.market.get_ticker(communications_config.PLX_FUT_ADA_CONFIG.SYMBOL)
        # print("[klines:] {}".format(json.dumps(klines,sort_keys=True,indent=4)))
        pass
    def _trade_methods_(self):
        # Trade Functions
        # cancel_id = trade.cancel_order(order_id['orderId'])
        # order_id = trade.create_limit_order(communications_config.PLX_FUT_ADA_CONFIG.SYMBOL, 'buy', '1', '30', '8600')
        # order_id = trade.create_limit_order(communications_config.PLX_FUT_ADA_CONFIG.SYMBOL, 'buy', '1', '30', '8600')
        # cancel_all = trade.cancel_all_limit_orders(communications_config.PLX_FUT_ADA_CONFIG.SYMBOL)
        pass
    def _user_methods_(self):
        # User Account Functions
        account_overview = self.user.get_account_overview()
        print("[account_overview:] {}".format(json.dumps(account_overview,sort_keys=True,indent=4)))

# --- --- --- --- SEND REQUEST

class SendRequest:
    def __init__(self, key=None, secret=None, passphrase=None, base_url=None, timeout=5):
        self._key = key
        self._secret = secret.encode('utf-8') if secret else None
        self._passphrase = passphrase
        self._base_url = base_url or communications_config.PLX_FUT_ADA_CONFIG.DEFAULT_BASE_URL
        self._timeout = timeout

    def __call__(self, method, path, params=None, auth=False):
        body = None
        if params:
            if method in ['GET', 'DELETE']:
                params = [f'{key}={value}' for key, value in params.items()]
                params = '&'.join(params)
                path += '?' + params
            else:
                body = json.dumps(params)

        headers = {
            'Content-Type': 'application/json'
        }

        if auth:
            now = int(time.time()) * 1000
            str_to_sign = str(now) + method + path + (body or '')
            signature = hmac.new(self._secret, str_to_sign.encode('utf-8'), hashlib.sha256)
            signature = signature.digest()
            signature = base64.b64encode(signature)

            headers.update({
                'PF-API-SIGN'      : signature,
                'PF-API-TIMESTAMP' : str(now),
                'PF-API-KEY'       : self._key,
                'PF-API-PASSPHRASE': self._passphrase
            })

        url = urljoin(self._base_url, path)

        response = requests.request(method, url, headers=headers, timeout=self._timeout, data=body)

        try:
            payload = response.json()
        except:
            if response.status_code != 200:
                response.raise_for_status()

            raise RuntimeError(response.text)

        if payload['code'] == '200000':
            return payload.get('data', None)

        raise RuntimeError(payload)

# --- ---  
# --- --- --- 
# --- --- --- --- 
# --- --- --- --- --- --- REST
# --- --- --- --- 
# --- --- ---
# --- --- 

# --- --- --- --- CLIENT

class RestClient:
    def __init__(self, key=None, secret=None, passphrase=None, base_url=None):
        self._user_client = UserClient(key, secret, passphrase, base_url)
        self._trade_client = TradeClient(key, secret, passphrase, base_url)
        self._market_client = MarketClient(key, secret, passphrase, base_url)

    def user_api(self):
        return self._user_client

    def trade_api(self):
        return self._trade_client

    def market_api(self):
        return self._market_client

# --- --- --- --- MARKET 

class MarketClient:
    def __init__(self, key=None, secret=None, passphrase=None, base_url=None):
        self._request = SendRequest(key, secret, passphrase, base_url)

    def get_server_timestamp(self):
        """
        Get the API server time. This is the Unix timestamp."""
        return self._request('GET', '/api/v1/timestamp')

    def get_interest_rate(self, symbol, **kwargs):
        """
        Check interest rate list.
        Param	    Type	Description
        symbol	    String	Symbol of the contract
        startAt	    long	[optional] Start time (milisecond)
        endAt	    long	[optional] End time (milisecond)
        reverse	    boolean	[optional] This parameter functions to judge whether the lookup is reverse. True means “yes”. False means no. This parameter is set as True by default.
        offset	    long	[optional] Start offset. The unique attribute of the last returned result of the last request. The data of the first page will be returned by default.
        forward	    boolean	[optional] This parameter functions to judge whether the lookup is forward or not. True means “yes” and False means “no”. This parameter is set as true by default
        maxCount	int	    [optional] Max record count. The default record count is 10"""

        params = {
            'symbol': symbol
        }
        params.update(kwargs)

        return self._request('GET', '/api/v1/interest/query', params)

    def get_index_list(self, symbol, **kwargs):
        """
        Check index list
        Param	    Type	Description
        symbol	    String	Symbol of the contract
        startAt	    long	[optional] Start time (milisecond)
        endAt	    long	[optional] End time (milisecond)
        reverse	    boolean	[optional] This parameter functions to judge whether the lookup is reverse. True means “yes”. False means no. This parameter is set as True by default
        offset	    long	[optional] Start offset. The unique attribute of the last returned result of the last request. The data of the first page will be returned by default.
        forward	    boolean	[optional] This parameter functions to judge whether the lookup is forward or not. True means “yes” and False means “no”. This parameter is set as true by default
        maxCount	int	    [optional] Max record count. The default record count is 10"""

        params = {
            'symbol': symbol
        }
        params.update(kwargs)

        return self._request('GET', '/api/v1/index/query', params)

    def get_current_mark_price(self, symbol):
        """
        Check the current mark price.
        Param	Type	Description
        symbol	String	Path Parameter. Symbol of the contract"""
        return self._request('GET', f'/api/v1/mark-price/{symbol}/current')

    def get_premium_index(self, symbol, **kwargs):
        """
        Submit request to get premium index.
        Param	    Type	Description
        symbol  	String	Symbol of the contract
        startAt 	long	[optional] Start time (milisecond)
        endAt	    long	[optional] End time (milisecond)
        reverse	    boolean	[optional] This parameter functions to judge whether the lookup is reverse. True means “yes”. False means no. This parameter is set as True by default
        offset	    long	[optional] Start offset. The unique attribute of the last returned result of the last request. The data of the first page will be returned by default.
        forward	    boolean	[optional] This parameter functions to judge whether the lookup is forward or not. True means “yes” and False means “no”. This parameter is set as true by default
        maxCount	int	    [optional] Max record count. The default record count is 10"""

        params = {
            'symbol': symbol
        }
        params.update(kwargs)

        return self._request('GET', '/api/v1/premium/query', params)

    def get_current_fund_rate(self, symbol):
        """
        Submit request to check the current mark price."""
        return self._request('GET', f'/api/v1/funding-rate/{symbol}/current')

    def get_trade_history(self, symbol):
        """
        List the last 100 trades for a symbol."""

        params = {
            'symbol': symbol
        }

        return self._request('GET', '/api/v1/trade/history', params)

    def get_l2_order_book(self, symbol):
        """
        Get a snapshot of aggregated open orders for a symbol.
        Level 2 order book includes all bids and asks (aggregated by price). This level returns only one aggregated size for each price (as if there was only one single order for that price).
        This API will return data with full depth.
        It is generally used by professional traders because it uses more server resources and traffic, and we have strict access frequency control.
        To maintain up-to-date Order Book, please use Websocket incremental feed after retrieving the Level 2 snapshot.
        In the returned data, the sell side is sorted low to high by price and the buy side is sorted high to low by price.
        """

        params = {
            'symbol': symbol
        }

        return self._request('GET', '/api/v1/level2/snapshot', params)

    def get_l2_messages(self, symbol, start, end):
        """
        If the messages pushed by Websocket is not continuous, you can submit the following request and re-pull the data to ensure that the sequence is not missing.
        In the request, the start parameter is the sequence number of your last received message plus 1, and the end parameter is the sequence number of your current received message minus 1.
        After re-pulling the messages and applying them to your local exchange order book, you can continue to update the order book via Websocket incremental feed.
        If the difference between the end and start parameter is more than 500, please stop using this request and we suggest you to rebuild the Level 2 orderbook.
        Level 2 message pulling method: Take price as the key value and overwrite the local order quantity with the quantity in messages.
        If the quantity of a certain price in the pushed message is 0, please delete the corresponding data of that price.
        Param	Type	Description
        symbol	String	Symbol of the contract
        start	long	Start sequence number (included in the returned data)
        end	    long	End sequence number (included in the returned data)
        """

        params = {
            'symbol': symbol,
            'start' : start,
            'end'   : end
        }

        return self._request('GET', '/api/v1/level2/message/query', params)

    def get_l3_order_book(self, symbol):
        """
        Get a snapshot of all the open orders for a symbol. Level 3 order book includes all bids and asks (the data is non-aggregated, and each item means a single order).
        This API is generally used by professional traders because it uses more server resources and traffic, and we have strict access frequency control.
        To maintain up-to-date order book, please use Websocket incremental feed after retrieving the Level 3 snapshot.
        In the orderbook, the selling data is sorted low to high by price and orders with the same price are sorted in time sequence.
        The buying data is sorted high to low by price and orders with the same price are sorted in time sequence.
        The matching engine will match the orders according to the price and time sequence.
        The returned data is not sorted, you may sort the data yourselves.
        """

        params = {
            'symbol': symbol
        }

        return self._request('GET', '/api/v1/level3/snapshot', params)

    def get_l3_messages(self, symbol, start, end):
        """
        If the messages pushed by Websocket is not continuous, you can submit the following request and re-pull the data to ensure that the sequence is not missing.
        In the request, the start parameter is the sequence number of your last received message plus 1, and the end parameter is the sequence number of your current received message minus 1.
        After re-pulling the messages and applying them to your local exchange order book, you can continue to update the order book via Websocket incremental feed.
        If the difference between the end and start parameter is more than 500, please stop using this request and we suggest you to rebuild the Level 3 orderbook."""

        params = {
            'symbol': symbol,
            'start' : start,
            'end'   : end
        }

        return self._request('GET', '/api/v1/level3/message/query', params)

    def get_ticker(self, symbol):
        """
        The real-time ticker includes the last traded price, the last traded size, transaction ID, the side of liquidity taker, the best bid price and size, the best ask price and size as well as the transaction time of the orders.
        These messages can also be obtained through Websocket. The Sequence Number is used to judge whether the messages pushed by Websocket is continuous."""

        params = {
            'symbol': symbol
        }

        return self._request('GET', '/api/v1/ticker', params)

    def get_contracts_list(self):
        """
        Submit request to get the info of all open contracts."""

        return self._request('GET', '/api/v1/contracts/active')

    def get_contract_detail(self, symbol):
        """
        Submit request to get info of the specified contract."""

        params = {
            'symbol': symbol
        }

        return self._request('GET', '/api/v1/ticker', params)

# --- --- --- TRADE
class TradeClient:
    def __init__(self, key=None, secret=None, passphrase=None, base_url=None):
        self._request = SendRequest(key, secret, passphrase, base_url)

    def get_fund_history(self, symbol, **kwargs):
        """
        Submit request to get the funding history.
        Param	    Type	Description
        symbol	    String	Symbol of the contract
        startAt	    long	[optional] Start time (milisecond)
        endAt	    long	[optional] End time (milisecond)
        reverse	    boolean	[optional] This parameter functions to judge whether the lookup is forward or not. True means “yes” and False means “no”. This parameter is set as true by default
        offset	    long	[optional] Start offset. The unique attribute of the last returned result of the last request. The data of the first page will be returned by default.
        forward	    boolean	[optional] This parameter functions to judge whether the lookup is forward or not. True means “yes” and False means “no”. This parameter is set as true by default
        maxCount	int	    [optional] Max record count. The default record count is 10"""

        params = {
            'symbol': symbol
        }
        params.update(kwargs)

        return self._request('GET', '/api/v1/funding-history', params, True)

    def get_position_details(self, symbol):
        """
        Get the position details of a specified position."""

        params = {
            'symbol': symbol
        }

        return self._request('GET', '/api/v1/position', params, True)

    def get_all_positions(self):
        """
        Get the position details of a specified position."""

        return self._request('GET', '/api/v1/positions', auth=True)

    def modify_auto_deposit_margin(self, symbol, status=True):
        """
        Enable/Disable of Auto-Deposit Margin"""

        params = {
            'symbol': symbol,
            'status': status
        }

        return self._request('POST', '/api/v1/position/margin/auto-deposit-status', params, True)

    def add_margin_manually(self, symbol, margin, biz_no):
        """
        Add Margin Manually
        Param	Type	    Description
        symbol	String	    Ticker symbol of the contract
        margin	BigDecimal	Margin amount (min. margin amount≥0.00001667XBT）
        biz_no	String	    A unique ID generated by the user, to ensure the operation is processed by the system only once"""

        params = {
            'symbol': symbol,
            'margin': margin,
            'bizNo' : biz_no
        }

        return self._request('POST', '/api/v1/position/margin/deposit-margin', params, True)

    def get_fills_details(self, symbol, **kwargs):
        """
        Get a list of recent fills.
        Param	Type	Description
        orderId	String	[optional] List fills for a specific order only (If you specify orderId, other parameters can be ignored)
        symbol	String	[optional] Symbol of the contract
        side	String	[optional] buy or sell
        type	String	[optional] limit, market, limit_stop or market_stop
        startAt	long	[optional] Start time (milisecond)
        endAt	long	[optional] End time (milisecond)"""

        params = {
            'symbol': symbol
        }
        params.update(kwargs)

        return self._request('GET', '/api/v1/fills', params, True)

    def get_recent_fills(self):
        """
        Get a list of recent 1000 fills in the last 24 hours. If you need to get your recent traded order history with low latency, you may query this endpoint."""

        return self._request('GET', '/api/v1/recentFills', auth=True)

    def get_open_order_details(self, symbol):
        """
        You can query this endpoint to get the the total number and value of the all your active orders."""

        params = {
            'symbol': symbol
        }

        return self._request('GET', '/api/v1/openOrderStatistics', params, True)

    def create_limit_order(self, symbol, side, leverage, size, price, client_oid=None, **kwargs):
        """
        You can place two types of orders: limit and market. Orders can only be placed if your account has sufficient funds.
        Once an order is placed, your funds will be put on hold for the duration of the order. The amount of funds on hold depends on the order type and parameters specified.
        Please be noted that the system would hold the fees from the orders entered the orderbook in advance.
        Read Get Fills to learn more.
        Do NOT include extra spaces in JSON strings.
        The maximum limit orders for a single contract is 100 per account, and the maximum stop orders for a single contract is 50 per account.
        Param	    type	Description
        clientOid	String	Unique order id created by users to identify their orders, e.g. UUID, Only allows numbers, characters, underline(_), and separator(-)
        side	    String	buy or sell
        symbol	    String	a valid contract code. e.g. XBTUSDM
        type	    String	[optional] Either limit or market
        leverage	String	Leverage of the order
        remark  	String	[optional] remark for the order, length cannot exceed 100 utf8 characters
        stop    	String	[optional] Either down or up. Requires stopPrice and stopPriceType to be defined
        stopPrice   Type	String	[optional] Either TP, IP or MP, Need to be defined if stop is specified.
        stopPrice	String	[optional] Need to be defined if stop is specified.
        reduceOnly	boolean	[optional] A mark to reduce the position size only. Set to false by default.
        closeOrder	boolean	[optional] A mark to close the position. Set to false by default.
        forceHold	boolean	[optional] A mark to forcely hold the funds for an order, even though it's an order to reduce the position size. This helps the order stay on the order book and not get canceled when the position size changes. Set to false by default.
        Advanced Order Details:
        Param	    type	Description
        price	    String	Limit price
        size	    Integer	Order size. Must be a positive number
        timeInForce	String	[optional] GTC, IOC(default is GTC), read Time In Force
        postOnly	boolean	[optional] Post only flag, invalid when timeInForce is IOC. When postOnly chose, not allowed choose hidden or iceberg.
        hidden  	boolean	[optional] Orders not displaying in order book. When hidden chose, not allowed choose postOnly.
        iceberg	    boolean	[optional] Only visible portion of the order is displayed in the order book. When iceberg chose, not allowed choose postOnly.
        visibleSize	Integer	[optional] The maximum visible size of an iceberg order"""

        client_oid = str(client_oid) if client_oid else str(uuid4())

        params = {
            'symbol'   : symbol,
            'size'     : size,
            'side'     : side,
            'price'    : price,
            'leverage' : leverage,
            'clientOid': client_oid
        }

        if kwargs:
            params.update(kwargs)

        return self._request('POST', '/api/v1/orders', params, True)

    def create_market_order(self, symbol, side, leverage, client_oid=None, **kwargs):
        """
        Place Market Order Functions
        Param	type	Description
        size	Integer	[optional] amount of contract to buy or sell"""

        client_oid = str(client_oid) if client_oid else str(uuid4())

        params = {
            'symbol'   : symbol,
            'side'     : side,
            'leverage' : leverage,
            'clientOid': client_oid
        }

        if kwargs:
            params.update(kwargs)

        return self._request('POST', '/api/v1/orders', params, True)

    def cancel_order(self, order_id):
        """
        Cancel an order (including a stop order).
        You will receive success message once the system has received the cancellation request.
        The cancellation request will be processed by matching engine in sequence.
        To know if the request has been processed, you may check the order status or update message from the pushes.
        The order id is the server-assigned order id，not the specified clientOid.
        If the order can not be canceled (already filled or previously canceled, etc), then an error response will indicate the reason in the message field."""

        return self._request('DELETE', f'/api/v1/orders/{order_id}', auth=True)

    def cancel_all_limit_orders(self, symbol):
        """
        Cancel all open orders (excluding stop orders). The response is a list of orderIDs of the canceled orders."""

        params = {
            'symbol': symbol
        }

        return self._request('DELETE', '/api/v1/orders', params, True)

    def cancel_all_stop_orders(self, symbol):
        """
        Cancel all untriggered stop orders. The response is a list of orderIDs of the canceled stop orders.
        To cancel triggered stop orders, please use 'Limit Order Mass Cancelation'."""

        params = {
            'symbol': symbol
        }

        return self._request('DELETE', '/api/v1/stopOrders', params, True)

    def get_order_list(self, **kwargs):
        """
        List your current orders.
        Param	Type	Description
        status	String	[optional] active or done, done as default. Only list orders for a specific status
        symbol	String	[optional] Symbol of the contract
        side	String	[optional] buy or sell
        type	String	[optional] limit, market, limit_stop or market_stop
        startAt	long	[optional] Start time (milisecond)
        endAt	long	[optional] End time (milisecond)"""

        return self._request('GET', '/api/v1/orders', kwargs, True)

    def get_open_stop_orders(self, **kwargs):
        """
        Get the un-triggered stop orders list.
        Param	Type	Description
        symbol	String	[optional] Symbol of the contract
        side	String	[optional] buy or sell
        type	String	[optional] limit, market
        startAt	long	[optional] Start time (milisecond)
        endAt	long	[optional] End time (milisecond)"""

        return self._request('GET', '/api/v1/stopOrders', kwargs, True)

    def get_24h_done_orders(self):
        """
        Get a list of recent 1000 orders in the last 24 hours.
        If you need to get your recent traded order history with low latency, you may query this endpoint."""

        return self._request('GET', '/api/v1/recentDoneOrders', auth=True)

    def get_order_details(self, order_id):
        """
        Get a single order by order id (including a stop order).
        Param	Type	Description
        order_id	String	Order ID"""

        return self._request('GET', f'/api/v1/orders/{order_id}', auth=True)

# --- --- --- --- USER
class UserClient:
    def __init__(self, key=None, secret=None, passphrase=None, base_url=None):
        self._request = SendRequest(key, secret, passphrase, base_url)

    def get_account_overview(self, **kwargs):
        """
        Get Account Overview
        Param   	Type	Description
        currency	String	[Optional] Currecny ,including XBT,USDT,Default XBT"""
        return self._request('GET', '/api/v1/account-overview', kwargs, True)

    def get_transaction_history(self, **kwargs):
        """
        If there are open positions, the status of the first page returned will be Pending, indicating the realised profit and loss in the current 8-hour settlement period.
        Please specify the minimum offset number of the current page into the offset field to turn the page.
        Param	    Type	Description
        startAt	    long	[Optional] Start time (milisecond)
        endAt	    long	[Optional] End time (milisecond)
        type	    String	[Optional] Type RealisedPNL-Realised profit and loss, Deposit-Deposit, Withdrawal-withdraw, Transferin-Transfer in, TransferOut-Transfer out
        offset	    long	[Optional] Start offset
        maxCount	long	[Optional] Displayed size per page. The default size is 50
        currency	String	[Optional] Currency of transaction history XBT or USDT"""

        return self._request('GET', '/api/v1/transaction-history', kwargs, True)

# --- ---  
# --- --- --- 
# --- --- --- --- 
# --- --- --- --- --- --- WEB Socket
# --- --- --- --- 
# --- --- ---
# --- --- 


# --- --- --- --- WEB SocketClient

class WsClient:
    def __init__(self, on_message, key=None, secret=None, passphrase=None, base_url=None):
        self.ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS)
        self.ssl_context.verify_mode = ssl.CERT_REQUIRED
        self.ssl_context.check_hostname = True
        self.ssl_context.load_default_certs()
        self.ssl_context.load_verify_locations(os.path.relpath(certifi.where()))
        # --- --- 
        self._on_message = on_message
        self._request = SendRequest(key, secret, passphrase, base_url)
        self._private = key is not None
        self._websocket = None
        self._conn_task = None
        self._conn_event = None
        self._ping_task = None
        self._keep_alive = False
        self._topics = {}
    async def connect(self):
        if self._websocket is not None:
            raise RuntimeError('Already connected to websocket')
        self._conn_event = asyncio.Event()
        self._keep_alive = True
        self._conn_task = asyncio.create_task(self._connect())
        self._ping_task = asyncio.create_task(self._ping())
        try:
            await asyncio.wait_for(self._conn_event.wait(), timeout=60)
        except asyncio.TimeoutError as e:
            self._keep_alive = False
            await self._cancel_ping_task()
            await self._cancel_conn_task()
            self._conn_event = None
            raise RuntimeError('Failed to connect to websocket')
    async def disconnect(self):
        if self._websocket is None:
            raise RuntimeError('Not connected to websocket')
        self._keep_alive = False
        await self._cancel_ping_task()
        await self._cancel_conn_task()
        self._conn_event = None
        self._topics.clear()
    async def _cancel_conn_task(self):
        self._conn_task.cancel()
        try:
            await self._conn_task
        except asyncio.CancelledError:
            pass
        self._conn_task = None
    async def _cancel_ping_task(self):
        self._ping_task.cancel()
        try:
            await self._ping_task
        except asyncio.CancelledError:
            pass
        self._ping_task = None
    def _get_ws_url(self):
        path = '/api/v1/bullet-public'
        if self._private:
            path = '/api/v1/bullet-private'
        token = self._request('POST', path, auth=self._private)
        params = {
            'connectId': uuid4(),
            'token': token['token'],
            'acceptUserMessage': self._private
        }
        params = [f'{key}={value}' for key, value in params.items()]
        params = '&'.join(params)
        url = token['instanceServers'][0]['endpoint']
        url = f'{url}?{params}'
        return url
    async def _connect(self):
        while self._keep_alive:
            try:
                url = self._get_ws_url()
            except:
                await asyncio.sleep(1)
                continue

            try:
                async with websockets.connect(url, ssl=self.ssl_context) as socket:
                    self._websocket = socket
                    self._conn_event.set()

                    for topic, kwargs in self._topics.items():
                        await self.subscribe(topic, **kwargs)

                    while self._keep_alive:
                        try:
                            msg = await socket.recv()
                            msg = json.loads(msg)
                        except json.decoder.JSONDecodeError:
                            pass
                        else:
                            try:
                                self._on_message(msg)
                            except:
                                pass
            except:
                # sleep before reconnecting
                await asyncio.sleep(1)
                continue
            finally:
                self._websocket = None
                self._conn_event.clear()
    async def subscribe(self, topic, **kwargs):
        msg = {
            'id': str(uuid4()),
            'privateChannel': False,
            'response': True
        }
        msg.update(kwargs)
        msg.update({
            'type': 'subscribe',
            'topic': topic
        })
        await self._send_socket_message(msg)
        self._topics[topic] = kwargs
    async def unsubscribe(self, topic, **kwargs):
        msg = {
            'id': str(uuid4()),
            'privateChannel': False,
            'response': True
        }
        msg.update(kwargs)
        msg.update({
            'type': 'unsubscribe',
            'topic': topic
        })
        await self._send_socket_message(msg)
        if topic in self._topics:
            del self._topics[topic]
    async def _ping(self):
        while self._keep_alive:
            await self._conn_event.wait()
            msg = {
                'type': 'ping',
                'id': str(uuid4())
            }
            try:
                await asyncio.wait_for(self._send_socket_message(msg), timeout=10)
            except:
                pass
            await asyncio.sleep(50)
    async def _send_socket_message(self, msg):
        if self._websocket is None:
            raise RuntimeError('Not connected to websocket')
        msg = json.dumps(msg)
        await self._websocket.send(msg)
    async def _send_rest_request(self,method,url,auth=None):
        responce = await self._request(method, url, auth=self._private)
        print("-> responce: {}".format(responce))
        print("-> responce: {}".format(responce))

if __name__=='__main__':
    # --- --- --- ---
    # SYMBOL = 'BTCUSDTPERP'
    # --- --- --- --- 
    # --- --- --- --- 
    c_trade_instrument = TRADE_INSTRUMENT()
    # asyncio.run(c_trade_instrument._ws_methods_())
    c_trade_instrument._user_methods_()
    c_trade_instrument._market_methods_()