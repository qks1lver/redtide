import json
import os
import requests
from datetime import datetime
from time import sleep
from lxml import html
from collections import defaultdict
from scipy.stats import linregress, ttest_ind
from multiprocessing import Pool

from src.constants import URL_YAHOO_PROFILE, DIR_DATA
from src.common import str_money, gen_symbol_batches


class Order(object):
    def __init__(self, symbol, method, shares, price_atm=None, status=None):
        self.date = datetime.now()
        self.symbol = symbol
        self.method = method
        self.shares = shares
        self.price_atm = price_atm      # price at the moment of transaction
        self.status = status

    @property
    def price(self):
        if self.status is None:
            return None
        return self.status.get('price')

    @property
    def order_status(self):
        if self.status is None:
            return None
        return self.status.get('status')

    @property
    def value(self):
        if self.status is None:
            return None
        return self.status.get('value')


class Orders(object):
    def __init__(self):
        self.history = []
        self.stock_history_idx = defaultdict(list)

    def has_stock(self, symbol):
        return symbol in self.stock_history_idx

    def append(self, order):
        self.history.append(order)
        idx = len(self.history) - 1
        self.stock_history_idx[order.symbol].append(idx)

    def stock_recent_order(self, symbol):
        if symbol in self.stock_history_idx:
            return self.history[self.stock_history_idx[symbol][-1]]
        return None


class Stock(object):

    def __init__(self, symbol='', auto_update=True, cache_stale_sec=15):
        self.symbol = symbol
        self.url = URL_YAHOO_PROFILE % symbol if symbol else ''
        self.cached_response = None
        self.cache_datetime = None
        self.cache_stale_sec = cache_stale_sec
        self.metrics = {}
        self._trend_data = {'dt': [], 'price': [], 'vol': []}

        # write any data collected in the day to file
        if self.symbol:
            d_live = os.path.join(DIR_DATA, 'live_quotes')
            if not os.path.isdir(d_live):
                os.makedirs(d_live)
            self.p_quotes = os.path.join(d_live, self.symbol)
        else:
            self.p_quotes = None

        self.__cache_datetime = None
        self.__price = None
        self.__volume = None

        self.auto_update = auto_update
        if auto_update:
            self.update()

    def _msg(self, msg):
        return msg + ' for %s' % self.symbol

    @staticmethod
    def to_str(v):
        return str_money(v, decimal=2, comma=False)

    def update(self):
        if self.url:
            self.cached_response = requests.get(self.url)
            self.cache_datetime = datetime.now()
            self.load_data()
            self.write_quote()
            price = self.metrics.get('price')
            volume = self.metrics.get('volume')
            if price and volume:
                if self.__cache_datetime and self.__price and self.__volume:
                    self._trend_data['dt'].append((self.cache_datetime - self.__cache_datetime).total_seconds())
                    self._trend_data['price'].append(price / self.__price - 1)
                    self._trend_data['vol'].append(volume / self.__volume - 1)
                else:
                    # delta-time is referenced to start
                    # this is risky, hoping price and volume are always populated in metrics upon update
                    self.__cache_datetime = self.cache_datetime
                    self.__price = price
                    self.__volume = volume
            else:
                print(self._msg('! trend update skipped (time={} price={} volume={})'.format(
                    self.cache_datetime, price, volume)))
        else:
            print(self._msg('No URL, stock not updated'))
        return self

    def write_quote(self):
        if self.p_quotes is not None:
            _ = open(self.p_quotes, 'a+').write(json.dumps(self.metrics) + '\n')

    def _parse_json_data(self):
        """
        Parse real-time data JSON payload Yahoo uses
        to update the webpage.

        :return: dict
        """
        nodes = html.fromstring(self.cached_response.text).xpath(
            "//script[contains(text(), '{\"context')]")
        if nodes:
            try:
                i0 = nodes[0].text.find('{"context')
                i1 = nodes[0].text.rfind('};') + 1
                return json.loads(nodes[0].text[i0:i1])
            except:
                return None
        return None

    @staticmethod
    def _load_data(data):
        x = {}
        for k, v in data.items():
            if v:
                if isinstance(v, dict) and 'raw' in v:
                    x[k] = v['raw']
                else:
                    x[k] = v
        return x

    def _populate_metrics(self, data):
        self.metrics['datetime'] = self.cache_datetime.isoformat()
        price_data = self._load_data(data.get('price'))
        self.metrics['price'] = price_data.get('regularMarketPrice')
        self.metrics['change'] = price_data.get('regularMarketChange')
        self.metrics['volume'] = price_data.get('regularMarketVolume')
        self.metrics['high'] = price_data.get('regularMarketDayHigh')
        self.metrics['low'] = price_data.get('regularMarketDayLow')
        self.metrics['shares'] = price_data.get('sharesOutstanding')
        self.metrics['market_cap'] = price_data.get('marketCap')
        self.metrics['currency'] = price_data.get('currency')
        self.metrics['previous_close'] = price_data.get('regularMarketPreviousClose')
        self.metrics['open'] = price_data.get('regularMarketOpen')
        summary_data = self._load_data(data.get('summaryDetail'))
        self.metrics['volumne10days'] = summary_data.get('averageVolume10days')
        self.metrics['bid'] = summary_data.get('bid')
        self.metrics['bid_size'] = summary_data.get('bidSize')
        self.metrics['ask'] = summary_data.get('ask')
        self.metrics['ask_size'] = summary_data.get('askSize')

    def load_data(self):
        data = self._parse_json_data()
        if data is None:
            print('Failed to parse data for %s' % self.symbol)
        else:
            try:
                self._populate_metrics(data['context']['dispatcher']['stores']['QuoteSummaryStore'])
            except:
                print(self._msg('Could not populate metrics'))
        return self

    def update_if_stale(self):
        if self.auto_update and (self.cache_datetime is None
                or (datetime.now() - self.cache_datetime).total_seconds() > self.cache_stale_sec):
            self.update()
            return True
        return False

    @property
    def currency(self):
        return self.metrics.get('currency')

    @property
    def price(self):
        self.update_if_stale()
        return self.metrics.get('price')

    @property
    def open_price(self):
        return self.metrics.get('open')

    @property
    def bid(self):
        self.update_if_stale()
        return self.metrics.get('bid')

    @property
    def ask(self):
        self.update_if_stale()
        return self.metrics.get('ask')

    @property
    def bid_size(self):
        self.update_if_stale()
        return self.metrics.get('bid_size')

    @property
    def ask_size(self):
        self.update_if_stale()
        return self.metrics.get('ask_size')

    @property
    def volume(self):
        self.update_if_stale()
        return self.metrics.get('volume')

    @property
    def open_close_change(self):
        if self.metrics.get('open') and self.metrics.get('previous_close'):
            return self.metrics['open'] / self.metrics['previous_close'] - 1
        return None

    @property
    def ask_bid_ratio(self):
        self.update_if_stale()
        if self.metrics.get('ask_size') and self.metrics.get('bid_size'):
            return self.metrics['ask_size'] / self.metrics['bid_size']
        return None

    def price_trend(self, k=5):
        """
        Return linregress result
        res.slope
        res.rvalue
        res.pvalue (null: slope = 0)
        res.stderrs
        :param k:
        :return:
        """
        self.update_if_stale()
        if len(self._trend_data['dt']) >= k:
            x = self._trend_data['dt'][-k:]
            y = self._trend_data['price'][-k:]
            return linregress(x, y)
        return None

    def volume_trend(self, k=5):
        self.update_if_stale()
        if len(self._trend_data['dt']) >= k:
            x = self._trend_data['dt'][-k:]
            y = self._trend_data['vol'][-k:]
            return linregress(x, y)
        return None


class StatRes(object):
    def __init__(self):
        self.__attributes = defaultdict(list)
        self.__appended = False
        self.__built = False

    @property
    def can_build(self):
        return self.__appended

    def __repr__(self):
        if self.__built:
            msg = 'StatRes({})'.format(
                ', '.join(['{}={}'.format(k, getattr(self, k))
                           for k in self.__attributes]))
            return msg
        else:
            return 'StatRes()'

    def append(self, k, v):
        if v is not None:
            self.__attributes[k].append(v)
            self.__appended = True

    def build(self):
        for k, v in self.__attributes.items():
            if v:
                setattr(self, k, sum(v)/len(v))
            else:
                setattr(self, k, None)
            self.__built = True
        return self


class Stocks(object):

    def __init__(self, symbols=None):
        self.stocks = defaultdict(Stock)

        self.add_symbols(symbols)

        self.__n_cpus = os.cpu_count()

    @property
    def n_stocks(self):
        return len(self.stocks)

    @property
    def symbols(self):
        return list(self.stocks.keys())

    def has_stock(self, symbol):
        return symbol in self.stocks

    def get_stock(self, symbol):
        if self.has_stock(symbol):
            return self.stocks[symbol]
        return None

    def add_symbols(self, symbols):
        if isinstance(symbols, list):
            for s in symbols:
                if self.has_stock(s):
                    continue
                try:
                    self.add_stock(Stock(s, auto_update=False))
                except:
                    print('Could not add {}'.format(s))
        elif isinstance(symbols, str) and not self.has_stock(symbols):
            try:
                self.add_stock(Stock(symbols, auto_update=False))
            except:
                print('Count not add {}'.format(symbols))

    def add_stock(self, stock):
        if not isinstance(stock, Stock):
            raise ValueError('stock arg must be a Stock object')

        if stock.symbol not in self.stocks:
            self.stocks[stock.symbol] = stock
            return True
        return False

    def remove(self, symbol):
        if self.has_stock(symbol):
            self.stocks.pop(symbol)
            return True
        return False

    def _update(self, symbol):
        return self.stocks[symbol].update()

    def update(self, symbol=None, batch_scale=10):
        if symbol is None:
            batches = gen_symbol_batches(self.symbols, batch_size=int(self.__n_cpus * batch_scale))
            for i, batch in enumerate(batches):
                with Pool(processes=self.__n_cpus) as pool:
                    res = pool.map(self._update, batch)
                for s in res:
                    self.stocks[s.symbol] = s
                sleep(1)
            return True
        elif self.has_stock(symbol):
            self.stocks[symbol].update()
            return True
        return False

    def price_trend(self, symbol=None, k=5, metric='rvalue', baseval=0):
        if k < 3:
            raise ValueError('Stocks.price_trend: k >= 3 is a must')
        if metric not in ['slope', 'rvalue', 'pvalue', 'stderr']:
            raise ValueError('Stocks.price_trend: invalid metric: {}'.format(metric))
        if symbol is None:
            values = []
            for s in self.stocks.values():
                r = s.price_trend(k)
                if r:
                    values.append(getattr(r, metric))
            n = len(values)
            if n >= 2:
                # n depends on stocks watching, meanless if few
                return ttest_ind(values, [baseval]*n, equal_var=False)
        elif self.has_stock(symbol):
            return self.stocks[symbol].price_trend(k)
        return None

    def volume_trend(self, symbol=None, k=5, metric='rvalue', baseval=0):
        if k < 3:
            raise ValueError('Stocks.volume_trend: k >= 3 is a must')
        if metric not in ['slope', 'rvalue', 'pvalue', 'stderr']:
            raise ValueError('Stocks.volume_trend: invalid metric: {}'.format(metric))
        if symbol is None:
            values = []
            for s in self.stocks.values():
                r = s.volume_trend(k)
                if r:
                    values.append(getattr(r, metric))
            n = len(values)
            if n >= 2:
                return ttest_ind(values, [baseval]*n, equal_var=False)
        elif self.has_stock(symbol):
            return self.stocks[symbol].volume_trend(k)
        return None
