import numpy as np
from time import sleep, time
from datetime import datetime
from collections import defaultdict

from analysis.financials import FinancialAnalysis
from src.common import get_wallstreet_time
from src.models import Stocks
from src.api import HoodAPI


class TradeBot(object):
    """
    Trade automatically on Robinhood
    Main strategy is modified scalping:
        no upper bound, but moving lower bound as price increase.
    1. Identify companies with great earning histories
       and meet all following criteria
        - Positive net profit past 4 years
        - Positive net profit past 4 quarters
        - high market cap, 2 billion
        - high daily volume 1 million
    2. Identify ones that opened high and remained high
       with no down slope during first 10 minutes since
       market open, and sort them by this criteria
       (note: of the stocks opened higher than previous close,
       the ones with smallest high/close ratio are selected
       - this is based on personal observation, my guess is
       if it opens too high, there's less room to go up more??)
       Also, only 3 x n_splits will be selected, so we don't
       have to track too many stocks.
    3. Of these, select top N stocks and evenly distribute
       allowance among them. N >= 10 and shares >= 1
       Use (allowance / N) > cost_per_share to scan the list
       of stocks from top to bottom and retrieve ones that
       qualifies. If reached the end and still partitions
       left unassigned, assign them to already assigned stock
       by simply scanning the list again
       The goal is to maximize diversity, chances are diversity
       is more important than any other metrics
    4. Onces the stocks are selected and their shares to buy
       are calculated, make these order immediately with
       market price.
    5. Make sell order only if price drops below lower bound
       (i.e. -0.5%) of the previously polled value, which include
       buy order price
    6. Once a buy-sell is complete, this partition is free to
       be assigned to any stock (not currently assigned). Take
       the top 10 qualifying stocks (price < partition) and
       monitor their movement for 3 minutes, then reassign this
       partition if trend is good as evaluated before.
    7. Repeat step 6. until market close.

    S1. Watch the overall market trend. If all the assigned stocks
        are all going down, then sell everything and stop. Watch
        the market using the top N (in step 2) as representatives.
        If market goes back up, then do step 6.
    """
    def __init__(self, trader=None, allowance=1000, n_splits=10, max_loss=0.95):
        print('Initializing TradeBotAPI ...')

        # initialize trader
        if trader == 'robinhood':
            self.trader = HoodAPI()
        elif trader == 'paper':
            # TODO - paper trade system
            self.trader = None
        else:
            self.trader = None

        self.trade_end_offset = (0, 10)     # offset by 10 min
        self.allowance = allowance
        self.net_worth = allowance
        self.max_loss = allowance * max_loss
        self.n_splits = n_splits
        self.partition_size = np.floor(self.allowance / self.n_splits)
        self.fa = FinancialAnalysis()
        self.symbols_qualified = [s for s in self.fa.greats if self.fa.price(s) and self.fa.price(s) < self.partition_size]
        self.stocks = Stocks(self.symbols_qualified)
        self.holding = {}   # dict of buy price
        self.pending_buy = []
        self.pending_sell = []
        self.pending_cancel = []

        self.__watch_interval_sec = 45
        self.__watch_iters = 10
        self.__trend_window = 5
        self.__paritions_used = 0   # increment when buy order made, decrement only when sold or buy order canceled
        self.lb_ratio = -0.005
        self.cached_price = {}
        print('TradeBotAPI initialized.')

    def run(self):
        wst = get_wallstreet_time()
        if not wst['is_market_open']:
            print("Waiting {:.2f} hours for market to open".format(wst['open_in']/3600))
            sleep(wst['open_in'])
        print('Market now open')
        if self.trader is not None:
            self.begin_trade()
        else:
            print('No trader')

    def begin_trade(self):
        """
        Call this at market open to find stocks to scalp.
        Watch the first 10 iterations before calculating trend.
        :return:
        """
        print('\nBegin trading ...\n')

        # 1. scan for open-high stocks
        # Take 60 - 75 sec for 342 stocks
        # About 5 sec for 30
        print('Scanning %d stocks for open-high ...' % self.stocks.n_stocks)
        t0 = time()
        self.stocks.update()
        remove_stocks = []
        for symb, stock in self.stocks.stocks.items():
            if stock.open_close_change is None or stock.open_close_change <= 0:
                remove_stocks.append(symb)
        for s in remove_stocks:
            self.stocks.remove(s)
        print('|- scan took {:.2f} sec'.format(time() - t0))

        # 2. Sort open-close ratio from low to high
        # take the first N-split X 3 to watch for
        # It seems like the ones that open too high do not growth much
        # but the ones the open slighly high are more likely to grow
        symbs = np.array(self.stocks.symbols)
        changes = np.array([self.stocks.get_stock(s).open_close_change for s in symbs])
        idx = np.argsort(changes)
        n_track = 3 * self.n_splits
        if len(symbs) > n_track:
            remove_stocks = symbs[idx][n_track:]
            for s in remove_stocks:
                self.stocks.remove(s)
        self.symbols_qualified = self.stocks.symbols
        print('Tracking %d qualifying stocks' % self.stocks.n_stocks)

        # 3. Conitnue to monitor the qualifying stocks for
        # more iterations
        for i_iter in range(self.__watch_iters):
            sleep(self.__watch_interval_sec)
            self.stocks.update()
            print('|- watch iter {} / {}'.format(i_iter+1, self.__watch_iters))

        # 4. run sequence until trade end or if there are pendings
        wst = get_wallstreet_time(offset_close=self.trade_end_offset)
        while wst['is_market_open'] or self.has_pending:
            self.trade_sequence()
            if self.net_worth <= self.max_loss:
                print('! Reach max loss, selling/cancelling everything.')
                if self.pending_buy:
                    self.cancel_all_pending(method='buy')
                if self.holding:
                    self.batch_order(list(self.holding.keys()), 'sell')
                break
            sleep(self.__watch_interval_sec)

        # 5. close trader
        self.trader.quit()
        print('\nHappy trade day!')
        print('${:,.2f} ===> ${:,.2f}'.format(self.allowance, self.net_worth))

    def sort_buyables(self):
        wst = get_wallstreet_time(offset_close=self.trade_end_offset)
        if not wst['is_market_open']:
            # prevent buy when near end of day
            return None

        rvalues = [self.stocks.get_stock(s).price_trend(k=self.__trend_window).rvalue
                   for s in self.stocks.symbols]
        idx = np.argsort(rvalues)[::-1]
        symbs = np.array(self.stocks.symbols)[idx]
        buy_symbs = []
        for s in symbs:
            if s not in self.holding \
                    and s not in self.pending_buy \
                    and s not in self.pending_sell \
                    and s not in self.pending_cancel:
                buy_symbs.append(s)
        return buy_symbs

    def sell_criteria(self, symbol):
        wst = get_wallstreet_time(offset_close=self.trade_end_offset)
        if wst['is_market_open']:
            # sell everything by end of day
            return True

        if symbol not in self.holding or symbol in self.pending_sell:
            return False

        stat = self.stocks.price_trend(symbol, k=self.__trend_window)
        stock = self.stocks.get_stock(symbol)
        # diff is relative to previous cached price
        diff = stock.price - self.cached_price[symbol]
        lb = self.lb_ratio * stock.open_price
        print(': {} rval={:.5f} pval={:.5f} diff={:.2f} lb={:.2f}'.format(
            symbol, stat.rvalue, stat.pvalue, diff, lb))
        if diff <= lb:
            print('sell criteria ({}): below lower bound'.format(symbol))
            return True
        # elif stat.pvalue <= 0.1 and stat.rvalue < 0:
        #     # Too sensitive at the moment
        #     print('sell criteria ({}): trending down'.format(symbol))
        #     return True
        return False

    @property
    def partitions_remain(self):
        return self.n_splits - self.__paritions_used

    @property
    def has_pending(self):
        if self.pending_buy:
            return True
        elif self.pending_sell:
            return True
        elif self.pending_cancel:
            return True
        return False

    def trade_sequence(self):
        print('\n___ sequence {} ______'.format(datetime.now()))
        # check pending status
        self.get_all_pending_status()

        # get updates on qualified stocks
        # don't update cached_price yet,
        # need the previous cached_price
        # to determine sell criteria
        self.stocks.update()

        # check if there are stocks that should be
        # sold off
        for s in self.stocks.symbols:
            # sell stocks that should be dumped off
            if self.sell_criteria(s) and self.sell(s):
                self.pending_sell.append(s)

        # check global trend and make sure still in trade period
        # TODO - sell/buy criteria are bad! Need to figure out
        #  a decent strategy
        stat = self.stocks.price_trend(k=self.__trend_window)
        global_statistic = stat.statistic
        global_pvalue = stat.pvalue
        print(': Global stat={} pval={}'.format(global_statistic, global_pvalue))
        wst = get_wallstreet_time(offset_close=self.trade_end_offset)
        if wst['is_market_open'] and global_statistic > 4 and global_pvalue < 1e-4:
            # see if there are partitions available
            # to buy stocks that are worth it
            if self.partitions_remain > 0:
                # get all the symbols worth investing
                buyable_symbols = self.sort_buyables()

                # this tracks N x parition for each symbol
                # so if there are more partitions left than
                # buyable_symbols, same symbol can be assigned
                # more than 1 partition
                stock_partitions = defaultdict(int)

                if buyable_symbols:
                    for i in range(self.partitions_remain):
                        symb = buyable_symbols[i % len(buyable_symbols)]
                        stock_partitions[symb] += 1
                    for symb, p in stock_partitions.items():
                        if self.buy(symb, p * self.partition_size):
                            self.pending_buy.append(symb)
        elif not wst['is_market_open'] or (global_statistic < -4 and global_pvalue < 1e-4):
            if not wst['is_market_open']:
                print('! End of day soon, selling everything...')
            # sell all and cancel all buy orders
            if self.pending_buy:
                self.cancel_all_pending('buy')
            if self.holding:
                self.batch_order(list(self.holding.keys()), 'sell')
        else:
            print('! Does not meet buy or sell criteria, continue watching market ...')

        # update cached_price
        for s in self.cached_price:
            if self.stocks.get_stock(s).price:
                self.cached_price[s] = self.stocks.get_stock(s).price


    def _check_order_complete_status(self, symbol, target='Done', max_try=10):
        if target not in ['Done', 'Canceled']:
            raise ValueError('target arg must be either Done or Canceled')
        status = self.trader.order_status(symbol)
        if status is None:
            print('>>> lost track of order for {} <<<'.format(symbol))
            return False
        i_try = 0
        while status['status'] != target and i_try < max_try:
            sleep(1)
            status = self.trader.order_status(symbol)
            i_try += 1
        if status['status'] == target:
            recent_order = self.trader.orders.stock_recent_order(symbol)
            # update cached order status
            recent_order.status = status
            if target == 'Canceled':
                msg_head = '[x] Canceled {} order'.format(symbol)
            elif status.get('type'):
                if 'Buy' in status.get('type'):
                    msg_head = '[+] Bought {} shared of {} at {}'.format(status.get('shares'), symbol, status.get('price'))
                elif 'Sell' in status.get('type'):
                    msg_head = '[-] Sold {} shares of {} at {}'.format(status.get('shares'), symbol, status.get('price'))
                else:
                    print('[?] Unknown status type for {}: {}'.format(symbol, status.get('type')))
                    return False
            else:
                print('[?] Status for {} is missing type'.format(symbol))
                return False

            print(msg_head, 'succesfully!')
            return True
        else:
            print('[?] Cannot confirm if status of {} order is {}'.format(symbol, target))
            return False

    def buy(self, symbol, partition_size=None):
        if partition_size is None:
            partition_size = self.partition_size
        shares = int(np.floor(partition_size / self.stocks.get_stock(symbol).update().price))
        try:
            print('Buying {} shares of {} ...'.format(shares, symbol))
            if self.trader.make_order('buy', symbol, shares=shares, order_type='market'):
                print('|- sent buy order')
            else:
                print('|- failed to send buy order')
                return False
        except Exception as e:
            print('|- failed to buy {} ({} shares), exception: {}'.format(symbol, shares, e))
            return False
        self.__paritions_used += 1
        return True

    def sell(self, symbol):
        if not self.trader.orders.has_stock(symbol):
            print('No {} shares to sell'.format(symbol))
            return False
        shares = self.trader.orders.stock_recent_order(symbol).shares
        try:
            print('Selling {} shares of {} ...'.format(shares, symbol))
            if self.trader.make_order('sell', symbol, order_type='market'):
                print('|- sent sell order')
            else:
                print('|- failed to send sell order')
                return False
        except Exception as e:
            print('|- failed to sell {}, exception: {}'.format(symbol, e))
            return False
        return True

    def cancel(self, symbol):
        if symbol in self.pending_buy:
            if self.trader.cancel_order(symbol):
                self.pending_buy.remove(symbol)
                self.pending_cancel.append(symbol)
                return True
        elif symbol in self.pending_sell:
            if self.trader.cancel_order(symbol):
                self.pending_sell.remove(symbol)
                self.pending_cancel.append(symbol)
                return True
        else:
            print('! No pending buy or sell for {}'.format(symbol))
        return False

    def cancel_all_pending(self, method):
        if method not in ['buy', 'sell']:
            raise ValueError('method must be either buy or sell')
        if method == 'buy':
            for symbol in self.pending_buy:
                if self.trader.cancel_order(symbol):
                    self.pending_buy.remove(symbol)
                    self.pending_cancel.append(symbol)
        else:
            for symbol in self.pending_sell:
                if self.trader.cancel_order(symbol):
                    self.pending_sell.remove(symbol)
                    self.pending_cancel.append(symbol)
        self.get_all_pending_status()

    def batch_order(self, symbols, method, gap_sec=1):
        """
        This make batch orders, but does not check status

        :param symbols:
        :param method:
        :param gap_sec:
        :return:
        """
        if method not in ['buy', 'sell']:
            raise ValueError('method must be either buy or sell')
        for symbol in symbols:
            if method == 'buy':
                if symbol not in self.holding \
                        and symbol not in self.pending_buy \
                        and symbol not in self.pending_sell \
                        and symbol not in self.pending_cancel:
                    if self.buy(symbol):
                        self.pending_buy.append(symbol)
                    sleep(gap_sec)
                else:
                    print('! Cannot buy: already pending or holding {}'.format(symbol))
            else:
                if symbol in self.holding and symbol not in self.pending_sell:
                    if self.sell(symbol):
                        self.pending_sell.append(symbol)
                    sleep(gap_sec)
                else:
                    print('! Cannot sell: not holding {}'.format(symbol))

    def update_net_worth(self, buy_value, sell_value):
        trade_value = sell_value - buy_value
        self.net_worth += trade_value
        if trade_value > 0:
            msg = '[ GAIN ]'
        elif trade_value < 0:
            msg = '[ LOSS ]'
        else:
            msg = '[ EVEN ]'
        msg += ' ${:,.2f} ==> net ${:,.2f}'.format(trade_value, self.net_worth)
        print(msg)

    def get_all_pending_status(self):
        for symbol in self.pending_buy:
            if self._check_order_complete_status(symbol, 'Done'):
                self.pending_buy.remove(symbol)
                order = self.trader.orders.stock_recent_order(symbol)
                self.holding[symbol] = order.value
                self.cached_price[symbol] = order.price
        for symbol in self.pending_sell:
            if self._check_order_complete_status(symbol, 'Done'):
                self.__paritions_used -= 1
                self.pending_sell.remove(symbol)
                order = self.trader.orders.stock_recent_order(symbol)
                self.update_net_worth(self.holding[symbol], order.value)
                self.holding.pop(symbol)
                self.cached_price.pop(symbol)
                self.trader.close_tab_by_stock(symbol)
        for symbol in self.pending_cancel:
            if self._check_order_complete_status(symbol, 'Canceled'):
                self.pending_cancel.remove(symbol)
                order = self.trader.orders.stock_recent_order(symbol)
                if 'Buy' in order.method:
                    self.__paritions_used -= 1
                self.trader.close_tab_by_stock(symbol)
