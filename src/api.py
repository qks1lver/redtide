from time import sleep
from selenium.common import exceptions as selenium_exceptions
from selenium.webdriver.common.keys import Keys

from backends.signinja.utils import predefined_auth, headless_login
from src.common import float_money, regxs
from src.models import Order, Orders
from src.constants import FILE_ROBINHOOD_AUTH, GECKODRIVER_PATH, SERVICE_LOG


class HoodAPI(object):

    """
    The API for automating Robinhood trade orders
    """

    __urls = {
        'portfolio': 'https://robinhood.com',
        'stocks': 'https://robinhood.com/stocks/{}'
    }

    @classmethod
    def stock_url(cls, symbol='AMD'):
        return cls.__urls['stocks'].format(symbol.upper())

    def __init__(self):
        print('Initializing HoodAPI...')

        # Start headless driver
        if FILE_ROBINHOOD_AUTH is None:
            raise IOError('Need Robinhood auth file for login')
        auth = predefined_auth('robinhood', username='foo', password='bar')
        auth['username'][1], auth['password'][1] = open(FILE_ROBINHOOD_AUTH, 'r')\
            .read().strip().split('\n')
        try:
            _, d = headless_login(auth, exe=GECKODRIVER_PATH, log_path=SERVICE_LOG)
            self.driver = d
        except:
            print('Failed to sign into Robinhood')
            raise

        self.worth = -1
        self.buy_power = -1

        # These keeps track of stock's tab
        # do not directly reference them.
        # Access data from methods
        self.__stock_tabs = {}
        self.__tab_stocks = {}

        # Ensure the first tab is always the portfolio page
        self.to_url(self.__urls['portfolio'])
        tab_id = self.current_tab_id
        self.add_stock_tab('portfolio', tab_id)
        self.get_portfolio_values()

        # portfolio value at initialization
        self.__worth_initial = self.worth
        self.__buy_power_initial = self.buy_power

        # track orders since initializatoin
        self.orders = Orders()

        print('HoodAPI initialized.')

    @property
    def driver(self):
        return self.__driver

    @driver.setter
    def driver(self, value):
        if not hasattr(self, 'driver'):
            self.__driver = value
        else:
            print('webdriver override is forbidden')

    @property
    def tabs(self):
        return self.driver.window_handles

    @property
    def current_tab_id(self):
        try:
            return self.driver.current_window_handle
        except selenium_exceptions.NoSuchWindowException:
            print('Failed to get current tab_id, likely close. Switch to tab 0')
            try:
                self.to_tab_by_index(0)
                return self.driver.current_window_handle
            except:
                print('Still failed to get current tab_id')
                raise
        except:
            print('Failed to get current tab_id for unexpected reason')
            raise

    @property
    def net_change(self):
        self.get_portfolio_values()
        return self.worth - self.__worth_initial

    def quit(self):
        self.driver.quit()

    def get_tab_id_from_stock(self, symbol):
        if symbol in self.__stock_tabs:
            return self.__stock_tabs[symbol]
        return None

    def get_stock_from_tab_id(self, tab_id):
        if tab_id in self.__tab_stocks:
            return self.__tab_stocks[tab_id]
        return None

    def stock_has_tab(self, symbol):
        return symbol in self.__stock_tabs

    def add_stock_tab(self, symbol, tab_id):
        if symbol in self.__stock_tabs:
            print('{} already has a tab {}'.format(
                symbol, self.__stock_tabs[symbol]))
            return False

        self.__stock_tabs[symbol] = tab_id
        self.__tab_stocks[tab_id] = symbol
        return True

    def remove_stock_tab(self, symbol=None, tab_id=None):
        if symbol is None and tab_id is None:
            raise ValueError('Must pass either symbol or tab_id arg')

        if symbol is not None:
            tab_id = self.get_tab_id_from_stock(symbol)
            if tab_id is not None:
                self.__stock_tabs.pop(symbol)
                if tab_id in self.__tab_stocks:
                    self.__tab_stocks.pop(tab_id)
            else:
                print('Stock {} does not exist'.format(symbol))
        else:
            symbol = self.get_stock_from_tab_id(tab_id)
            if symbol is not None:
                self.__tab_stocks.pop(tab_id)
                if symbol in self.__stock_tabs:
                    self.__stock_tabs.pop(symbol)
            else:
                print('Tab {} does not exist'.format(tab_id))

    def new_tab(self):
        tabs0 = self.tabs[:]
        self.driver.execute_script("window.open()")
        if self.tabs != tabs0:
            tab_id = (set(self.tabs) - set(tabs0)).pop()
            self.to_tab_by_id(tab_id)
            return tab_id
        else:
            raise RuntimeError('Could not create new tab')

    def close_tab(self, tab_id=None):
        if tab_id is None:
            tab_id = self.current_tab_id
        if tab_id == self.get_tab_id_from_stock('portfolio'):
            print('Closing the first portfolio page is not allowed')
            return False

        # if manually closed portfolio tab
        if len(self.tabs) <= 1:
            print('No close: must have at least 1 tab open; otherwise .quit()')
            return False

        def _close_tab():
            tab0 = self.current_tab_id
            try:
                self.driver.execute_script("window.close()")
            except:
                print('Failed to execute window.close()')
                raise
            if tab0 not in self.tabs:
                if self.get_stock_from_tab_id(tab0) is not None:
                    self.remove_stock_tab(tab_id=tab0)
            else:
                raise RuntimeError('Could not close tab_id {}'.format(tab_id))

        if tab_id not in self.tabs:
            raise IndexError('tab_id {} not in {}'.format(tab_id, self.tabs))

        if self.current_tab_id != tab_id:
            return_tab_id = self.current_tab_id
            self.to_tab_by_id(tab_id)
        else:
            return_tab_id = None
            for i in self.tabs:
                if i != tab_id:
                    return_tab_id = self.tabs[0]
                    break
            if return_tab_id is None:
                msg = 'Could not find a return_tab_id from {}' \
                      ' that not {}'.format(self.tabs, tab_id)
                raise IndexError(msg)
        _close_tab()
        self.to_tab_by_id(return_tab_id)

        print('Tab_id {} closed'.format(tab_id))
        return True

    def close_tab_by_stock(self, symbol):
        tab_id = self.get_tab_id_from_stock(symbol)
        if tab_id is not None:
            return self.close_tab(tab_id)
        else:
            print('No stock tab to close for stock', symbol)
            return False

    def to_tab_by_id(self, tab_id):
        try:
            self.driver.switch_to_window(tab_id)
        except:
            print('Failed to switch to tab_id {}'.format(tab_id))
            raise

    def to_tab_by_index(self, idx):
        try:
            self.to_tab_by_id(self.tabs[idx])
        except:
            print('Failed to switch to tab {}'.format(idx))
            raise

    def to_url(self, url):
        tab_id = self.current_tab_id
        if self.get_stock_from_tab_id(tab_id) is not None:
            print('Going to another url from a stock tab is not allowed')
            return False
        else:
            self.driver.get(url)
            sleep(4)
            return True

    def to_portfolio_tab(self):
        try:
            self.to_tab_by_id(self.get_tab_id_from_stock('portfolio'))
            return True
        except Exception as e:
            print('Failed to switch to portfolio tab: {}'.format(e))
            return False

    def new_tab_url(self, url=''):
        if not url:
            url = self.__urls['portfolio']

        new_tab_id = None
        try:
            new_tab_id = self.new_tab()
            self.to_url(url)
        except:
            print('Failed to open url in new tab: {}'.format(url))
            if new_tab_id is not None:
                try:
                    print('Try closing new tab')
                    self.close_tab(tab_id=new_tab_id)
                except Exception as e:
                    print('Could not close new tab because: {}'.format(e))
            raise
        return new_tab_id

    def new_tab_stock(self, symbol):
        if not symbol:
            raise ValueError('Need symbol')

        if not self.stock_has_tab(symbol):
            tab_id = self.new_tab_url(self.stock_url(symbol))
            self.add_stock_tab(symbol, tab_id)
            sleep(1)
        else:
            tab_id = self.get_tab_id_from_stock(symbol)

        # navigate to the stock tab
        self.to_tab_by_id(tab_id)

        return tab_id

    def get_portfolio_values(self):
        return_tab_id = None
        try:
            portfolio_id = self.get_tab_id_from_stock('portfolio')
            if portfolio_id != self.current_tab_id:
                return_tab_id = self.current_tab_id
                self.to_tab_by_id(self.get_tab_id_from_stock('portfolio'))

            # Worth
            res = self.driver.find_element_by_xpath(
                '//main[@class="main-container"]//header').text
            self.worth = float_money(res.split('\n')[0])
            print('Worth:', self.worth)

            # Buying power
            res = self.driver.find_element_by_xpath(
                '//div[@class="sidebar-content"]//button').text
            self.buy_power = float_money(res.split('\n')[-1])
            print('Buying power:', self.buy_power)
        except:
            print('Failed to get protfolio values')
            raise
        finally:
            if return_tab_id is not None:
                self.to_tab_by_id(return_tab_id)

    def get_portfolio_stocks(self):
        self.to_portfolio_tab()
        portfolio_stocks = {}
        if 'class="sidebar-content"' in self.driver.page_source:
            text = self.driver.find_element_by_xpath('//div[@class="sidebar-content"]').text
            if text:
                res = regxs['rh_port_stocks'].findall(text)
                if res:
                    for symb, shares in res:
                        portfolio_stocks[symb] = shares
        return portfolio_stocks

    def _order_method(self, action='buy'):
        if action not in ['buy', 'sell']:
            raise ValueError('Invalid order method, action={}'.format(action))

        if self._available_shares() > 0:
            if action == 'buy':
                self.driver.find_elements_by_xpath('//div[@role="button"]')[0].click()
            else:
                self.driver.find_elements_by_xpath('//div[@role="button"]')[1].click()
        elif action == 'sell':
            return False
        return True

    def _send_shares(self, n):
        self.driver.find_element_by_name("quantity").send_keys(str(n))

    def _review_order(self):
        self.driver.find_element_by_xpath(
            '//button[@data-testid="OrderFormControls-Review"]').click()

    def _edit_order(self):
        # for when in review, to go back to edit
        self.driver.find_element_by_xpath('//button[contains(.,"Edit")]').click()

    def _change_order_type(self, order_type='market'):
        if order_type not in ['market', 'limit', 'stopLoss', 'stopLimit', 'trailingStop']:
            raise ValueError('Invalid order type, order_type={}'.format(order_type))

        def _select(v):
            self.driver.find_element_by_xpath(
                '//span[contains(., "{} Order")]/parent::span'
                '/parent::div/parent::div'.format(v)).click()

        # click the drop-down
        self.driver.find_elements_by_xpath(
            '//form[@data-testid="OrderForm"]/div[1]/div[1]/div')[-1].click()

        if order_type == 'market':
            _select('Market')
        elif order_type == 'limit':
            _select('Limit')
        elif order_type == 'stopLoss':
            _select('Stop Loss')
        elif order_type == 'stopLimit':
            _select('Stop Limit')
        elif order_type == 'trailingStop':
            _select('Trailing Stop')
        else:
            self.driver.find_element_by_tag_name('body').send_keys(Keys.ESCAPE)
            raise ValueError('No such order type: {}'.format(order_type))

    def _set_limit_price(self, price):
        self.driver.find_element_by_xpath('//input[@name="limitPrice"]').send_keys(str(price))

    def _current_market_price(self):
        self._change_order_type('market')
        res = self.driver.find_element_by_xpath(
            '//span[contains(., "Market Price")]/parent::a/parent::div/parent::div').text
        return float_money(res.split('\n')[-1])

    def _estimate_cost(self):
        res = self.driver.find_element_by_xpath(
            '//span[contains(., "Estimated Cost")]/parent::div/parent::div').text
        return float_money(res.split('\n')[-1])

    def _make_order(self):
        self.driver.find_element_by_xpath('//button[@data-testid="OrderFormControls-Submit"]').click()

    def _done_after_order(self):
        self.driver.find_element_by_xpath('//button[@data-testid="OrderFormDone"]').click()

    def _order_status(self):
        """
        pending buy order: {'type': 'Limit Buy', 'date': 'May 21, 2020', 'status': 'Pending'}
        completed buy (after placed): {'type': 'Limit Buy', 'date': 'May 21, 2020', 'status': 'Placed'}
        completed buy (after purchase): {'type': 'Limit Buy', 'date': '15m', 'cost': 1767.0, 'shares': 5, 'price': 353.4, 'status': 'Done'}
        :return:
        """
        res = self.driver.find_element_by_xpath('//header[@data-testid="rh-ExpandableItem-buttonContent"]').text
        tmp = res.split('\n')
        if len(tmp) == 3:
            status = {
                'type': tmp[0],
                'time': tmp[1],     # date when queued but elapse time since cancel when caceled
                'status': tmp[2]
            }
        elif len(tmp) == 4:
            tmp2 = tmp[-1].split()
            status = {
                'type': tmp[0],
                'time': tmp[1],
                'value': float_money(tmp[2]),
                'shares': int(tmp2[0]),
                'price': float_money(tmp2[-1]),
                'status': 'Done'
            }
        else:
            status = {
                'status': None,
                'raw': res
            }
        return status

    def _cancel_order(self):
        self.driver.find_element_by_xpath('//button[@data-testid="rh-ExpandableItem-button"]').click()
        i_try = 0
        while i_try < 10:
            try:
                self.driver.find_element_by_xpath('//a[text()[contains(., "Cancel Order")]]').click()
                break
            except Exception as e:
                if i_try < 9:
                    i_try += 1
                    sleep(1)
                else:
                    raise e

    def _available_shares(self):
        if 'class="grid-2"' in self.driver.page_source:
            text = self.driver.find_element_by_xpath('//div[@class="grid-2"]').text
            if text:
                res = regxs['rh_shares'].findall(text)
                if res and res[0]:
                    if len(res) > 1:
                        print('! _available_shares: more than 1 regex match: {}'.format(res))
                    try:
                        return int(res[0].replace(',', ''))
                    except Exception as e:
                        print('! failed to parse shares: {}'.format(e))
        return 0

    def _can_trade(self):
        if 'Page not found' not in self.driver.page_source and 'not supported' not in self.driver.page_source:
            return True
        else:
            return False

    def make_order(self, method, symbol, shares=None, order_type='market', price=None):
        if method not in ['buy', 'sell']:
            raise ValueError('Order method can only be buy or sell, not: {}'.format(method))

        if method == 'buy' and shares is None:
            raise ValueError('Need shares to make buy order')

        if order_type not in ['market', 'limit']:
            raise ValueError('"{}" is not a valid order type'.format(order_type))

        if order_type == 'limit' and price is None:
            raise ValueError('Need price for Limit Order')

        if shares is not None and not isinstance(shares, int):
            raise ValueError('shares must be an integer')

        if price is not None and not isinstance(price, float) and not isinstance(price, int):
            raise ValueError('price must be int or float')

        # 1. navigate to or make new tab for stock
        self.new_tab_stock(symbol)

        if method == 'buy':
            # check if tradable
            if not self._can_trade():
                print('Cannot trade this stock:', symbol)
                return False

            # 2. set to Buy
            self._order_method('buy')

            # 3. check portfolio buy power to make sure this purchase is possible
            self.get_portfolio_values()
            if order_type == 'limit':
                cost = shares * price
            else:
                stock_price = self._current_market_price()
                cost = shares * stock_price
                print('{} is trading at {}, {} shares is {}'.format(
                    symbol, stock_price, shares, cost))
            if cost >= self.buy_power:
                print('Not enough buying power ({}) for this'
                      ' order, cost: {}'.format(self.buy_power, cost))
                return False
        else:
            # 2. set to Sell
            self._order_method('sell')

            # 3. check if there are enough shares
            shares_avail = self._available_shares()
            if shares_avail == 0:
                print('No shares of {} to sell'.format(symbol))
                return False

            if shares is None:
                shares = shares_avail
            elif shares > shares_avail:
                print('Not enough shares of {}, own {}, trying to'
                      ' sell {}'.format(symbol, shares_avail, shares))
                return False

        # 4. set order type
        self._change_order_type(order_type)

        # 5. enter shares
        self._send_shares(shares)

        # 6. if not "Market Order", add addition requirements
        if order_type == 'limit':
            self._set_limit_price(price)

        # 7. go to Review
        if price is None:
            price = self._current_market_price()
        self._review_order()

        # 8. buy
        self._make_order()
        self._done_after_order()

        # add to order history
        order = Order(symbol, method, shares, price)
        self.orders.append(order)
        return True

    def cancel_order(self, symbol):
        if not self.orders.has_stock(symbol):
            print('No orders for', symbol)
            return False

        # either to go stock tab or create new
        self.new_tab_stock(symbol)

        recent_order = self.orders.stock_recent_order(symbol)
        if recent_order.order_status is None:
            i_try = 0
            while recent_order.order_status is None and i_try < 10:
                sleep(1)
                recent_order.status = self._order_status()
                i_try += 1
            if recent_order.order_status is None:
                print('Cannot cancel ({}): could not get status of most recent order'.format(symbol))
                return False

        if recent_order.order_status == 'Done':
            print('Cannot cancel ({}): order already went through'.format(symbol))
            return False
        elif recent_order.order_status == 'Canceled':
            print('Recent order for {} is already canceled'.format(symbol))
            return False

        try:
            self._cancel_order()
        except Exception as e:
            print('Cannot cancel ({}): cancel failed:\n{}'.format(symbol, e))
            return False
        return True

    def order_status(self, symbol):
        if not self.orders.has_stock(symbol):
            print('No orders for', symbol)
            return None

        # either to go stock tab or create new
        self.new_tab_stock(symbol)

        return self._order_status()
