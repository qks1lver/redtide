#!/usr/bin/env python3

# Import
import requests
import re
import os
import json
import datetime
import pandas as pd
import numpy as np
import pickle
import subprocess
from time import sleep, time
from lxml import html
from src.common import gen_rand_name, gen_symbol_batches, get_wallstreet_time
from backends.signinja.utils import headless_login, predefined_auth
from multiprocessing import Pool
from platform import mac_ver
from src.constants import (DIR_DATA, FILES_LISTINGS, FILE_ALL_SYMBOLS,
                           FILE_EXCLUDED_SYMBOLS, URL_YAHOO, URL_YAHOO_DAILY,
                           URL_ALPHA_VANTAGE_INTRADAY, ALPHA_VANTAGE_API_KEY,
                           FILE_EODDATA_AUTH, URL_EODDATA_GET_SYMBOLS,
                           EODDATA_EXCHANGES, URL_YAHOO_FINANCIALS, DIR_FINANCIALS,
                           URL_YAHOO_OPTIONS, DIR_OPTIONS, GECKODRIVER_PATH, SERVICE_LOG)


# To by-pass Mac's new security things that causes multiprocessing to crash
v = None
try:
    v = mac_ver()
except:
    print('Did not detect MAC')

if v and v[0] and int(v[0].split('.')[0]) >= 10:
    print('Detected Mac > High Sierra, deploy multiprocessing fix')
    try:
        _ = subprocess.Popen('export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES', shell=True)
    except:
        print('\tFailed to send fix: export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES')


# Classes
# consider segmenting this into separate classes for getting stock data and processing for ML
class Stock:
    """
    A class for scraping stock data

    mode is either full_history, live_quotes, or intraday

    """
    def __init__(self, mode='full_history', verbose=False):

        # os.environ['TZ'] = 'America/New_York'
        # tzset()

        self._dt_wallstreet_ = get_wallstreet_time()['datetime'] - datetime.datetime.now()
        self._dir_out_ = os.path.join(DIR_DATA, mode)
        self._date_format_ = '%Y-%m-%d'
        self._date_time_format_ = '%Y-%m-%d-%H-%M-%S'
        self._max_connection_attempts_ = 20

        self.rexp_live_dollar = re.compile(r'starQuote.*?<')
        self.rexp_live_volume = re.compile(r'olume<.+?</td>')
        self.rexp_yahoo_prices_list = re.compile(r'"prices":\[.*?\]')

        self.n_cpu = os.cpu_count()
        self.verbose = verbose
        self.symbs = list()
        self.live_now = False
        self.dfs = None
        self.alpha_vantage_state = 'ok'

        if mode and not os.path.isdir(self._dir_out_):
            os.makedirs(self._dir_out_)
            print('Created: %s' % self._dir_out_)

    def pull_daily(self, symb, period1=0, period2=0):
        """
        Grabs historical data for symbol from dates period1 to period2
        :param symb: symbol to grab
        :param period1: starting time period in seconds
        :param period2: ending time period in seconds
        :return: data: nested list containing pricing data in following format:
                        [... [date, volume, open, close, high, low, adjusted close]...]
                note - consider adjusting this to standard OHLC order
        """

        # grab prices up to the current day if period2 not set
        if not period2:
            period2 = datetime.datetime.now().timestamp()

        # url to grab prices
        url = URL_YAHOO_DAILY % (symb, period1, period2)

        # try requesting for daily history
        r, msg = self._try_request(url)

        # if request failed
        if r is None or r.status_code != requests.codes.ok:
            if self.verbose:
                print('Page load failed for {}: {}'.format(symb, msg))
            return -1

        # this parse the list of prices under "HistoricalPriceStore"
        res = self.rexp_yahoo_prices_list.search(r.text)
        if res:
            price_data = json.loads('{%s}' % re.search(r'"prices":\[.*?\]', r.text).group(0))['prices']
        else:
            print('Cannot find data for %s' % symb)
            return -1

        data = {
            'date': [],
            'volume': [],
            'open': [],
            'high': [],
            'low': [],
            'close': [],
            'adjclose': [],
        }
        for d in price_data:
            # TODO - rows without volume (or any of these metrics)
            # are likely Splits. Need to store this data somehow
            # currently, ignoring them
            if d.get('volume', None):
                timestamp = datetime.datetime.fromtimestamp(int(d['date'])) \
                            + self._dt_wallstreet_
                data['date'].append(timestamp.strftime(self._date_format_))
                data['volume'].append(d.get('volume', None))
                data['open'].append(d.get('open', None))
                data['high'].append(d.get('high', None))
                data['low'].append(d.get('low', None))
                data['close'].append(d.get('close', None))
                data['adjclose'].append(d.get('adjclose', None))

        if self.verbose:
            print('Pulled %d days of %s' % (len(data['date']), symb))

        return data

    def write_data(self, data, symb, dir_out='', columns=None, fmt='csv'):
        """
        Writes pricing data out to csv file as dir_out/symb.csv
        :param data: list, pricing data
        :param symb: string, stock symbol for pricing data
        :param dir_out: string, the directory to write to
        :param columns: list or None, the order of columns
        :param fmt: string, output format, default csv
        :return: p_out, string of 'dir_out/symb.csv'
        """
        if not dir_out:
            dir_out = self._dir_out_
        p_out = os.path.join(dir_out, '%s.%s' % (symb, fmt))

        if not columns:
            columns = list(data.keys())
            # columns = 'date volume open close high low adjclose'.split()

        with open(p_out, 'w+') as f:
            _ = f.write(','.join(columns) + '\n')
            for i in range(len(data[columns[0]])):
                out_str = ','.join(['{}'.format(data[c][i]) for c in columns]) + '\n'
                _ = f.write(out_str)

        if self.verbose:
            print('Wrote %s data to %s' % (symb, p_out))

        return p_out

    def get_all_symbols(self, try_compiled=True):
        """
        Returns a sorted list of all symbols from the all_symbols.txt files if present,
        or from a combination of all symbols in the listing files in DIR_LISTINGS
        :param try_compiled, Bool, whether to attempt getting symbols from all_symbols.txt
        :return: sorted_symbs, a sorted list of stock symbols
        """

        symbs = []

        if try_compiled and os.path.isfile(FILE_ALL_SYMBOLS):

            if self.verbose:
                print('Using %s for symbols ...' % FILE_ALL_SYMBOLS)

            with open(FILE_ALL_SYMBOLS, 'r') as f:
                symbs = list(set(f.read().strip().split('\n')))

        elif FILES_LISTINGS:

            # Remove subgroup extensions of tickers,
            # i.e. ABCD-A -> ABCD
            # Some stocks do need it, but this is not
            # compatible with Yahoo Finance at the moment
            # So currently, just removed, and hope for
            # the best. Need improvement!
            rexp = re.compile(r'-[a-zA-Z]$')

            for ex_name, f_listing in FILES_LISTINGS.items():
                symbs += [rexp.sub('',s) for s in
                          pd.read_table(f_listing)['Symbol'].values]

            # remove redundant symbols
            symbs = list(set(symbs))

        elif FILE_EODDATA_AUTH and EODDATA_EXCHANGES:
            if self.verbose:
                print('Attempting to get symbols from eoddata.com ...')
            auth = predefined_auth('eoddata', username='foo', password='bar')
            auth['username'][1], auth['password'][1] = open(FILE_EODDATA_AUTH, 'r')\
                .read().strip().split('\n')
            if self.verbose:
                print('Try signing into eoddata.com ...')
            session = headless_login(auth, exe=GECKODRIVER_PATH, log_path=SERVICE_LOG)
            if self.verbose:
                print('Signed into eoddata.com')

            def _verify_response(parsed_symbols, maxlen=10, cutoff=0.95):
                # if parsed_symbols do actually contains symbols
                # almost all of the values should be less than 10 chars
                ratio = len([1 for s in parsed_symbols if len(s) <= maxlen]) \
                        / len(parsed_symbols)
                return ratio >= cutoff

            for exchange in EODDATA_EXCHANGES:
                r = session.get(URL_EODDATA_GET_SYMBOLS % exchange)
                s = [i.split('\t')[0] for i in r.text.strip().split('\n')][1:]
                if _verify_response(s):
                    symbs += s
                    if self.verbose:
                        print('Got %d symbols for %s from eoddata.com'
                              % (len(s), exchange))

        elif self.verbose:
            print('Missing symbol file.')

        if symbs and self.verbose:
            print('\tFound %d symbols' % len(symbs))

        sorted_symbs = sorted(symbs)
        return sorted_symbs

    def pull_daily_and_write_batch(self, symbs=None, p_symbs='', i_pass=1, max_pass=2):
        """
        Grabs pricing history for all stock symbols in symbs and writes out data to csv's
        by making n = len(symbs) calls of retrieve_symb(symbol) for each symbol in symbs.
        Uses multiple CPUs if available.
        :param symbs: list of stock symbols to retrieve from yahoo
        :param p_symbs: file containing list of symbols for symbs
        :param i_pass: current attempt at grabbing failed symbols
        :param max_pass: max attempts to grab failed symbols
        :return nothing, as we'll call retrieve_symb to write our csv files
        """

        t0 = time()

        if p_symbs:
            if os.path.isfile(p_symbs):
                with open(p_symbs, 'r') as f:
                    symbs = f.read().strip().split('\n')
            else:
                print('No such file: %s' % p_symbs)

        elif symbs is None:
            symbs = self.get_all_symbols()
            updated_symbs = self.get_updated_symbs()
            if updated_symbs:
                symbs = list(set(symbs) - set(updated_symbs))
                if self.verbose:
                    print('Skip pull for %d symbols, only pulling for %d symbols ...' % (len(updated_symbs), len(symbs)))

        n_symbs = len(symbs)
        n_success = 0
        failed_symbs = []
        print('Pulling for %d symbols ...' % n_symbs)

        if self.n_cpu > 1:

            print('\tUsing %d CPUs' % self.n_cpu)

            # To avoid getting blocked by Yahoo, pause for 10 - 20 seconds after 100 symbols
            symb_batches = gen_symbol_batches(symbs, batch_size=100)
            n_symb_completed = 0

            # for each batch of symbols have a worker thread execute self.retrieve_symb(batch)
            # which then calls write_history to write our pricing csv
            for batch in symb_batches:
                with Pool(processes=self.n_cpu) as pool:
                    res = pool.map(self.pull_daily_and_write, batch)

                for symb,success in res:
                    if success:
                        n_success += 1
                    else:
                        failed_symbs.append(symb)

                n_symb_completed += len(batch)
                if self.verbose:
                    print('{0:.1f}% completed - {1:.0f} / {2:.0f}'.format(n_symb_completed / n_symbs * 100, n_symb_completed, n_symbs))

                # pause to avoid Yahoo block
                if n_symb_completed != n_symbs:
                    tpause = np.random.randint(10, 21)
                    print('Pause for %d seconds' % tpause)
                    sleep(tpause)

        else:

            for i, symb in enumerate(symbs):

                if self.verbose:
                    print('Pulling %d / %d - %s ...' % (i+1, n_symbs, symb))

                data = self.pull_daily(symb)
                if data != -1:
                    _ = self.write_data(data, symb)
                    n_success += 1
                else:
                    failed_symbs.append(symb)

        if self.verbose:
            print('\nRetrieved full histories for %d / %d symbols' % (n_success, n_symbs))

        # does this suggest failed_symbs should be it's own keyword arg?
        if failed_symbs:
            print('Failed for:')
            for symb in failed_symbs:
                print('\t%s' % symb)

            if i_pass < max_pass:
                i_pass += 1
                print('\n|--- Pass %d (try to fetch %d failed ones, maximum %d passes ---|' % (i_pass, len(failed_symbs), max_pass))
                self.pull_daily_and_write_batch(symbs=failed_symbs, p_symbs=p_symbs, i_pass=i_pass, max_pass=max_pass)
            else:
                p_symbs_fail = 'failed_symbs-%s.txt' % (''.join(np.random.choice(list('abcdefgh12345678'), 5)))
                with open(p_symbs_fail, 'w+') as f:
                    _ = f.write('\n'.join(failed_symbs))
                print('Failed symbols written to: %s' % p_symbs_fail)
                print('Run this to try fetching the missed symbols again:\npython3 redtide.py -v -d --file %s' % p_symbs_fail)

        print('\tTime elapsed: %.2f hours\n' % ((time() - t0)/3600))

        return

    def get_updated_symbs(self):
        """ Returns a list of updated symbols.
        This is determined based on the last modified timestamp of the symbol
        file. If timestamp is greater than market close time, then it is most
        likely updated.
        EOD is 4 PM NY time
        """

        t = datetime.datetime.today()
        t_close = (datetime.datetime(year=t.year, month=t.month, day=t.day, hour=16)
                   - self._dt_wallstreet_).timestamp()

        symbs = []
        for p in os.listdir(self._dir_out_):
            t = os.path.getmtime(os.path.join(self._dir_out_, p))
            if t > t_close:
                symbs.append(p.replace('.csv', ''))

        if self.verbose:
            print('Full history up-to-date for %d symbols' % len(symbs))

        return symbs

    def pull_daily_and_write(self, symb):
        """ calls pull_history to grab price data on symb and writes result
        to csv using write_history
        """

        success = False

        if self.verbose:
            print('Pulling %s ...' % symb)

        data = self.pull_daily(symb)
        if data != -1:
            _ = self.write_data(data, symb)
            success = True

        return symb, success

    def get_full_histories_from_file(self, symbs=None):
        """ reads price history from csv files into Pandas Dataframes stored in self.dfs """

        self.dfs = {}

        if not symbs:

            symbs = self.get_all_symbols()

        if self.verbose:
            print('Reading %d full histories to dataframes ...' % len(symbs))

        for symb in symbs:

            p_data = os.path.join(self._dir_out_, '%s.csv' % symb)

            if not os.path.isfile(p_data):
                if self.verbose:
                    print('No full history available: %s' % symb)
                continue

            df = pd.read_csv(p_data, index_col='date', parse_dates=True)
            self.dfs[symb] = df

        if self.verbose:
            print('Read %d / %d full histories to dataframes' % (len(self.dfs), len(symbs)))

        return self

    def save_financial_data(self, symb, as_json=False):
        r, _ = self._try_request(URL_YAHOO_FINANCIALS % symb)
        if r is None or r.status_code != requests.codes.ok \
                or 'Symbol Lookup' in r.text[:300]:
            print('%s does not have financials' % symb)
            return

        try:
            root = html.fromstring(r.text)
            node = root.xpath("//script[contains(text(), '{\"context')]")[0]
            i0 = node.text.find('{"context')
            i1 = node.text.rfind('};') + 1
            json_data = json.loads(node.text[i0:i1])['context']['dispatcher']['stores']['QuoteSummaryStore']
            if as_json:
                json.dump(
                    json_data,
                    open(os.path.join(
                        DIR_FINANCIALS, symb + '.json'), 'w+')
                )
            else:
                with open(os.path.join(
                        DIR_FINANCIALS, symb + '.pkl'), 'wb') as f:
                    pickle.dump(json_data, f)
        except:
            if self.verbose:
                print('Could not parse financials for {}' % symb)
            raise

        return

    def save_options_data(self, symb, as_json=False):
        r, _ = self._try_request(URL_YAHOO_OPTIONS % symb)
        if r is None or r.status_code != requests.codes.ok \
                or 'Symbol Lookup' in r.text[:300]:
            print('%s does not have options' % symb)
            return

        try:
            root = html.fromstring(r.text)
            node = root.xpath("//script[contains(text(), '{\"context')]")[0]
            i0 = node.text.find('{"context')
            i1 = node.text.rfind('};') + 1
            json_data = json.loads(node.text[i0:i1])['context']['dispatcher']['stores']['OptionContractsStore']
            if as_json:
                json.dump(
                    json_data,
                    open(os.path.join(
                        DIR_OPTIONS, symb + '.json'), 'w+')
                )
            else:
                with open(os.path.join(
                        DIR_OPTIONS, symb + '.pkl'), 'wb') as f:
                    pickle.dump(json_data, f)
        except:
            if self.verbose:
                print('Could not parse options for {}' % symb)
            raise

        return

    def compile_symbols(self, p_symbs=None, append=False, batch_size=-1):

        print('Compiling symbols ...')

        if p_symbs:
            with open(p_symbs, 'r') as f:
                self.symbs = f.read().strip().split('\n')
        else:
            self.symbs = self.get_all_symbols(try_compiled=False)
        n_symbs = len(self.symbs)
        n_compiled = 0
        n_excluded = 0

        # Initialize/clear write files
        if not append:
            with open(FILE_ALL_SYMBOLS, 'w+') as f, open(FILE_EXCLUDED_SYMBOLS, 'w+') as fx:
                _ = f.write('')
                _ = fx.write('')

        if self.verbose:
            print('Looking up symbols on Yahoo Finance ...')

        if batch_size == -1:
            batch_size = np.random.randint(30, 100)

        symb_batches = gen_symbol_batches(self.symbs, batch_size=batch_size)
        n_symb_completed = 0
        for batch in symb_batches:
            with Pool(processes=self.n_cpu, maxtasksperchild=1) as pool:
                res = pool.map(self._compile_symb, batch)

            with open(FILE_ALL_SYMBOLS, 'a+') as f, open(FILE_EXCLUDED_SYMBOLS, 'a+') as fx:
                for symb, success in res:
                    if success:
                        _ = f.write('%s\n' % symb)
                        n_compiled += 1
                    else:
                        _ = fx.write('%s\n' % symb)
                        n_excluded += 1

            n_symb_completed += len(batch)
            if self.verbose:
                print('{0:.1f}% completed - {1:.0f} / {2:.0f}'.format(n_symb_completed / n_symbs * 100, n_symb_completed, n_symbs))

            tpause = np.random.randint(5, 11)
            print('Pause for %s seconds' % tpause)
            sleep(tpause)

        # Remove duplicates when using append
        if append:
            with open(FILE_ALL_SYMBOLS, 'r') as f:
                symbs = set(f.read().strip().split('\n'))
            with open(FILE_ALL_SYMBOLS, 'w+') as f:
                _ = f.write('\n'.join(symbs))

        print('Started with: %d\nCompiled: %d\nExcluded: %d\nCompiled to: %s' % (n_symbs, n_compiled, n_excluded, FILE_ALL_SYMBOLS))

        return

    def _try_request(self, url):

        r = None
        n_tries = 0
        msg = ''
        while n_tries < self._max_connection_attempts_:
            try:
                r = requests.get(url)
                if r.status_code == 200:
                    break
                else:
                    n_tries += 1
                    if n_tries >= self._max_connection_attempts_:
                        msg = 'Try exceeded, response code {}'.format(r.status_code)
            except(KeyboardInterrupt, SystemExit):
                raise
            except Exception as e:
                n_tries += 1
                if n_tries >= self._max_connection_attempts_:
                    msg = e

        return r, msg

    def _check_symbol_(self, symb, parse_financials=True, parse_options=False):

        """
        Check whether a symbol exists on Yahoo Finance, where historical data will be retrieved

        :param symb: String
        :param parse_financials: get all financial data on the company
        :return: 1 - found, 0 - not found, -1 - connection error, -x - request message code x
        """

        msg = 0

        # Try original symbol string
        url = URL_YAHOO % symb
        r, _ = self._try_request(url)
        if r is None:
            msg = -1
        else:
            if r.status_code != requests.codes.ok:
                msg = -r.status_code
            elif 'Symbol Lookup' not in r.text[:300]:
                msg = 1

        # Try replace . with - if do-able
        if msg != 1 and '.' in symb:
            url = URL_YAHOO % symb.replace('.', '-')
            r, _ = self._try_request(url)
            if r is None:
                msg = -1
            else:
                if r.status_code != requests.codes.ok:
                    msg = -r.status_code
                elif 'Symbol not found' not in r.text[:250]:
                    msg = 1
                    symb = symb.replace('.', '-')

        # parse financial data
        if parse_financials and msg == 1 and 'financials?' in r.text:
            self.save_financial_data(symb)

        # parse options data
        if parse_options and msg == 1 and 'options?' in r.text:
            self.save_options_data(symb)

        return msg, symb

    def _compile_symb(self, symb):

        msg, symb = self._check_symbol_(symb)
        if msg < 1:

            if self.verbose:
                if msg == 0:
                    print('%s excluded - not found' % symb)
                elif msg == -1:
                    print('%s excluded - max connection attempt reached' % symb)
                else:
                    print('%s excluded - request code: %d' % (symb, -msg))

            return symb, False

        return symb, True

    def concat(self, from_date=None, to_date=None, p_out=None, return_df=False):

        if not to_date:
            to_date = datetime.datetime.now().strftime('%Y-%m-%d')

        if not from_date:
            # Default 60 days
            from_date = (datetime.datetime.strptime(to_date, '%Y-%m-%d') - datetime.timedelta(days=60)).strftime('%Y-%m-%d')

        if not p_out:
            p_out = os.path.join(DIR_DATA, gen_rand_name())

        symbs = self.get_all_symbols()
        n_symbs = len(symbs)

        print('Concatenating %d symbols between %s - %s ...' % (n_symbs, from_date, to_date))

        new_file = True
        for i, symb in enumerate(symbs):

            p_data = os.path.join(self._dir_out_, '%s.csv' % symb)

            if not os.path.isfile(p_data):
                if self.verbose:
                    print('No full history available: %s' % symb)
                continue

            dtmp = pd.read_csv(p_data, index_col='date', parse_dates=True)[to_date:from_date].assign(symbol=symb)
            if new_file:
                dtmp.to_csv(p_out, mode='w+', header=True)
                new_file = False
            else:
                dtmp.to_csv(p_out, mode='a', header=False)

            if (i + 1) % 400 == 0:
                print('{0:.1f}% completed'.format((i+1)/n_symbs*100))

        print('Concatenated to', p_out)

        if return_df:
            print('Building dataframe...')
            df = pd.read_csv(p_out, index_col='date', parse_dates=True)
            return df
        else:
            return None

    def pull_intraday(self, symb, interval='1min'):

        api_params = {
            'symbol': symb,
            'interval': interval,
            'apikey': ALPHA_VANTAGE_API_KEY,
        }
        data = None

        url = URL_ALPHA_VANTAGE_INTRADAY.format(**api_params)
        try:
            r = requests.get(url)
            if r.status_code == 200:
                time_series = r.json().get('Time Series ({})'.format(interval), None)
                data = {
                    'datetime': [],
                    'open': [],
                    'high': [],
                    'low': [],
                    'close': [],
                    'volume': [],
                }
                if time_series:
                    def to_numeric(val, dtype=float):
                        if val is None:
                            return None
                        try:
                            return dtype(val)
                        except:
                            if self.verbose:
                                print('Unknown Alpha Vantage value: {}'.format(val))
                            return val

                    for timestamp, values in time_series.items():
                        data['datetime'].append(timestamp)
                        data['open'].append(to_numeric(values.get('1. open', None)))
                        data['high'].append(to_numeric(values.get('2. high', None)))
                        data['low'].append(to_numeric(values.get('3. low', None)))
                        data['close'].append(to_numeric(values.get('4. close', None)))
                        data['volume'].append(to_numeric(values.get('5. volume', None), dtype=int))
            # else:
                # with self.lock:
                #     self.alpha_vantage_state = 'max_per_min_reached'
                    # TODO - complete max limit exceed, break-out logic
                print('Response from Alpha Vantage is not ok, response {}'.format(r.status_code))
        except Exception as e:
            print('Failed to pull intraday for {}: {}'.format(symb, e))

        return data

    def pull_intraday_batch_and_write(self, symbs=None, p_symbs='', i_pass=1, max_pass=5, interval='1min'):
        """
        Using Alpha Vantage's API to get intraday data for all symbols in symbs.
        :param symbs:
        :param p_symbs:
        :param i_pass:
        :param max_pass:
        :param interval:
        :return:
        """

        t0 = time()

        if p_symbs:
            if os.path.isfile(p_symbs):
                with open(p_symbs, 'r') as f:
                    symbs = f.read().strip().split('\n')
            else:
                print('No such file: %s' % p_symbs)

        elif symbs is None:
            symbs = self.get_all_symbols()
            updated_symbs = self.get_updated_symbs()
            if updated_symbs:
                symbs = list(set(symbs) - set(updated_symbs))
                if self.verbose:
                    print(
                        'Skip pull for %d symbols, only pulling for %d symbols ...' % (len(updated_symbs), len(symbs)))

        n_symbs = len(symbs)
        n_success = 0
        failed_symbs = []
        print('Pulling for %d symbols ...' % n_symbs)

        if self.n_cpu > 1:

            print('\tUsing %d CPUs' % self.n_cpu)

            symb_batches = gen_symbol_batches(symbs, batch_size=5)
            n_symb_completed = 0

            # for each batch of symbols have a worker thread execute self.retrieve_symb(batch)
            # which then calls write_history to write our pricing csv
            for batch in symb_batches:
                with Pool(processes=self.n_cpu) as pool:
                    res = pool.map(self.pull_daily_and_write, batch)

                for symb, success in res:
                    if success:
                        n_success += 1
                    else:
                        failed_symbs.append(symb)

                n_symb_completed += len(batch)
                if self.verbose:
                    print('{0:.1f}% completed - {1:.0f} / {2:.0f}'.format(n_symb_completed / n_symbs * 100,
                                                                          n_symb_completed, n_symbs))

        else:

            for i, symb in enumerate(symbs):

                if self.verbose:
                    print('Pulling %d / %d - %s ...' % (i + 1, n_symbs, symb))

                data = self.pull_daily(symb)
                if data != -1:
                    _ = self.write_data(data, symb)
                    n_success += 1
                else:
                    failed_symbs.append(symb)

        if self.verbose:
            print('\nRetrieved intraday for %d / %d symbols' % (n_success, n_symbs))

        # does this suggest failed_symbs should be it's own keyword arg?
        if failed_symbs:
            print('Failed for:')
            for symb in failed_symbs:
                print('\t%s' % symb)

            if i_pass < max_pass:
                i_pass += 1
                print('\n|--- Pass %d (try to fetch %d failed ones, maximum %d passes ---|' % (
                i_pass, len(failed_symbs), max_pass))
                self.pull_intraday_batch_and_write(symbs=failed_symbs, p_symbs=p_symbs, i_pass=i_pass, max_pass=max_pass, interval=interval)
            else:
                p_symbs_failed = 'failed_symbs-%s.txt' % (''.join(np.random.choice(list('abcdefgh12345678'), 5)))
                with open(p_symbs_failed, 'w+') as f:
                    _ = f.write('\n'.join(failed_symbs))
                print('Failed symbols written to: %s' % p_symbs_failed)
                print('Run this to try fetching the missed symbols again:\npython3 redtide.py -v -d --file %s' % p_symbs_failed)

        print('\tTime elapsed: %.2f hours\n' % ((time() - t0) / 3600))

        return
