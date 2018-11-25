#!/usr/bin/env python3

# Import
import requests
import re
import datetime
import os
import pandas as pd
import numpy as np
import torch.nn as nn
import pdb
import subprocess
from time import tzset, sleep, time
from multiprocessing import Pool
from sklearn.decomposition import KernelPCA
from sklearn.cluster import AffinityPropagation
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from sklearn.metrics import precision_score, roc_auc_score
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn import linear_model
from sklearn import preprocessing
from platform import mac_ver


# Constants
_d_data_ = '../data/'
_p_nasdaq_listing_ = _d_data_ + 'NASDAQ.txt'
_p_nyse_listing_ = _d_data_ + 'NYSE.txt'
_p_amex_listing_ = _d_data_ + 'AMEX.txt'
_p_all_symbols_ = _d_data_ + 'all_symbols.txt'
_p_excluded_symbols_ = _d_data_ + 'excluded_symbols.txt'
_url_cnn_ = 'https://money.cnn.com/quote/quote.html?symb=%s'
_url_yahoo_ = 'https://finance.yahoo.com/quote/%s'
_url_yahoo_daily_ = 'https://finance.yahoo.com/quote/%s/history?period1=%d&period2=%d&interval=1d&filter=history&frequency=1d'

# To by-pass Mac's new security things that causes multiprocessing to crash
try:
    v = mac_ver()
    if v and float(v[0]) >= 10:
        print('Detected Mac > High Sierra, deploy multiprocessing fix')
        _ = subprocess.run('export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES'.split())
except:
    print('Did not detect MAC')


# Classes
class Stock:

    def __init__(self):

        os.environ['TZ'] = 'America/New_York'
        tzset()

        self._date0_ = datetime.datetime(1969,12,31,20,0)
        self._dir_full_history_ = _d_data_ + 'full_history/'
        self._dir_live_quotes_ = _d_data_ + 'live_quotes/'
        self._open_time_ = 9 * 3600
        self._close_time_ = 18 * 3600
        self._date_format_ = '%Y-%m-%d'
        self._date_time_format_ = '%Y-%m-%d-%H-%M-%S'
        self._max_connection_attempts_ = 20

        self.rexp_data_row = re.compile(r'{"date".*?}')
        self.rexp_dollar = re.compile(r'starQuote.*?<')
        self.rexp_volume = re.compile(r'olume<.+?</td>')

        self.n_cpu = os.cpu_count()
        self.verbose = False
        self.symbs = list()
        self.live_now = False
        self.dfs = None

        if not os.path.isdir(self._dir_full_history_):
            os.makedirs(self._dir_full_history_)
            print('Created: %s' % self._dir_full_history_)

        if not os.path.isdir(self._dir_live_quotes_):
            os.makedirs(self._dir_live_quotes_)
            print('Created: %s' % self._dir_live_quotes_)

    def pull_history(self, symb, period1=0, period2=0):

        if not period2:
            period2 = (datetime.datetime.now() - self._date0_).total_seconds()

        url = _url_yahoo_daily_ % (symb, period1, period2)

        r = self._try_request(url)
        if r is None:
            if self.verbose:
                print('Page load failed for %s' % symb)
            return -1

        if r.status_code != requests.codes.ok:
            if self.verbose:
                print('Page load failed for %s' % symb)
            return -1
        
        raw = self.rexp_data_row.findall(r.text)
        if not raw:
            if self.verbose:
                print('Cannot find data for %s' % symb)
            return -1
        raw.pop(0)

        data = []
        for r in raw:
            parts = r[1:-1].replace('"','').split(',')
            got_data = 0

            date = ''
            open_price = 0.
            high_price = 0.
            low_price = 0.
            close_price = 0.
            volume = 0
            adj_close_price = 0.

            for p in parts:

                d = p.split(':')
                if d[0] == 'date' and d[1] != 'null':
                    date = (self._date0_ + datetime.timedelta(seconds=int(d[1]))).strftime(self._date_format_)
                    got_data += 1

                elif d[0] == 'open' and d[1] != 'null':
                    open_price = float(d[1])
                    got_data += 1

                elif d[0] == 'high' and d[1] != 'null':
                    high_price = float(d[1])
                    got_data += 1

                elif d[0] == 'low' and d[1] != 'null':
                    low_price = float(d[1])
                    got_data += 1

                elif d[0] == 'close' and d[1] != 'null':
                    close_price = float(d[1])
                    got_data += 1

                elif d[0] == 'volume' and d[1] != 'null':
                    volume = int(d[1])
                    got_data += 1

                elif d[0] == 'adjclose' and d[1] != 'null':
                    adj_close_price = float(d[1])
                    got_data += 1

            if got_data == 7:
                data.append([date, volume, open_price, close_price, high_price, low_price, adj_close_price])

        if self.verbose:
            print('Pulled %d days of %s' % (len(data), symb))

        return data

    def write_history(self, data, symb, dir_out=''):

        if not dir_out:
            dir_out = self._dir_full_history_

        p_out = dir_out + '%s.csv' % symb

        with open(p_out, 'w+') as f:

            _ = f.write('date,volume,open,close,high,low,adjclose\n')

            for d in data:
                _ = f.write('%s\n' % (','.join(['{}'.format(x) for x in d])))

        if self.verbose:
            print('Wrote %s history at %s' % (symb, p_out))

        return p_out

    def all_symbols(self, try_compiled=True):

        symbs = []

        if try_compiled and os.path.isfile(_p_all_symbols_):

            if self.verbose:
                print('Using %s for symbols ...' % _p_all_symbols_)

            with open(_p_all_symbols_, 'r') as f:
                symbs = list(set(f.read().strip().split('\n')))

        elif os.path.isfile(_p_nasdaq_listing_) and os.path.isfile(_p_nyse_listing_) and os.path.isfile(_p_amex_listing_):

            if self.verbose:
                print('Getting symbols from:\n\t%s\n\t%s\n\t%s' % (_p_nasdaq_listing_, _p_nyse_listing_, _p_amex_listing_))

            rexp = re.compile(r'-[a-zA-Z]$')

            nasdaq_symbs = [rexp.sub('',s) for s in pd.read_table(_p_nasdaq_listing_)['Symbol'].values]
            nyse_symbs = [rexp.sub('',s) for s in pd.read_table(_p_nyse_listing_)['Symbol'].values]
            amex_symbs = [rexp.sub('', s) for s in pd.read_table(_p_amex_listing_)['Symbol'].values]
            symbs = list(set(nasdaq_symbs) | set(nyse_symbs) | set(amex_symbs))

        elif self.verbose:
            print('Missing symbol file.')

        if symbs and self.verbose:
            print('\tFound %d symbols' % len(symbs))

        return sorted(symbs)

    def retrieve_all_symbs(self, symbs=None, p_symbs='', i_pass=1, max_pass=5):

        if p_symbs:
            if os.path.isfile(p_symbs):
                with open(p_symbs, 'r') as f:
                    symbs = f.read().strip().split('\n')
            else:
                print('No such file: %s' % p_symbs)

        elif symbs is None:
            symbs = self.all_symbols()
            updated_symbs = self.updated_symbs()
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

            with Pool(processes=self.n_cpu) as pool:

                res = pool.map(self.retrieve_symb, symbs)

            for symb,success in res:
                if success:
                    n_success += 1
                else:
                    failed_symbs.append(symb)

        else:

            for i, symb in enumerate(symbs):

                if self.verbose:
                    print('Pulling %d / %d - %s ...' % (i+1, n_symbs, symb))

                data = self.pull_history(symb)
                if data != -1:
                    _ = self.write_history(data, symb)
                    n_success += 1
                else:
                    failed_symbs.append(symb)

        if self.verbose:
            print('\nRetrieved full histories for %d / %d symbols' % (n_success, n_symbs))

        if failed_symbs:
            print('Failed for:')
            for symb in failed_symbs:
                print('\t%s' % symb)

            if i_pass < max_pass:
                i_pass += 1
                print('\n|--- Pass %d (try to fetch %d failed ones, maximum %d passes ---|' % (i_pass, len(failed_symbs), max_pass))
                self.retrieve_all_symbs(symbs=failed_symbs, i_pass=i_pass, max_pass=max_pass)
            else:
                p_symbs = 'failed_symbs-%s.txt' % (''.join(np.random.choice(list('abcdefgh12345678'), 5)))
                with open(p_symbs, 'w+') as f:
                    _ = f.write('\n'.join(failed_symbs))
                print('Failed symbols written to: %s' % p_symbs)
                print('Run this to try fetching the missed symbols again:\npython3 redtide.py -v -d --file %s' % p_symbs)

        return

    def updated_symbs(self):

        t = datetime.datetime.today()
        t_close = datetime.datetime(year=t.year, month=t.month, day=t.day, hour=18).timestamp()

        symbs = []
        for p in os.listdir(self._dir_full_history_):
            t = os.path.getmtime(self._dir_full_history_ + p)
            if t > t_close:
                symbs.append(p.replace('.csv', ''))

        if self.verbose:
            print('Full history up-to-date for %d symbols' % len(symbs))

        return symbs

    def retrieve_symb(self, symb):

        success = False

        if self.verbose:
            print('Pulling %s ...' % symb)

        data = self.pull_history(symb)
        if data != -1:
            _ = self.write_history(data, symb)
            success = True

        return symb, success

    def read_full_histories(self, symbs=None):

        self.dfs = {}

        if not symbs:

            symbs = self.all_symbols()

        if self.verbose:
            print('Reading %d full histories to dataframes ...' % len(symbs))

        for symb in symbs:

            p_data = self._dir_full_history_ + '%s.csv' % symb

            if not os.path.isfile(p_data):
                if self.verbose:
                    print('No full history available: %s' % symb)
                continue

            df = pd.read_csv(p_data, index_col='date', parse_dates=True)
            self.dfs[symb] = df

        if self.verbose:
            print('Read %d / %d full histories to dataframes' % (len(self.dfs), len(symbs)))

        return

    @staticmethod
    def transform(df, shift0=1, shift1=-1, ratio0=0.01, ratio1=0.01):

        tmp0 = (1 - df['low'].shift(shift0) / df['open'] >= ratio0).values
        tmp1 = (df['high'].shift(shift1) / df['open'] - 1 >= ratio1).values

        n_df0 = len(df[tmp0])
        if n_df0:
            rate = len(df[tmp0 & tmp1]) / n_df0
        else:
            rate = np.nan

        freq = n_df0 / (len(df) - (abs(shift0) + abs(shift1)))

        return rate, freq

    def gen_features(self, symbs, n_sampling=10, n_project=1):

        not_in_dfs = []
        not_enough_data = []
        for symb in symbs:
            if symb not in self.dfs:
                not_in_dfs.append(symb)
            if len(self.dfs[symb]) < n_sampling + 1:
                not_enough_data.append(symb)

        if not_enough_data and not_in_dfs:
            raise ValueError('%s not in dataframe, and not enough data for %s' % (''.join(not_in_dfs), ''.join(not_enough_data)))
        elif not_in_dfs:
            raise ValueError('%s not in dataframe' % ''.join(not_in_dfs))
        elif not_enough_data:
            raise ValueError('Not enough data for %s' % ''.join(not_enough_data))

        feats = None
        labels = None
        for symb in symbs:
            f, l = self.gen_feature(symb=symb, n_sampling=n_sampling, n_project=n_project)
            if feats is None and labels is None:
                feats = f
                labels = l
            else:
                feats = np.concatenate([feats, f])
                labels = np.concatenate([labels, l])

        return feats, labels

    def gen_feature(self, symb, n_sampling=10, n_project=1):

        if symb not in self.dfs:
            raise ValueError('{} not in read stock dataframe dictionary'.format(symb))

        if len(self.dfs[symb]) < n_sampling+1:
            raise ValueError('Not enough data for %s to sample %d + %d' % (symb, n_sampling, n_project))

        if self.verbose:
            print('Generating features/labels for %s' % symb)
        df = self.dfs[symb].iloc[::-1]
        features = []
        labels = []
        for i in range(1, len(df) - n_sampling - n_project):
            data = df[(i-1):(n_sampling+i)].values
            if not data.all():
                continue
            x = (data[1:n_sampling+1] / data[:n_sampling]).flatten()

            tmp = df.iloc[n_sampling+i-1]['close']
            if not tmp:
                continue
            tell = df.iloc[n_sampling+i]['open'] / tmp

            tmp = df.iloc[n_sampling+i]['open']
            if not tmp:
                continue
            y1 = df.iloc[n_sampling+i]['low'] / tmp

            tmp = df.iloc[n_sampling+i]['low']
            if not tmp:
                continue
            y2 = df.iloc[n_sampling+i+n_project-1]['high'] / tmp

            features.append(np.concatenate([x, np.array([tell], float)], 0))
            labels.append(np.concatenate([[y1], [y2]]))

        return np.array(features), preprocessing.scale(np.array(labels), axis=0)
    
    def analyze(self, from_date='', to_date=''):

        if not to_date:
            to_date = self.dfs[list(self.dfs.keys())[0]][0:1].index[0].to_pydatetime().strftime('%Y-%m-%d')

        if not from_date:
            from_date = (datetime.datetime.strptime(to_date, '%Y-%m-%d') - datetime.timedelta(days=365)).strftime('%Y-%m-%d')

        print('Analyzing from %s to %s ...' % (from_date, to_date))

        for symb in self.dfs:

            rate, freq = self.transform(self.dfs[symb].loc[to_date:from_date], 1, -1, 0.01, 0.01)
            if not np.isnan(rate) and rate >= 0.9 and freq >= 0.5:
                    print('%s - %.2f (%.2f)' % (symb, rate, freq))

        return

    def pca(self, X):

        print('\nLinear PCA')

        m = KernelPCA(kernel='linear', n_jobs=self.n_cpu)
        m.fit(X)

        l = m.lambdas_ / sum(m.lambdas_)
        for i in range(5):
            print('%d: %.4f' % (i+1, l[i]))
        print('First 10: %.4f' % sum(l[:10]))

        return m

    def range_norm(self, from_date='', to_date=''):

        if not to_date:
            to_date = self.dfs[list(self.dfs.keys())[0]][0:1].index[0].to_pydatetime().strftime('%Y-%m-%d')

        if not from_date:
            from_date = (datetime.datetime.strptime(to_date, '%Y-%m-%d') - datetime.timedelta(days=365)).strftime('%Y-%m-%d')

        print('Normalizing from %s to %s ...' % (from_date, to_date))

        X = []
        days = []
        symbs = []
        for symb in self.dfs:
            df = self.dfs[symb].loc[to_date:from_date]
            if not df.empty:
                x = df['high'].values
                if not np.isnan(x).any():
                    x_min = np.min(x)
                    dx = np.max(x) - x_min
                    if dx != 0:
                        x = (x - x_min) / dx
                        X.append(x)
                        days.append(len(x))
                        symbs.append(symb)

        max_days = max(days)
        idx = [i for i,x in enumerate(X) if len(x) == max_days]
        X = [X[i] for i in idx]
        symbs = [symbs[i] for i in idx]
        X = np.array(X)
        print('\tNumber of samples: %d' % len(X))

        return X, symbs

    def cluster(self, X, symbs):

        print('\nClustering ...')

        '''m = self.pca(X)
        X = m.alphas_[:,0:50]'''

        clf = AffinityPropagation().fit(X)

        labels = list(set(clf.labels_))
        for l in labels:
            print('Group %d: %s\n' % (l, ','.join([symbs[i] for i,j in enumerate(clf.labels_) if j == l])))

        return clf

    def get_live_quote(self, symbs=None, interval=600):

        if not symbs:
            symbs = self.all_symbols()

        if self.verbose:
            print('Retrieving live quotes for %d symbols ...' % len(symbs))

        while 1:

            wday = datetime.datetime.now().weekday()

            t_current = self._get_current_time()
            while self.live_now or (0 <= wday <= 4 and self._open_time_ <= t_current <= self._close_time_):
                t0 = time()
                with Pool(processes=self.n_cpu) as pool:
                    pool.map(self._get_live_quote, symbs)

                if self.verbose:
                    print('%s - %d s' % (datetime.datetime.now().strftime(self._date_time_format_), time() - t0))

                t_wait = t_current + interval - self._get_current_time()
                if t_wait > 0:
                    sleep(t_wait)
                t_current = self._get_current_time()

            t2open = self._open_time_ - self._get_current_time()
            if t2open < 0:
                t2open += 86400

            if 0 <= wday < 4:
                while t2open > 0:
                    if self.verbose:
                        print('Waiting %s to open ...' % self._time_str(t2open))
                    sleep(t2open)
                    t2open = self._open_time_ - self._get_current_time()
            else:
                t2open += (7 - wday) * 86400

                while t2open > 0:
                    if self.verbose:
                        print('Waiting %s to open ...' % self._time_str(t2open))
                    sleep(t2open)
                    t2open = self._open_time_ - self._get_current_time()

    @staticmethod
    def _get_current_time():

        t = datetime.datetime.now()
        h = t.hour * 3600
        m = t.minute * 60
        s = t.second

        return h + m + s

    def _get_live_quote(self, symb):

        ts = datetime.datetime.now().strftime(self._date_time_format_)

        try:
            n_try = 0
            url = 'https://money.cnn.com/quote/quote.html?symb=%s' % symb
            r = requests.get(url)
            while r.status_code != requests.codes.ok and n_try < 10:
                r = requests.get(url)
                n_try += 1
        except(KeyboardInterrupt, SystemExit):
            raise
        except:
            if self.verbose:
                print('Failed to get %s at %s' % (symb, ts))
            return

        if r.status_code != requests.codes.ok:
            if self.verbose:
                print('Page load failed for %s' % symb)
            return

        match = self.rexp_dollar.search(r.text)
        if not match:
            if self.verbose:
                print('Cannot find quote for %s' % symb)
            return
        quote = match.group()[11:-1]

        match = self.rexp_volume.search(r.text)
        if not match:
            if self.verbose:
                print('Cannot find volumn for %s' % symb)
            return
        volume = match.group()[42:].replace('</td>','').replace(',','')

        p_data = self._dir_live_quotes_ + '%s.csv' % symb

        if not os.path.isfile(p_data):
            with open(p_data, 'w+') as f:
                _ = f.write('date,price,volume\n')

        with open(p_data, 'a') as f:
            _ = f.write('%s,%s,%s\n' % (ts, quote, volume))

        return

    @staticmethod
    def _time_str(dtime):

        minutes = dtime / 60
        hours = minutes / 60
        days = hours / 24

        if days >= 1.:
            return '%.1f days' % days

        elif hours >= 1.:
            return '%.1f hours' % hours

        elif minutes >= 1.:
            return '%.1f minutes' % minutes

        return '%d seconds' % dtime

    def compile_symbols(self):

        print('Compiling symbols ...')

        self.symbs = self.all_symbols(try_compiled=False)
        n_symbs = len(self.symbs)
        n_compiled = 0
        n_excluded = 0

        # Initialize/clear write files
        with open(_p_all_symbols_, 'w+') as f, open(_p_excluded_symbols_, 'w+') as fx:
            _ = f.write('')
            _ = fx.write('')

        if self.verbose:
            print('Looking up symbols on Yahoo Finance ...')

        symb_batches = self._gen_symbol_batches(self.symbs)
        n_symb_completed = 0
        for batch in symb_batches:
            with Pool(processes=self.n_cpu) as pool:
                res = pool.map(self._compile_symb, batch)
            pool.close()
            pool.terminate()

            with open(_p_all_symbols_, 'a+') as f, open(_p_excluded_symbols_, 'a+') as fx:
                for symb, success in res:
                    if success:
                        _ = f.write('%s\n' % symb)
                        n_compiled += 1
                    else:
                        _ = fx.write('%s\n' % symb)
                        n_excluded += 1

            n_symb_completed += len(batch)
            if self.verbose:
                print('{}% completed - {} / {}'.format(int(n_symb_completed / n_symbs * 100), n_symb_completed, n_symbs))

        print('Started with: %d\nCompiled: %d\nExcluded: %d\nCompiled to: %s' % (n_symbs, n_compiled, n_excluded, _p_all_symbols_))

        return

    @staticmethod
    def _gen_symbol_batches(symbs, n_batches=20):

        symb_batches = []
        n_symbs = len(symbs)
        batch_size = np.ceil(n_symbs / n_batches)

        batch = []
        for i, s in enumerate(symbs):

            if i % batch_size == 0:
                if batch:
                    symb_batches.append(batch)
                batch = []

            batch.append(s)

        if batch:
            symb_batches.append(batch)

        return symb_batches

    def _try_request(self, url, n_tries=1):

        r = None

        try:
            r = requests.get(url)
        except(KeyboardInterrupt, SystemExit):
            raise
        except:
            if n_tries < self._max_connection_attempts_:
                r = self._try_request(url, n_tries+1)

        return r

    def _check_symbol_(self, symb, try_cnn=False):

        """
        Check whether a symbol exists

        :param symb: String
        :return: 1 - found, 0 - not found, -1 - connection error, -x - request message code x
        """

        msg = 0

        # Try Yahoo Finance first
        url = _url_yahoo_ % symb
        r = self._try_request(url)
        if r is None:
            msg = -1
        else:
            if r.status_code != requests.codes.ok:
                msg = -r.status_code
            elif 'Symbol Lookup' not in r.text[:300]:
                msg = 1

            if msg != 1 and try_cnn:
                # Try CNN Money
                url = _url_cnn_ % symb
                r = self._try_request(url)
                if r is None:
                    msg = -1
                else:
                    if r.status_code != requests.codes.ok:
                        msg = -r.status_code
                    elif 'Symbol not found' not in r.text[:250]:
                        msg = 1

        return msg

    def _compile_symb(self, symb):

        msg = self._check_symbol_(symb)
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

class NN(nn.Module):

    def __init__(self, n_sampling=10):
        super(NN, self).__init__()

        self.n_sampling = n_sampling

        self.n_l1 = 2 * self.n_sampling

        self.bn = nn.BatchNorm1d(self.n_sampling)
        self.l1 = nn.Linear(self.n_sampling, self.n_l1)
        self.l2 = nn.Linear(self.n_l1, self.n_l1)
        self.l3 = nn.Linear(self.n_l1, self.n_l1)
        self.l4 = nn.Linear(self.n_l1, 1)

    def forward(self, input):

        return self.l4(self.l3(self.l2(self.l1(self.bn(input)))))

class Regressor:

    def __init__(self, n_sampling0=10, clf_type='rf', kfold=5):

        self.n_sampling0 = n_sampling0
        self.clf_type = clf_type
        self.kfold = kfold

        self.model = None
        self.model2 = None
        self.lr0 = None
        self.lr1 = None

    def train(self, features, labels):

        if self.clf_type == 'nn':
            self.train_nn()
        elif self.clf_type == 'rf':
            self.train_rf(features, labels)

        return

    def train_nn(self):

        self.model = NN(n_sampling=self.n_sampling0)

        return

    def train_rf(self, features, labels):

        print('Training random forest ...')

        self.model = RandomForestRegressor(n_estimators=100,
                                           max_features='sqrt',
                                           max_depth=np.ceil(len(features[0])/5),
                                           min_samples_leaf=3,
                                           n_jobs=-1)

        self.model2 = RandomForestClassifier(
            n_estimators=100,
            max_features='sqrt',
            max_depth=np.ceil(len(features[0])/5),
            min_samples_leaf=3,
            n_jobs=-1
        )

        self.lr0 = linear_model.TheilSenRegressor()
        self.lr1 = linear_model.TheilSenRegressor()

        reg_dummy = DummyRegressor()
        clf_dummy = DummyClassifier()

        kfold = KFold(n_splits=self.kfold, shuffle=True)
        kfold2 = KFold(n_splits=self.kfold, shuffle=True)

        features, labels = shuffle(features, labels)

        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set(style='whitegrid', context='paper')

        for ifold, (train, test) in enumerate(kfold.split(labels)):
            self.model.fit(features[train], labels[train])
            score_train = self.model.score(features[train], labels[train])
            score_test = self.model.score(features[test], labels[test])
            reg_dummy.fit(features[train], labels[train])
            score_dummy = reg_dummy.score(features[test], labels[test])
            print('Fold %d: %.4f / %.4f (%.4f)' % (ifold, score_test, score_train, score_dummy))

            labels_t = labels.transpose()
            y_pred = self.model.predict(features)
            y_pred_t = y_pred.transpose()
            # self.lr0.fit(labels_t[0][train].reshape(-1, 1), y_pred_t[0][train])
            self.lr1.fit(labels_t[1][train].reshape(-1, 1), y_pred_t[1][train])
            y_lr = self.lr1.predict(labels_t[1][test].reshape(-1,1))
            dy = np.abs(y_pred_t[1][test] - y_lr) < 0.2
            print('\t%d / %d' % (np.sum(dy), np.sum(1 - dy)))
            for jfold, (train2, test2) in enumerate(kfold2.split(dy)):
                self.model2.fit(features[test[train2]], dy[train2])
                y_pred2 = self.model2.predict(features[test[test2]])
                score_train2 = precision_score(dy[train2], self.model2.predict(features[test[train2]]), average='binary')
                score_test2 = precision_score(dy[test2], y_pred2, average='binary')
                clf_dummy.fit(features[test[train2]], dy[train2])
                score_dummy = precision_score(dy[test2], clf_dummy.predict(features[test[test2]]), average='binary')
                print('\tFold %d: %.4f / %.4f (%.4f)' % (jfold, score_test2, score_train2, score_dummy))

                score_final_train = self.model.score(features[test[train2]], labels[test[train2]])
                score_final_test = self.model.score(features[test[test2[y_pred2]]], labels[test[test2[y_pred2]]])
                print('\tFinal: %.4f / %.4f' % (score_final_test, score_final_train))

            fig, axs = plt.subplots(2,2)
            train_truth = labels[train].transpose()
            train_pred = self.model.predict(features[train]).transpose()
            test_truth = labels[test].transpose()
            test_pred = y_pred[test].transpose()
            sns.scatterplot(x=train_truth[0], y=train_pred[0], ax=axs[0,0])
            sns.scatterplot(x=train_truth[1], y=train_pred[1], ax=axs[0,1])
            sns.scatterplot(x=test_truth[0][test2[y_pred2]], y=test_pred[0][test2[y_pred2]], ax=axs[1, 0])
            sns.scatterplot(x=test_truth[1][test2[y_pred2]], y=test_pred[1][test2[y_pred2]], ax=axs[1,1])
            plt.draw()

        plt.show()

        return
