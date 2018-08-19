#!/usr/bin/env python3

# Import
import requests
import re
import datetime
import os
import pandas as pd
import numpy as np
from time import tzset, sleep, time
from multiprocessing import Pool
from sklearn.decomposition import KernelPCA
from sklearn.cluster import SpectralClustering


# Constants
_d_data_ = '../data/'
_nasdaq_listing_ = _d_data_ + 'NASDAQ.txt'
_nyse_listing_ = _d_data_ + 'NYSE.txt'


# Classes
class Stock():

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

        self.rexp_data_row = re.compile(r'{"date".*?}')
        self.rexp_dollar = re.compile('currentPrice":{"raw":.+?,')

        self.n_cpu = os.cpu_count()
        self.verbose = False

        if not os.path.isdir(self._dir_full_history_):
            os.makedirs(self._dir_full_history_)
            print('Created: %s' % self._dir_full_history_)

        if not os.path.isdir(self._dir_live_quotes_):
            os.makedirs(self._dir_live_quotes_)
            print('Created: %s' % self._dir_live_quotes_)

    def pull_history(self, symb, period1=0, period2=0):

        if not period2:
            period2 = (datetime.datetime.now() - self._date0_).total_seconds()

        url = 'https://finance.yahoo.com/quote/%s/history?period1=%d&period2=%d&interval=1d&filter=history&frequency=1d' % (symb, period1, period2)

        try:
            r = requests.get(url)
        except(KeyboardInterrupt, SystemExit):
            raise
        except:
            if self.verbose:
                print('Failed to pull for %s' % symb)
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

        return(data)

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

        return(p_out)

    def _all_symbols(self):

        rexp = re.compile(r'-[a-zA-Z]$')

        nasdaq_symbs = [rexp.sub('',s) for s in pd.read_table(_nasdaq_listing_)['Symbol'].values]
        nyse_symbs = [rexp.sub('',s) for s in pd.read_table(_nasdaq_listing_)['Symbol'].values]
        symbs = list(set(nasdaq_symbs) | set(nyse_symbs))

        return(symbs)

    def retrieve_all_symbs(self):

        symbs = self._all_symbols()

        n_symbs = len(symbs)
        n_success = 0
        failed_symbs = []
        print('Pulling for %d symbols on NASDAQ and NYSE ...' % n_symbs)

        if self.n_cpu > 1:

            print('\tUsing %d CPUs' % self.n_cpu)

            with Pool(processes=self.n_cpu) as pool:

                res = pool.map(self._retrieve_symb, symbs)

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

        return

    def _retrieve_symb(self, symb):

        success = False

        if self.verbose:
            print('Pulling %s ...' % symb)

        data = self.pull_history(symb)
        if data != -1:
            _ = self.write_history(data, symb)
            success = True

        return(symb, success)

    def read_full_histories(self, symbs=list()):

        dfs = {}

        if not symbs:

            symbs = self._all_symbols()

        if self.verbose:
            print('Reading %d full histories to dataframes ...' % len(symbs))

        for symb in symbs:

            p_data = self._dir_full_history_ + '%s.csv' % symb

            if not os.path.isfile(p_data):
                if self.verbose:
                    print('No full history available: %s' % symb)
                continue

            df = pd.read_csv(p_data, index_col='date', parse_dates=True)
            dfs[symb] = df

        if self.verbose:
            print('Read %d / %d full histories to dataframes' % (len(dfs), len(symbs)))

        return(dfs)

    def transform(self, df, shift0=1, shift1=-1, ratio0=0.01, ratio1=0.01):

        tmp0 = (1 - df['close'].shift(shift0) / df['open'] >= ratio0).values
        tmp1 = (df['open'].shift(shift1) / df['open'] - 1 >= ratio1).values

        rate = len(df[tmp0 & tmp1]) / len(df[tmp0])

        return(rate)

    def analyze(self, dfs):

        rates = []

        for symb in dfs:

            rate = self.transform(dfs[symb], 1, -1, 0.01, 0.01)
            rates.append(rate)

        mean = np.mean(rates)
        std = np.std(rates)
        min_rate = np.min(rates)
        max_rate = np.max(rates)

        print('\nMean (SD): %.3f (%.3f)' % (mean, std))
        print('%.3f - %.3f' % (min_rate, max_rate))

        return

    def pca(self, dfs, from_date='', to_date=''):

        print('\nLinear PCA')

        X, _ = self.range_norm(dfs, from_date=from_date, to_date=to_date)

        m = KernelPCA(kernel='linear', n_jobs=self.n_cpu)
        m.fit(X)

        l = m.lambdas_ / sum(m.lambdas_)
        for i in range(5):
            print('%d: %.4f' % (i+1, l[i]))
        print('First 10: %.4f' % sum(l[:10]))

        return

    def range_norm(self, dfs, from_date='', to_date=''):

        if not to_date:
            to_date = dfs[list(dfs.keys())[0]][0:1].index[0].to_pydatetime().strftime('%Y-%m-%d')

        if not from_date:
            from_date = (datetime.datetime.strptime(to_date, '%Y-%m-%d') - datetime.timedelta(days=365)).strftime('%Y-%m-%d')

        print('Normalizing from %s to %s ...' % (from_date, to_date))

        X = []
        days = []
        symbs = []
        for symb in dfs:
            df = dfs[symb].loc[to_date:from_date]
            if not df.empty:
                df = df['open']
                x = ((df - df.min())/(df.max() - df.min())).values
                X.append(x)
                days.append(len(x))
                symbs.append(symb)

        max_days = max(days)
        idx = [i for i,x in enumerate(X) if len(x) == max_days]
        X = [X[i] for i in idx]
        symbs = [symbs[i] for i in idx]
        X = np.array(X)
        print('\tNumber of samples: %d' % len(X))

        return(X, symbs)

    def cluster(self, dfs, from_date='', to_date=''):

        print('\nClustering ...')

        X,symbs = self.range_norm(dfs, from_date=from_date, to_date=to_date)

        clf = SpectralClustering().fit(X)

        labels = list(set(clf.labels_))
        for l in labels:
            print('Group %d: %s\n' % (l, ','.join([symbs[i] for i,j in enumerate(clf.labels_) if j == l])))

        return(clf)

    def get_live_quote(self, symbs=list(), interval=120):

        if not symbs:
            symbs = self._all_symbols()

        while 1:

            wday = datetime.datetime.now().weekday()

            while 0 <= wday <= 4 and self._open_time_ <= self._get_current_time() <= self._close_time_:
                t0 = time()
                with Pool(processes=self.n_cpu) as pool:
                    pool.map(self._get_live_quote, symbs)
                print('%s - %d s' % (datetime.datetime.now().strftime(self._date_time_format_), time() - t0))
                sleep(interval)

            t2open = self._open_time_ - self._get_current_time()
            if t2open < 0:
                t2open += 86400

            if 0 <= wday < 4:
                while t2open > 0:
                    print('Waiting %s to open ...' % self._time_str(t2open))
                    sleep(t2open)
                    t2open = self._open_time_ - self._get_current_time()
            else:
                t2open += (6 - wday) * 86400

                while t2open > 0:
                    print('Waiting %s to open ...' % self._time_str(t2open))
                    sleep(t2open)
                    t2open = self._open_time_ - self._get_current_time()

    def _get_current_time(self):

        t = datetime.datetime.now()
        h = t.hour * 3600
        m = t.minute * 60
        s = t.second

        return(h + m + s)

    def _get_live_quote(self, symb):

        ts = datetime.datetime.now().strftime(self._date_time_format_)

        try:
            r = requests.get('https://finance.yahoo.com/quote/%s' % symb)
            quote = self.rexp_dollar.search(r.text).group()[21:-1]

            p_data = self._dir_live_quotes_ + '%s.csv' % symb

            if not os.path.isfile(p_data):
                with open(p_data, 'w+') as f:
                    _ = f.write('date,price\n')

            with open(p_data, 'a') as f:
                _ = f.write('%s,%s\n' % (ts, quote))

        except(KeyboardInterrupt, SystemExit):
            raise

        except:
            print('Failed to get %s at %s' % (symb, ts))

        return

    def _time_str(self, dtime):

        minutes = dtime / 60
        hours = minutes / 60
        days = hours / 24

        if days >= 1.:
            return('%.1f days' % days)

        elif hours >= 1.:
            return('%.1f hours' % hours)

        elif minutes >= 1.:
            return('%.1f minutes' % minutes)

        return('%d seconds' % dtime)
