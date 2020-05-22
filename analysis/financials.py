import os
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict

from src.constants import DIR_FINANCIALS


class FinancialAnalysis(object):

    def __init__(self):
        symbols = [p[:-4] for p in os.listdir(DIR_FINANCIALS)
                   if os.path.isfile(os.path.join(DIR_FINANCIALS, p))]
        self.financials = self.load_financial_data(symbols)

        # compile balancesheets
        print('Compiling balancesheets ...')
        self.req_cols = {'stock', 'T', 'netIncome'}
        self.df_yr = None
        self.symbols_yr = None
        self.df_qt = None
        self.symbols_qt = None
        self.symbols_both = None
        self.compile_all_balancesheets(req_cols=self.req_cols)
        self.prices = {}

        # evaluate
        print('Evaluating performances ...')
        self.greats = None
        self.bads = None
        self.sorted_greats = None
        self.great_corrs = None
        self.sorted_bads = None
        self.bad_corrs = None
        self.evaluate()

    @property
    def symbols(self):
        return list(self.financials.keys())

    @property
    def n_symbols(self):
        return len(self.financials.keys())

    @property
    def n_symbols_yr(self):
        return len(self.symbols_yr)

    @property
    def n_symbols_qt(self):
        return len(self.symbols_qt)

    @property
    def n_symbols_both(self):
        return len(self.symbols_both)

    @property
    def n_symbols_greats(self):
        return len(self.greats)

    @property
    def n_symbols_bads(self):
        return len(self.bads)

    def price(self, symbol):
        if symbol in self.financials:
            return self.financials[symbol].get('price', {})\
                .get('regularMarketPrice', {}).get('raw')

    @staticmethod
    def load_financial_data(symbols, currency='USD'):
        data = defaultdict(dict)
        for s in symbols:
            try:
                d = pickle.load(open(os.path.join(
                    DIR_FINANCIALS, s + '.pkl'), 'rb'))
            except:
                continue
            if currency:
                try:
                    if d['earnings']['financialCurrency'] != currency:
                        continue
                except:
                    continue
            data[s] = d
        print('Loaded quote summary for %d stocks' % len(data))
        return data

    @staticmethod
    def _stock_filter(data, min_market_cap=2e9, min_daily_vol=1e6):
        try:
            if data['price']['marketCap']['raw'] >= min_market_cap \
                    and data['price']['averageDailyVolume10Day']\
                    ['raw'] >= min_daily_vol:
                return True
        except:
            pass
        return False

    def compile_metric(self, metric='regularMarketOpen'):
        x = []
        for v in self.financials.values():
            if self._stock_filter(v):
                x.append(v['price'][metric]['raw'])
        return np.array(x)

    def _get_history(self, symb, quarterly=False, key='cashflowStatementHistory', fmt={'endDate'}, ignore={'maxAge'}):
        data = self.financials[symb]
        key = key + ('Quarterly' if quarterly else '')
        bals = data.get(key)
        if bals is None:
            return None
        subkeys = list(bals.keys())
        subkeys.remove('maxAge')
        if subkeys:
            bals = bals.get(subkeys[0])
        else:
            return None
        if not bals:
            return None

        x = defaultdict(list)
        n = len(bals)
        for i, b in enumerate(bals):
            x['T'].append(n - i)
            for k, v in b.items():
                if k in ignore:
                    continue
                if k in fmt:
                    x[k].append(v['fmt'])
                elif v:
                    x[k].append(v['raw'] if 'raw' in v else v)
        d = {}
        while x:
            k, v = x.popitem()
            if len(v) != n:
                continue
            d[k] = v
        return pd.DataFrame(d).assign(stock=symb)

    def full_compile(self, quarterly=False, key='cashflowStatementHistory'):
        df = pd.concat(
            [self._get_history(s, quarterly=quarterly, key=key) for s in self.financials],
            ignore_index=True)
        cols = list(df.columns)
        cols.remove('stock')
        cols.remove('T')
        cols.remove('endDate')
        cols = ['stock', 'endDate'] + cols
        return df[cols]

    def _compile_balancesheets(self, quarterly=False, req_cols={'netIncome'}, col_standard=None):
        print('Compiling from %d loaded stocks' % self.n_symbols)

        df_list = []
        if not req_cols:
            if col_standard is None:
                col_standard = ['AMD', 'AAPL', 'INTC']

            req_cols = set.intersection(
                *[set(self._get_history(s, quarterly=quarterly).columns) for s in col_standard])

        print('Only with columns: {}'.format(req_cols))
        for s, d in self.financials.items():
            if not self._stock_filter(d):
                continue
            df = self._get_history(s, quarterly=quarterly)
            if df is not None:
                cols = set(df.columns)
                if not (req_cols - cols):
                    df_list.append(df[req_cols])

        print('Building dataframe - %d stocks' % len(df_list))
        return pd.concat(df_list)

    def compile_all_balancesheets(self, req_cols={'netIncome'}, col_standard=None):
        self.df_yr = self._compile_balancesheets(False, req_cols=req_cols, col_standard=col_standard)
        self.symbols_yr = list(self.df_yr['stock'].unique())
        self.df_qt = self._compile_balancesheets(True, req_cols=req_cols, col_standard=col_standard)
        self.symbols_qt = list(self.df_qt['stock'].unique())
        self.symbols_both = list(set(self.symbols_yr) & set(self.symbols_qt))
        return self

    def sort_history_symbs(self, df, symbs, by='netIncome', ascending=True):
        grouped = df.loc[np.any([df.stock == s for s in symbs], axis=0), ['T', 'stock', by]].groupby('stock')
        ser_corr = grouped.corr().loc[pd.IndexSlice[:, 'T'], by].droplevel(level=1).sort_values(ascending=ascending)
        return list(ser_corr.index), ser_corr

    @staticmethod
    def rank_stocks(lists, w=None):
        """
        Sort based on ranks in each list of sorted symbols in lists

        :param lists: list of list of symbols
        :param w: weight of each list
        :return:
        """
        if w is None:
            w = [1.] * len(lists)
        pts = defaultdict(list)
        for i, l in enumerate(lists):
            for j, s in enumerate(l):
                pts[s].append(j * w[i])
        return [k[0] for k in sorted(pts.items(), key=lambda kv: np.prod(kv[1]))]

    def find_bads(self, df):
        df_income = df[['T', 'stock', 'netIncome']].pivot(index='T', columns='stock', values='netIncome')
        df_noreturn = (df_income < 0).sum(axis=0)
        return set(df_noreturn[df_noreturn == 4].index)

    def find_greats(self, df):
        df_income = df[['T', 'stock', 'netIncome']].pivot(index='T', columns='stock', values='netIncome')
        df_allreturn = (df_income > 0).sum(axis=0)
        return set(df_allreturn[df_allreturn == 4].index)

    def is_bad(self, symb):
        if symb in self.symbols_both:
            if symb in self.bads:
                print('bad, rank %d / %d' % (self.sorted_bads.index(symb) + 1, len(self.sorted_bads)))
                return 1
            else:
                # not shit
                return 0
        else:
            # Don't know
            return -1

    def is_great(self, symb):
        if symb in self.symbols_both:
            if symb in self.greats:
                print('great, rank %d / %d' % (self.sorted_greats.index(symb) + 1, len(self.sorted_greats)))
                return 1
            else:
                # not great
                return 0
        else:
            # Don't know
            return -1

    def eval_bads(self, recent=True):
        if recent:
            self.bads = list(self.find_bads(self.df_qt))
        else:
            bad_yr = self.find_bads(self.df_yr)
            bad_qt = self.find_bads(self.df_qt)
            self.bads = list(bad_yr & bad_qt)

    def eval_greats(self, recent=True):
        if recent:
            self.greats = list(self.find_greats(self.df_qt))
        else:
            great_yr = self.find_greats(self.df_yr)
            great_qt = self.find_greats(self.df_qt)
            self.greats = list(great_yr & great_qt)

    def evaluate(self, recent=True):
        self.eval_bads(recent=recent)
        symbs, self.bad_corrs = self.sort_history_symbs(self.df_qt, self.bads)
        self.sorted_bads = self.rank_stocks([symbs])

        self.eval_greats(recent=recent)
        symbs, self.great_corrs = self.sort_history_symbs(self.df_qt, self.greats, ascending=False)
        self.sorted_greats = self.rank_stocks([symbs])

    def random_greats(self, n=10, trend_up=True):
        if trend_up:
            return np.random.choice(self.great_corrs[self.great_corrs > 0].index, n, replace=False)
        else:
            return np.random.choice(self.greats, n, replace=False)

    def random_bads(self, n=10, trend_down=True):
        if trend_down:
            return np.random.choice(self.bad_corrs[self.bad_corrs < 0].index, n, replace=False)
        else:
            return np.random.choice(self.bads, n, replace=False)
