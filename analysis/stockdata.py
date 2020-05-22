import os
import pandas as pd

from src.constants import DIR_HISTORY, DIR_FINANCIALS


class StockData(object):

    def __init__(self, symbols=None, date0='2010-01-01'):
        if symbols:
            if not isinstance(symbols, list) or not isinstance(symbols[0], str):
                raise ValueError('symbols arg must be a list of strings')
            self.symbols = symbols
            print('Given %d symbols' % len(symbols))
        else:
            self.symbols = [f.replace('.csv', '') for f in os.listdir(DIR_HISTORY)
                            if f.endswith('.csv')]
            print('Found %d symbols' % len(self.symbols))

        # entities with fiancial reports are considered "companies"
        self.companies = [f.replace('.pkl', '') for f in os.listdir(DIR_FINANCIALS)
                          if f.endswith('.pkl')]
        print('Found %d companies' % len(self.companies))

        self.df_history = None
        self.date0 = date0
        self.load_histories()

        self.df_corr = None
        self.corr_date0 = None
        self.df_interest = self.build_col_ratio()
        self.correlate(df=self.df_interest)

    @property
    def symbols(self):
        return self._symbols

    @symbols.setter
    def symbols(self, value):
        if not isinstance(value, list):
            raise ValueError('symbols must be list')
        self._symbols = value

    @property
    def companies(self):
        return self._companies

    @companies.setter
    def companies(self, value):
        if not isinstance(value, list):
            raise ValueError('companies must be list')
        self._companies = value

    @property
    def df_history(self):
        return self._df_history

    @df_history.setter
    def df_history(self, value):
        if not isinstance(value, pd.DataFrame) and value is not None:
            raise ValueError('df_history must be Pandas DataFrame')
        self._df_history = value

    @property
    def df_interest(self):
        return self._df_interest

    @df_interest.setter
    def df_interest(self, value):
        if not isinstance(value, pd.DataFrame) and value is not None:
            raise ValueError('df_interest must be Pandas DataFrame')
        self._df_interest = value

    @property
    def df_corr(self):
        return self._df_corr

    @df_corr.setter
    def df_corr(self, value):
        if not isinstance(value, pd.DataFrame) and value is not None:
            raise ValueError('df_corr must be Pandas DataFrame')
        self._df_corr = value

    def load_histories(self):
        print('Loading price histories...')
        history = []
        for s in self.symbols:
            p_history = os.path.join(DIR_HISTORY, '%s.csv' % s)
            history.append(pd.read_csv(p_history)\
               .set_index('date')\
               .sort_index()\
               .loc[self.date0:].reset_index()\
               .assign(symbol=s))

        print(' |- Building dataframe...')
        self.df_history = pd.concat(history)\
            .set_index(['date', 'symbol']).sort_index()

        print('Loaded price histories')
        return self

    def get_history_of_stock(self, symbol):
        return self.df_history.loc[pd.IndexSlice[:, symbol], :]

    def build_col_ratio(self, col1='high', col2='open', days=30):
        print('Building dataframe of {} / {} ratio over past {}'
              ' days...'.format(col1, col2, days))
        idx = self.df_history[col2] != 0
        df = (self.df_history[col1][idx] / self.df_history[col2][idx])\
            .reset_index().rename(columns={0:'ratio'})\
            .pivot(index='date', columns='symbol', values='ratio')\
            .iloc[-days:].dropna(axis=1)
        return df

    def correlate(self, df=None, col='adjclose', days=30, companies_only=True):
        if df is None and col:
            print('Correlating {} over past {} trade-days...'.format(col, days))
            df = self.df_history[col].reset_index()\
                     .pivot(index='date', columns='symbol', values=col)\
                     .iloc[-days:].dropna(axis=1)
            df = df.pct_change().dropna()
        else:
            print('Correlating with provided dataframe...')

        self.corr_date0 = df.index[0]

        if companies_only:
            symbs = list(set(self.companies) & set(df.columns))
            self.df_corr = df[symbs].corr()
        else:
            self.df_corr = df.corr()
        return self
