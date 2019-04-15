import os

_d_classes_ = os.path.realpath(os.path.split(__file__)[0]) + '/'
_d_data_ = _d_classes_ + '../data/'
_p_nasdaq_listing_ = _d_data_ + 'NASDAQ.txt'
_p_nyse_listing_ = _d_data_ + 'NYSE.txt'
_p_amex_listing_ = _d_data_ + 'AMEX.txt'
_p_all_symbols_ = _d_data_ + 'all_symbols.txt'
_p_excluded_symbols_ = _d_data_ + 'excluded_symbols.txt'
_url_cnn_ = 'https://money.cnn.com/quote/quote.html?symb=%s'
_url_yahoo_ = 'https://finance.yahoo.com/quote/%s'
_url_yahoo_daily_ = 'https://finance.yahoo.com/quote/%s/history?period1=%d&period2=%d&interval=1d&filter=history&frequency=1d'