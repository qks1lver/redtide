import os
import yaml


# Redtide folder
DIR_MAIN = os.path.realpath(os.path.join(os.path.split(__file__)[0], '..'))


# Configuration file
FILE_CONFIG = os.path.join(DIR_MAIN, 'config.yaml')


# Load config
redtide_configs = yaml.load(open(FILE_CONFIG), Loader=yaml.FullLoader)


# Necessary directories and files
file_path_configs = redtide_configs.get('file_paths', {})
DIR_DATA = os.path.join(DIR_MAIN, file_path_configs.get('dir_data', 'data'))
DIR_LISTINGS = os.path.join(DIR_DATA, file_path_configs.get('dir_listings', 'listings'))
DIR_HISTORY = os.path.join(DIR_DATA, file_path_configs.get('dir_full_history', 'full_history'))
DIR_FINANCIALS = os.path.join(DIR_DATA, file_path_configs.get('dir_financials', 'financials'))
DIR_OPTIONS = os.path.join(DIR_DATA, file_path_configs.get('dir_options', 'options'))
FILE_ALL_SYMBOLS = os.path.join(DIR_DATA, file_path_configs.get('all_symbols', 'all_symbols.txt'))
FILE_EXCLUDED_SYMBOLS = os.path.join(DIR_DATA, file_path_configs.get('excluded_symbols', 'excluded_symbols.txt'))

# Selenium
GECKODRIVER_PATH = redtide_configs.get('selenium', {}).get('path_geckodriver', 'geckodriver')
SERVICE_LOG = 'NUL' if os.name == 'nt' else '/dev/null'


# Create data folder if it does not exist
if not os.path.isdir(DIR_DATA):
    os.makedirs(DIR_DATA)
    print('Created:', DIR_DATA)


# Compile listing files of exchanges
# These files should be under data/listings/
# and should have name <exchange_name>.txt
# i.e. NYSE.txt
def compile_listings():
    listing_files = {}
    if os.path.isdir(DIR_LISTINGS):
        for f in os.listdir(DIR_LISTINGS):
            ex_name = os.path.splitext(f)[0]
            f_path = os.path.join(DIR_LISTINGS, f)
            if os.path.isfile(f_path):
                listing_files[ex_name] = f_path
            else:
                print('Invalid path for {} at {}'.format(ex_name, f_path))

    if listing_files:
        print('Found listing files for {} exchanges: {}'.format(
            len(listing_files), ', '.join(list(listing_files.keys()))))
    return listing_files
FILES_LISTINGS = compile_listings()


# Create the folder to store all basic financial data
if not os.path.isdir(DIR_FINANCIALS):
    os.makedirs(DIR_FINANCIALS)

# Create the folder to store all options data
if not os.path.isdir(DIR_OPTIONS):
    os.makedirs(DIR_OPTIONS)


# URLs to format later
URL_YAHOO = 'https://finance.yahoo.com/quote/%s'
URL_YAHOO_FINANCIALS = URL_YAHOO + '/financials?'
URL_YAHOO_PROFILE = URL_YAHOO + '/profile?'
URL_YAHOO_PERFORMANCE = URL_YAHOO + '/performance?'
URL_YAHOO_OPTIONS = URL_YAHOO + '/options?'
URL_YAHOO_DAILY = URL_YAHOO + '/history?period1=%d&period2=%d&interval=1d&filter=history&frequency=1d'
URL_ALPHA_VANTAGE_INTRADAY = 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&' \
                             'symbol={symbol}&interval={interval}&outputsize=full&apikey={apikey}'


# Alpha Vantage user API key
if 'alpha_vantage' in redtide_configs and 'api_key_file' in redtide_configs['alpha_vantage'] \
        and os.path.isfile(redtide_configs['alpha_vantage'].get('api_key_file', '')):
    ALPHA_VANTAGE_API_KEY = open(redtide_configs['alpha_vantage']['api_key_file'], 'r').read().strip()
else:
    ALPHA_VANTAGE_API_KEY = None


# eoddata.com authentication file
# username (line 1), pw (line 2)
URL_EODDATA = 'http://eoddata.com/symbols.aspx'
URL_EODDATA_GET_SYMBOLS = 'http://eoddata.com/Data/symbollist.aspx?e=%s'
EODDATA_EXCHANGES = []
FILE_EODDATA_AUTH = None
if 'eoddata' in redtide_configs:
    eoddata_configs = redtide_configs['eoddata']
    if 'auth_file' in eoddata_configs and os.path.isfile(eoddata_configs['auth_file']):
        FILE_EODDATA_AUTH = eoddata_configs['auth_file']
    if 'exchanges' in eoddata_configs and eoddata_configs['exchanges']:
        EODDATA_EXCHANGES = eoddata_configs['exchanges']

# Robinhood authentication file
# username (line 1), pw (line 2)
FILE_ROBINHOOD_AUTH = None
if 'robinhood' in redtide_configs \
        and 'auth_file' in redtide_configs['robinhood'] \
        and os.path.isfile(redtide_configs['robinhood']['auth_file']):
    FILE_ROBINHOOD_AUTH = redtide_configs['robinhood']['auth_file']
