#!/usr/bin/env python3

# Import
import argparse
from classes import Stock

# Run
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Redtide is coming')
    parser.add_argument('-l', dest='live_quote', action='store_true', help='Retrieve live quotes')
    parser.add_argument('-d', dest='daily_history', action='store_true', help='Retrieve historic data (daily resolution)')
    parser.add_argument('--symb', dest='symbol', action='store', default='', help='Specific ticker symbol')
    parser.add_argument('-a', dest='analyze', action='store_true', help='Analyze')
    parser.add_argument('-v', dest='verbose', action='store_true', help='Verbose')

    args = parser.parse_args()

    s = Stock()
    s.verbose = args.verbose

    if args.live_quote:
        s.get_live_quote()

    if args.daily_history:

        if args.symbol:
            symb,success = s.retrieve_symb(args.symbol)
            print('Symb: %s (%d)' % (symb, success))
        else:
            s.retrieve_all_symbs()

    if args.analyze:
        dfs = s.read_full_histories()
        # s.analyze(dfs)
        s.cluster(dfs, from_date='2018-6-1')
