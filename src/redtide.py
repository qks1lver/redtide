#!/usr/bin/env python3

# Import
import argparse
from classes import Stock, Regressor

# Run
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Redtide is coming')
    parser.add_argument('-l', dest='live_quote', action='store_true', help='Retrieve live quotes')
    parser.add_argument('-d', dest='daily_history', action='store_true', help='Retrieve historic data (daily resolution)')
    parser.add_argument('--symb', dest='symbol', default='', help='Specific ticker symbol')
    parser.add_argument('-a', dest='analyze', action='store_true', help='Analyze')
    parser.add_argument('-c', dest='compile', action='store_true', help='Compile symbols')
    parser.add_argument('-v', dest='verbose', action='store_true', help='Verbose')
    parser.add_argument('--file', dest='list', default='', help='Symbol file')
    parser.add_argument('--clust', dest='cluster', action='store_true', help='Cluster')
    parser.add_argument('--from', dest='from_date', default='2018-1-1', help='Analyze data from this date')
    parser.add_argument('--to', dest='to_date', default='', help='Analyze data until this date')
    parser.add_argument('--now', dest='live_now', action='store_true', help='Get live now, even if market is closed')
    parser.add_argument('--debug', dest='debug', action='store_true', help='Debug current updates')

    args = parser.parse_args()

    s = Stock()
    s.verbose = args.verbose
    s.live_now = args.live_now

    if args.live_quote:
        s.get_live_quote()

    if args.daily_history:

        if args.symbol:
            symb,success = s.retrieve_symb(args.symbol)
            print('Symb: %s (%d)' % (symb, success))
        elif args.list:
            s.retrieve_all_symbs(p_symbs=args.list)
        else:
            s.retrieve_all_symbs()

    if args.cluster:
        s.read_full_histories()
        X, symbs = s.range_norm(from_date=args.from_date, to_date=args.to_date)
        s.cluster(X, symbs)

    if args.compile:
        s.compile_symbols()

    if args.analyze:
        s.read_full_histories()
        s.analyze(from_date=args.from_date, to_date=args.to_date)

    if args.debug:
        symbs = ['NVDA']
        s.read_full_histories(symbs=symbs)
        f,l = s.gen_features(symbs=symbs, n_sampling=50, n_project=1)
        m = Regressor()
        m.train(f,l)
