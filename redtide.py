#!/usr/bin/env python3

# Import
import argparse
from src.scraper import Stock
from src.tradebot import TradeBot

# Run
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Redtide is coming')
    parser.add_argument('-d', dest='daily_history', action='store_true', help='Retrieve historic data (daily resolution)')
    parser.add_argument('--symb', dest='symbol', default='', help='Specific ticker symbol')
    parser.add_argument('-c', dest='compile', action='store_true', help='Compile symbols')
    parser.add_argument('-v', dest='verbose', action='store_true', help='Verbose')
    parser.add_argument('--concat', dest='concat', default=None, help='Concatenate data into one file, give path')
    parser.add_argument('--file', dest='list', default=None, help='Symbol file')
    parser.add_argument('--from', dest='from_date', default=None, help='Analyze data from this date')
    parser.add_argument('--to', dest='to_date', default=None, help='Analyze data until this date')
    parser.add_argument('--bot', dest='bot', action='store_true', help='Run trade bot')
    parser.add_argument('--budget', dest='budget', default=500, type=int, help='Max budget for trade bot')
    parser.add_argument('--stocks', dest='stocks', default=5, type=int, help='Max stocks to hold by trade bot')
    parser.add_argument('--maxloss', dest='maxloss', default=0.9, type=float, help='Max loss by trade bot')

    args = parser.parse_args()

    # Compile stock symbols
    # Basically build a list of symbols that's acceptable with Yahoo Finance's URL
    if args.compile:
        s = Stock(verbose=args.verbose)
        if args.list:
            s.compile_symbols(p_symbs=args.list, append=True, batch_size=40)
        else:
            s.compile_symbols()

    # Scrape either daily history or live quotes on Yahoo Finance
    # Live quote is so slow, so bad, I hate it, I build it, I apologize for it
    # But daily history is what Redtide is really for
    if args.daily_history:
        s = Stock(mode='full_history', verbose=args.verbose)
        if args.symbol:
            symb,success = s.pull_daily_and_write(args.symbol)
            print('Symb: %s (%d)' % (symb, success))
        elif args.list:
            s.pull_daily_and_write_batch(p_symbs=args.list)
        else:
            s.pull_daily_and_write_batch()

    # To concatenate individual stock historic data files into one
    # It'd take a long long time to concatenate everything, and not recommended
    # use --from and --to to select range (no --to means anything between --from to now)
    # i.e. $ python redtide -v --concat out_file.csv --from 2019-01-10
    if args.concat:
        s = Stock(verbose=args.verbose)
        p_out = args.concat + '.csv' if not args.concat.endswith('.csv') else args.concat
        _ = s.concat(from_date=args.from_date, to_date=args.to_date, p_out=p_out)
        print('Saved to: %s' % p_out)

    # Run Robinhood tradebot
    if args.bot:
        bot = TradeBot('robinhood', n_splits=args.stocks,
                       allowance=args.budget, max_loss=args.maxloss)
        bot.run()
