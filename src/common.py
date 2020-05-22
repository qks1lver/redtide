#!/usr/bin/env python3

import re
import requests
from numpy import ceil
from random import choices
from string import ascii_letters
from datetime import datetime, timedelta
from dateutil import parser


regxs = {
    'currency': re.compile(r'[$,]'),
    'rh_shares': re.compile(r'Shares* ([0-9,]*)'),
    'rh_port_stocks': re.compile(r'([A-Z]+)\n([0-9,]+)\s+Share'),
}


def gen_rand_name(k=6):
    return ''.join(choices(ascii_letters, k=k))


def float_money(v):
    return float(regxs['currency'].sub('', v))


def str_money(v, decimal=2, comma=False):
    if comma:
        return ("{:,.%df}" % decimal).format(v)
    return ("{:.%df}" % decimal).format(v)


def gen_symbol_batches(symbs, n_batches=None, batch_size=None):
    """
    For multiprocessing.
    :param symbs:
    :param n_batches:
    :param batch_size:
    :return:
    """

    if not batch_size:
        if not n_batches:
            n_batches = 20
        n_symbs = len(symbs)
        batch_size = ceil(n_symbs / n_batches)

    symb_batches = []
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


def get_wallstreet_time(open_time=(9,30), close_time=(16,0), offset_close=(0, 0)):
    res = {
        'is_market_open': False,
        'datetime_str': requests.get('http://worldtimeapi.org/api/timezone/'
                                     'America/New_York').json()['datetime']
    }
    dt_ny = parser.parse(res['datetime_str'])
    dt_start = datetime(year=dt_ny.year, month=dt_ny.month, day=dt_ny.day,
                        hour=open_time[0], minute=open_time[1])
    dt_end = datetime(year=dt_ny.year, month=dt_ny.month, day=dt_ny.day,
                      hour=close_time[0], minute=close_time[1]) \
             - timedelta(hours=offset_close[0], minutes=offset_close[1])
    dt_ny = datetime(year=dt_ny.year, month=dt_ny.month, day=dt_ny.day,
                     hour=dt_ny.hour, minute=dt_ny.minute, second=dt_ny.second)
    res['datetime'] = dt_ny
    weekday = dt_ny.weekday()
    if weekday < 5 and dt_start <= dt_ny < dt_end:
            res['is_market_open'] = True
            res['open_in'] = 0
            res['close_in'] = (dt_end - dt_ny).total_seconds()
    else:
        if dt_ny < dt_start:
            add_days = 0 if weekday < 5 else (7 - weekday)
        else:
            add_days = 1 if weekday < 4 else (7 - weekday)
        dt_start = dt_start + timedelta(days=add_days)
        res['open_in'] = (dt_start - dt_ny).total_seconds()
        dt_end = dt_end + timedelta(days=add_days)
        res['close_in'] = (dt_end - dt_ny).total_seconds()
    return res
