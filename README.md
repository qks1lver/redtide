# Redtide v0.2
My attempt to scrape stock data, analyze the market, ... and tradebot

**Python 3.5+**

**( 2020.06.07 ) For those who already used Redtide before.**
1. Thank you!
2. I pretty much rebuilt it from the ground up and SQUASHED ALL the
commits. **Pull with caution!**

#### Main changes
1. **Live-quote option is deprecated!** It sucked, pretty sure no one
used it. Pulling live-quote still exists as a functionality of the new
`Stock` data model. More on this below.
2. Streamlined the stock symbol compilation step by automatically
logging into **eoddata.com** (with your auth using Selenium)
3. Scrape **company's financial data** during the symbol compilation
step
4. Introducing HoodAPI that uses Selenium to automate trades on
Robinhood, example:
    ```
    from src.api import HoodAPI
    hood = HoodAPI()
    hood.make_order('buy', 'AMD', shares=5, order_type='limit', price=52)
    hood.order_status('AMD')
    ```
5. An experimental **trade bot** option (`--bot`), but go through it
carefully first (at least read the brief description below). Run this
**after compiling symbols and financial data:**
    ```
    $ python3 redtide.py --bot --budget 500 --stocks 5 --maxloss 0.9
    ```
6. The `tzset` issue on Windows is resolved.
Everything **should** work on Windows. I tested on Windows 10.
7. All that ML stuff is removed because they are utterly useless! I
honestly believe that everyone is modeling it better than I am. T_T

## Intro
If you are looking at this, chances are you were on Reddit. This is
currently documented very very poorly. You have been warned. Hopefully,
you just want **a simple way to grab historic data** (end-of-day data)
from Yahoo Finance to do your own awesome analysis.
Or if you want to **venture into automated day-trade**. Then Redtide may
be a solution for you.
For now, I'm a bit too busy and stressed out by work
to document anything thoroughly, so... have fun!

## Install / Setup
#### A. Download code base
1. Just clone this repository. You'll want to do everything
in the **redtide/** folder (I'll make a wheel at some point)
    - I highly recommend creating and doing everything in a virtual
    environment, i.e.
        ```
        $ cd redtide/
        $ python3 -m venv .
        $ source bin/activate
        ```
2. Use the **requirements.txt** file to make sure you have
everything you'll need. Run this:
```
$ pip3 install -r requirements.txt
```

#### B. Setup Selenium
If you are not familiar with Selenium, it is basically a we browser
instance that you can control in Python. Generally used in website
test automation, but I'm using it for difficult logins and navigating
complex websites.

Skip this if you already have Firefox (gecko webdriver) in PATH. As in,
you know running `webdriver.Firefox()` without any arguments will work.

**Steps (assume you already installed the Python reqs):**
1. Download gecko webdriver from here (scroll down to Assets):
https://github.com/mozilla/geckodriver/releases
2. Unpack and put the executable anywhere you want it
3. In `config.yaml`, point `"path_geckodriver"` to the path of the
gecko executable (the file you just unpacked), example:
    ```
    selenium:
        path_geckodriver: "where/you/put/geckodriver"
    ```

#### C. Setup authentications
1. **[eoddata.com](http://eoddata.com/symbols.aspx) auth**
    - This is so Redtide knows what stock symbols are out there
    - Register a free account on
    **[eoddata.com](http://eoddata.com/symbols.aspx)**
    - Create a text file (whatever name you want, anywhere you want)
    - In the text file, line 1: username, line 2: password; example:
        ```
        admin
        password
        ```
    - In **config.yaml** point the `auth_file` path under `eoddata` to
    the path of the login file you just created
    - **Alternatively,** if you are concerned with security, you can
    just log in and download the symbol list of each exchange manually
    and put them in a folder named "**listings**" under the folder
    **data/**. You'll have to create the **listings/** folder but
    Redtide will look in there first. But note that you'll want to
    repeat this everytime you pull data since stocks can get delisted
    and enlisted, and at the moment I don't know if there's a source for
    "diffs" on stock listings.
    - **Note: I don't work for eoddata.com, just find it to be
 a good source**
2. **Robinhood auth** (Optional, if you want to use **HoodAPI**)
    - Same steps as eoddata.com auth, create a username-password file
    but path pointed for `auth_file` under `robinhood` in
    **config.yaml**

#### You are now ready to go!

## Pull stock histories

### How this works
1. **"stock exchange files,"** which are just lists of stock
symbols tell Redtide which symbols to look for data
on Yahoo Fiance. Redtide check with Yahoo to make sure
these symbols can be found and verify spelling (i.e. "-" or ".").
2. While checking the symbols, Redtide also grabs the **financial data**
of each company as it compiles the symbols.
3. User gets a chance to examine the compiled list of stock
symbols to make sure they are good to go, or go straight the pulling
historical data by stacking command line options.
4. Redtide pulls the **End-of-Day** movements all these stocks
as far back as Yahoo provides.

### Steps to run

#### 1. Auto-update and compile stock exchange listings
 - Navigate to the **redtide/** folder and run the following command
 in a terminal.
 ```
 $ python3 redtide.py -v -c
 ```

 - Compilation can take an hour or so depending on the **number of
 processors** and your **internet speed**.
 - When it's done, you'll see the new **all_symbols.txt** and
 **excluded_symbols.txt** under the **redtide/data/** folder

 **Pro tip** If you have a list of stocks that you care about, and don't
 care about any other tickers. Then create your own **all_symbols.txt**
 and keep it in **redtide/data/** and skip this step (`-c`) entirely
 until you want to update their financial data.

#### 2. Pull data

  - Navigate to **redtide/** folder, and run:
  ```
  $ python3 redtide.py -v -d
  ```
  - This will about an hour or so. How long depends on the **number of
  processors** you have and your **internet connection speed**.
  - When this is done, you will see a **full_history/** folder under
  **redtide/data/** that contains all the goodies (i.e. AMD.csv).

  **Pro tip** You can chain both steps by doing:
  ```
  $ python3 redtide.py -v -c -d
  ```

## Extra stuff

If you want to see some other options:
```
$ python3 redtide.py -h
```

### HoodAPI (Robinhood trade automation)
At the moment, Robinhood does not have an official API.
There are a few unofficial Robhinhood APIs but it seems that they are
either not maintained anymore or "working with Robinhood LLC" to stay
online. Meaning there's a dependency on Robinhood. This can also mean
that they are more stable but also mean Robinhood can have control
over it's access (i.e. daily call limits, is it profitable, etc.)
So... **HoodAPI** is built using Selenium with the intention of having
full control but not as reliable (i.e. if Robinhood changes layout,
HoodAPI will need to be updated with how to navigate).

**Note: will need to setup Robinhood auth in the setup step.

There are 2 basic concepts:
1. **Action** Making buy/sell order is a type of action. Cancel order is
another type of action. **Only market order and limit order for now**,
but the code to switch any order type is done, just not the form-filling
part for the other types.
2. **Verify Status** Check the status of the action you requested to
make sure it was successful.

```
from src.api import HoodAPI
hood = HoodAPI()
hood.make_order('buy', 'AMD', shares=5)
status = hood.order_status('AMD')
```

If order went through then `status['status']` would be `"Done"`.
Otherwise, it can be another Robinhood status message, like
`"Pending"`.
It is **important** to understand that `order_status(s)` checks your
most recent order of that stock (i.e. the first element in the order
history for that stock).
If status is not `"Done"`, then you can cancel it with

```
hood.cancel_order('AMD')
```

Here also, `cancel_order(s)` only tries to cancel your most recent order
of that stock.

When you are done, call `quit()` to exit safely. This is to prevent a
lingering Firefox session in the background.
```
hood.quit()
```

**Must be very mindful of the current caveat of `check_status(s)` and
`cancel_order(s)` as you implement, which, again, is that they only
act on your most recent order of the specified stock**

### Stock and Stocks classes
The goal here is to track stock prices, volumes, bids, asks, etc. of
each stock. A `Stock('AMD')` object would retrieve and store the latest
data on AMD from Yahoo when you call `.update()`. A `Stocks([...])`
object lets you track multiple tickers via multiprocessing.

#### Stock
A **Stock()** object can be instantiated for any symbol found on
Yahoo.
When `auto_update=True` (default), then update is automatically
done as the Stock object is instantiated. Thus, you can check the
fields like price and volume. `auto_update=True` would also enforce
update when certain fields are called, but to prevent update being
too frequence, a staleness limit on the data (i.e. 15 seconds default)
is used so that update is only performed if data is stale.

```
from src.models import Stock
from time import sleep

s = Stock('AMD', cache_stale_sec=15)
print(s.price)
print(s.volume)
print(s.bid)
print(s.bid_size)

sleep(15)
print(s.price)  # update before returning price
```

A bunch of metrics are track during each `.update()` call, but not
all of them are as useful as other, therefore, don't have a dedicated
field for them. But all data can be access under `s.metrics` dict. Any
of the following metrics that also have a dedicated field would also
perform auto-update if `auto_update=True`.

| Yahoo metrics | Stock obj metrics | has field |
|---|---|---|
| regularMarketPrice | price | yes |
| regularMarketVolume | volume | yes |
| bid | bid | yes |
| bidSize | bid_size | yes |
| ask | ask | yes |
| askSize | ask_size | yes |
| currency | currency | yes |
| regularMarketChange | change | no |
| regularMarketDayHigh | high | no |
| regularMarketDayLow | low | no |
| sharesOutstanding | shares | no |
| marketCap | market_cap | no |
| regularMarketPreviousClose | previous_close | no |
| regularMarketOpen | open | no |
| averageVolume10days | volumne10days | no |

During each update(), these metrics are also written to
`data/live_quotes/<symbol> (default path) as lines of JSON (note: not
a JSON file!) Each line in the file can be interpreted with a JSON
interpreter. The sole reason that this is so weird is because all this
is still quite experimental, and I want a structure that's as easily
accessible as it is dynamic.

**s.price_trend(k=5)** returns a Scipy linregress result object on the
linear regression fit of the last 5 price data points (5 `update()`
calls are required)

**s.volume_trend(k=5)** returns a Scipy linregress result object on the
linear regression fit of the last 5 volume data points (5 `update()`
calls are required)

#### Stocks
**Stocks()** object manages multiple `Stock()` objects. Example:
```
from src.models import Stocks
from time import sleep

ss = Stocks(['AMD', 'AAPL', 'TSLA']).update()
ss.add_symbols(['UAL', 'DAL', 'AAL'])
ss.remove('TSLA')

# update with 5 sec intervals
for _ in range(5):
    sleep(5)
    ss.update()

# ask "what is the price trend of all the stocks I'm tracking?"
# this does a T-test with the R^2 values of all the stocks against
# zeros. Not the best way to do this statistics, but less data
# required to get a rough estimate.
ss.price_trend(k=5, metric='rvalue')
```


### Trade bot (very very experimental)
It's so experimental, I haven't made a single cent with it yet! Trading
works, but I'm new to day-trade and I'm not sure what's a good strategy
here. The trade bot basically utilizes the **HoodAPI** to automate
trades. Feel free to look into to the code **src/tradebot.py** and I
sincerely hope you are more successful than I am.

To run with default ($500 budget, at at most 5 stocks, 90% budget max
loss)
```
$ python3 redtide.py --bot
```

Or more controls
```
$ python3 redtide.py --bot --budget 1000 --stocks 10 --maxloss 0.95
```

May the Force be with you...
s
## AlphaVantage
**I want to exploit this great resource via crowd sourcing.**
They limit free users to 5 requests per minute and 500
requests per day. There are about 8600 tickers in AMEX, NYSE,
and NASDAQ. Their API lets user get intraday data up to a week
so it'll take **18 users less than 2 hours** to collect intraday
data each EOD. Code for this is in progress. There are
much more we can do by joining force. Let me know if this
is interesting to you. :)

## Issues and workarounds

- **u/siem** found a problem with how Macs handles forking. His full comment:

"I had problems running the code at first on my Mac
("python +\[__NSPlaceholderDate initialize] may have been in progress
in another thread when fork() was called.") -
supposedly Apple has changed their fork implementation to disallow forking with active threads.
When I ran this before running the code I didn't get errors:

```export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES```"

- **u/siem** also found that after 2000-3000 pulls,
connection to Yahoo Finance could fail.
Possibly due to a tempoary IP ban.
After that, I also noticed that around 120 fast page crawls, there's a temporary IP ban.
To circumvent this issue, I implemented pauses around 10 - 20 seconds for every 100 page loads.
Also, did the same for compiling the symbols, except the pauses are 5 - 10 seconds for every 200 page loads.
- **New:** If connection fails or you get banned temporary, it will try
to fetch for the failed ones again (**maximum of 5 passes**). If after 5 passes
, there are still failed symbols left, they
will be written to a **failed_symbs-<5 random characters>.txt** file.
And you'll see a suggestion to run something like the following to retry.
```
# For failed daily history fetches
$ python3 redtide.py -v -d --file failed_symbs-abe93.txt

# For failed symbols during symbol compile
$ python3 redtide.py -v -c --file data/excluded_symbols.txt
```

## Shout-outs
- Helpful Redditor **u/siem** discovered ways to resolve forking issue on Mac and IP ban issue with Yahoo.
- Big thanks to **John Ecker** (JECKER@rollins.edu) and **Lukáš Vokráčko** (vokracko) for help make Redtide better with debugging and better documentations! Much appreciated!

## TODO
Gosh... Where do I even begin... email me if you are interested.

## Contact
Let me know what you think, and if you want to help out on my odd
endeavor to be less poor.
I don't respond very quickly, but I always try to respond:
jiunyyen@gmail.com
