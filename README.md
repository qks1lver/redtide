# Redtide
My attempt to model stock...

**Python 3.5+**

Notes about updates at the bottom... also shout out to helpful Redditor **u/siem**

## Intro
If you are looking at this, chances are you were on Reddit. This is
currently documented very very poorly. You have been warned. Hopefully,
you just want to pull stock histories yourself, and if so, just scroll
down and follow the steps. If you really want to know what else it does
(i.e. retrieve live stock prices, dimensionality reduction,
transformation, all that ML stuff) You will probably find much better
resources else where. But, of course, you are more than welcome to look
through the code. For now, I'm a bit too busy and streassed out by work
to document anything thoroughly, so... have fun!

## Dependencies
- Pandas
- Numpy
- PyTorch
- Scikit-learn
- Seaborn (if you decided to play around with the other stuff)

**Only Pandas is really involved in scraping data, but since I import
these at the top, you might want to have these so you don't get errors**

## Pull stock histories
#### 1. Update AMEX, NYSE, and NASDAQ listing files.
 - This is not a
mission critical step, but it's a good idea to do to prevent getting too
excited about a symbol that's already delisted or have repeated rows of
the same data because it's delisted (the latter can be solved by more
careful programming though).
 - Register and download AMEX.txt, NYSE.txt,
and NASDAQ.txt from http://eoddata.com/symbols.aspx (click the little
down arrow icon next to "DOWNLOAD SYMBOL LIST")
 - Replace the current AMEX.txt, NYSE.txt, and NASDAQ.txt files under
 **redtide/data/** folder with the new files
#### 2. Compile listing symbols.
**ONLY if you updated the listing files in step 1, if not, skip this step**
 - Delete **all_symbols.txt** and **excluded_symbols.txt** under the
 **redtide/data/** folder
 - Navigate to the **redtide/** folder and run the following command
 in a terminal.
 ```
 $ python3 redtide.py -v -c
 ```
 - This will check each symbol in the 3 exchange listing files to see if
 their data can be pulled from Yahoo Finance. If yes, they are compiled
 to all_symbols.txt. If not, they are compiled to excluded_symbols.txt
 - When it's done, you'll see the new **all_symbols.txt** and
 **excluded_symbols.txt** under the **redtide/data/** folder
#### 3. Pull data.
  - Navigate to **redtide/** folder, and run:
  ```
  $ python3 redtide.py -v -d
  ```
  - This will take a long time. How long depends on the **number of
  processors** you have and your **internet connection speed**.
  - When this is done, you will see a **full_history/** folder under
  **redtide/data/** that contains all the goodies (i.e. AMD.csv).

## Other stuff
If you want to know some other options:
```
$ python3 redtide.py -h
```

## Updates
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

## Contact
I don't respond very quickly, but I always respond: jiunyyen@gmail.com
