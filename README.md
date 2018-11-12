# Redtide
My attempt to model stock...

**Python 3.5+**

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
 - Navigate to the **redtide/src/** folder and run the following command
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
  - Navigate to **redtide/src/** folder, and run:
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

## Contact
I don't respond very quickly, but I always respond: jiunyyen@gmail.com
