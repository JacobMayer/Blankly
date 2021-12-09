# LSTM stands for Long Short-Term Memory

import pandas as pd
import numpy as np
import yfinance as yf
import datetime
import time
import requests
import io
# import matplotlib.pylot as plt
# import matplotlib.pyldab as rcParams
# import seaborn as sns
# import tensorflow
from pandas_datareader.data import DataReader


# Sotck data that will be pulled from Yahoo Finance
ticker = "TSLA" # Ticker of stock
startDate = datetime.datetime(2015, 1, 1) # Start date of when data will pulled
endDate = datetime.datetime.today() # End date of when data will be pulled




stockDataFrame = pd.DataFrame() # Creates an empty data frame


# Downloads the stocks price from yahoo finance
stock = [] 
stock = yf.download(ticker, start=startDate, end=endDate, progress=False) 

stockDataFrame = stockDataFrame.append(stock ,sort=False) # Adds the stock price to the data frame (sort if false because we are not trying to sort any of the stocks data, we just want to put it in the data frame)
stockDataFrame['Symbol'] = ticker


print(stockDataFrame)