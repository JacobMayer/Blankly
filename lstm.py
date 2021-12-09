# LSTM stands for Long Short-Term Memory


import matplotlib
import pandas as pd
import numpy as np
import yfinance as yf
import datetime
import time
import requests
import io
import matplotlib.pyplot as plt
# import matplotlib.pyldab as rcParams
import seaborn as sns
# import tensorflow
from pandas_datareader.data import DataReader


# Streamlit Imports
import streamlit as st
from bokeh.plotting import figure


# Sotck data that will be pulled from Yahoo Finance
ticker = "TSLA" # Ticker of stock
startDate = datetime.datetime(2015, 1, 1) # Start date of when data will pulled
endDate = datetime.datetime.today() # End date of when data will be pulled




stockDataFrame = pd.DataFrame() # This stock data frame is what will hold stock data (after we append stock data to it). You can print the stock data to the screen if you want to see what is looks like


# Downloads the stocks price from yahoo finance
stock = [] 
stock = yf.download(ticker, start=startDate, end=endDate, progress=False) 

stockDataFrame = stockDataFrame.append(stock ,sort=False) # Adds the stock price to the data frame (sort if false because we are not trying to sort any of the stocks data, we just want to put it in the data frame)
stockDataFrame['Symbol'] = ticker





sns.set_style("whitegrid") # Sets style for plotting

movingAverage = [10, 30, 60] # Moving average of stock for 10, 30 & 60 days


for ma in movingAverage: # Will add all three moving averages to the new data frame
    columnName = f"MA For {ma} Days"
    stockDataFrame2 = stockDataFrame
    stockDataFrame2[columnName] = stockDataFrame2["Adj Close"].rolling(ma).mean() # Calculcates current moving average and makes a new column for it



stockDataFrame["Daily Return"] = stockDataFrame["Adj Close"].pct_change()
sns.displot(stockDataFrame["Daily Return"].dropna(), bins=100, color="goldenrod") # Shows whether or not the daily return is evenly distributed
# plt.title("Daily Return")
# plt.show()

# Mat plot lib plot
# plt.figure(figsize=(13,4))
# plt.plot(stockDataFrame["Close"], label="Closing Price History") 
# plt.plot(stockDataFrame["MA For 10 Days"], label="10 Day Moving Average") 
# plt.plot(stockDataFrame["MA For 30 Days"], label="30 Day Moving Average") 
# plt.plot(stockDataFrame["MA For 60 Days"], label="60 Day Moving Average") 
# plt.title("Tesla Closing Price and Moving Average") 
# plt.xlabel("Time") 
# plt.ylabel("Price") 


# plt.style.use("fivethirtyeight")
# plt.legend()
# plt.show()
