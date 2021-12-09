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
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM


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


print("Hello")


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





# Creating a test and data set
data = stockDataFrame.filter(["Close"]) # Creating a new dataframe with the close column
dataset = data.values # Converts the dataframe to a numpy array


trainingDataLength = int(np.ceil( len(dataset) * .95)) # Gets the number of rows to train the model on
# print(trainingDataLength)



# Creating the scaled data set
scaler = MinMaxScaler(feature_range=(0,1))
scaledData = scaler.fit_transform(dataset)
# print(scaledData)


# Creating the training data set
trainingData = scaledData[0:int(trainingDataLength), :]


# Splits the training data into two seperate data sets
xTrainingData = []
yTrainingData = []


for i in range (60, len(trainingData)): # This for loop fills the x and y training data sets
    xTrainingData.append(trainingData[i - 60:i, 0])
    yTrainingData.append(trainingData[i, 0])

    # if i <= 61:
    #     print(xTrainingData)
    #     print(yTrainingData)
    #     print(" ")


# Converting the x training data and the y training data to numpy arrays
xTrainingData = np.array(xTrainingData)
yTrainingData = np.array(yTrainingData)


xTrainingData = np.reshape(xTrainingData, (xTrainingData.shape[0], xTrainingData.shape[1], 1))





# Building the LSTM (Long Short Term Memory) Model
lstm_Model = Sequential() # provides training and inference features on this model.
lstm_Model.add(LSTM(128, return_sequences=True, input_shape=(xTrainingData.shape[1], 1))) # model.add Adds a layer instance on top of the layer stack
lstm_Model.add(LSTM(64, return_sequences=False))
lstm_Model.add(Dense(25))
lstm_Model.add(Dense(1))


# Compiling the LSTM model
lstm_Model.compile(optimizer="Adam", loss="mse") # configs the model with losses and metrics

# Trains the lstm Model
lstm_Model.fit(xTrainingData, yTrainingData, batch_size=1, epochs=1) # Give the model 45 or so seconds to run



# Creates new testing data set
testData = scaledData[trainingDataLength - 60: , : ]


# Splits the training data into two seperate data sets
xTestData = []
yTestData = dataset[trainingDataLength:, :]


for i in range(60, len(testData)): # Adds data to x Testing data
    xTestData.append(testData[i - 60:i, 0])



xTestData = np.array(xTestData)# Converts the testing data to a numpy array


xTestData = np.reshape(xTestData, (xTestData.shape[0], xTestData.shape[1], 1)) # Reshapes the x testing data



# Gets the LSTM model's predicted price values
predictions = lstm_Model.predict(xTestData)
predictions = scaler.inverse_transform(predictions)


# Calculates the root mean squared error
rootMeanSquaredError = np.sqrt(np.mean(((predictions - yTestData) ** 2)))
print('rmse ', rootMeanSquaredError)



# Calculates the mean absolute percentage error
meanAbsolutePercentageError = np.mean(np.abs((yTestData - predictions) / yTestData)) * 100
print('mape ', meanAbsolutePercentageError)





pd.options.mode.chained_assignment = None

# Plot the data
training = data[:trainingDataLength]
validation = data[trainingDataLength:]
validation["Predictions"] = predictions

plt.figure(figsize=(13,4))
plt.title("Stock Price Forecast LSTM Model")
plt.xlabel("Date")
plt.ylabel("Closing Price (USD)")
plt.plot(training["Close"])
plt.plot(validation[["Close", "Predictions"]])
plt.legend(["Training Data", "Validation", "Predictions"])
plt.show()




# Visualize the data 
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
