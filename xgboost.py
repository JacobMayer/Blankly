import os
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV

import plotly as py
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# put data into a dataframe for easy data reading and manipulation
df = pd.read_csv('/home/nicholas/Downloads/bitcoin.csv')

df['Date'] = pd.to_datetime(df['Date'])
df = df[(df['Date'].dt.year >= 2020)].copy()


df['Close'] = df['Close'].shift(-1)

df = df.iloc[33:] 
df = df[:-1]      

df.index = range(len(df))

test_size  = 0.15
valid_size = 0.15

#splits the data into train, validation, and testing
test_split_idx  = int(df.shape[0] * (1-test_size))
valid_split_idx = int(df.shape[0] * (1-(valid_size+test_size)))

train_df  = df.loc[:valid_split_idx].copy()
valid_df  = df.loc[valid_split_idx+1:test_split_idx].copy()
test_df   = df.loc[test_split_idx+1:].copy()


#uncomment this if you want to see the split of the data in train/validation/testing
#fig = go.Figure()
#fig.add_trace(go.Scatter(x=train_df.Date, y=train_df.Close, name='Training'))
#fig.add_trace(go.Scatter(x=valid_df.Date, y=valid_df.Close, name='Validation'))
#fig.add_trace(go.Scatter(x=test_df.Date,  y=test_df.Close,  name='Test'))
#fig.show()

#collumns to drop to get the target values of 'Close']
drop_cols = ['Date', 'Volume', 'Open', 'Low', 'High', 'Market Cap']

train_df = train_df.drop(drop_cols, 1)
valid_df = valid_df.drop(drop_cols, 1)
test_df  = test_df.drop(drop_cols, 1)

y_train = train_df['Close'].copy()
X_train = train_df.drop(['Close'], 1)

y_valid = valid_df['Close'].copy()
X_valid = valid_df.drop(['Close'], 1)

y_test  = test_df['Close'].copy()
X_test  = test_df.drop(['Close'], 1)


eval_set = [(X_train, y_train), (X_valid, y_valid)]

#creates the xgboost model and trains it on the evaluation set 
model = xgb.XGBRegressor(eval_set=eval_set, objective='reg:squarederror', verbose=False)
model.fit(X_train, y_train, eval_set=eval_set, verbose=False)

#model predicts values using the testing dataset
y_pred = model.predict(X_test)
print(f'y_true = {np.array(y_test)[:5]}')
print(f'y_pred = {y_pred[:5]}')

predicted_prices = df.loc[test_split_idx+1:].copy()
predicted_prices['Close'] = y_pred

print(f'mean_squared_error = {mean_squared_error(y_test, y_pred)}')

#plots the final prediction trend lines
fig = make_subplots(rows=2, cols=1)
fig.add_trace(go.Scatter(x=df.Date, y=df.Close,
                         name='Truth',
                         marker_color='LightSkyBlue'), row=1, col=1)

fig.add_trace(go.Scatter(x=predicted_prices.Date,
                         y=predicted_prices.Close,
                         name='Prediction',
                         marker_color='MediumPurple'), row=1, col=1)

fig.add_trace(go.Scatter(x=predicted_prices.Date,
                         y=y_test,
                         name='Truth',
                         marker_color='LightSkyBlue',
                         showlegend=False), row=2, col=1)

fig.add_trace(go.Scatter(x=predicted_prices.Date,
                         y=y_pred,
                         name='Prediction',
                         marker_color='MediumPurple',
                         showlegend=False), row=2, col=1)

fig.show()
