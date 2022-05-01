from sqlite3 import TimeFromTicks
from flask import Flask, redirect, url_for, render_template, request, send_from_directory, make_response, send_file

#from flask import Flask, redirect, url_for, render_template, request, send_from_directory, make_response
import json
import numpy as np
import os
from uuid import uuid4
import io
import shutil

########## Blankly Begin ##########
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import blankly
#alpaca = blankly.Alpaca()
from blankly import trunc
from blankly import Strategy, StrategyState, Interface
from blankly import CoinbasePro, Alpaca
from blankly.indicators import rsi, sma
from bokeh.embed import components
########## Blankly End ##########
from datetime import datetime
import plotly as py
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

########## Machine Learning Begins ##########

# You may need to "pip install scikit-learn" if you do not have this installed
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import xgboost
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV

########## Machine Learning End ##########
#mainpush

app = Flask(__name__,  static_url_path='')

global history
global has_bought_xgboost 
global has_bought_mlp 

########## Assets Begin ##########

@app.route('/assets/<path:path>')
def asset(path):
    print(path)
    return send_from_directory('assets', path)

########## Assets End ##########

########## Pages Begin ##########



#Faq page
@app.route("/faq")
def faq():
    token = request.cookies.get('token')
    return render_template("pages-faq.html", user=getUserFromToken(token)['username'])

#Login page
@app.route("/login", methods=['GET', 'POST'])
def login():
        if request.method == 'POST':

            if not (os.path.isfile('users.json')):
                return redirect("/register")

            username = request.form.get('username')
            password = request.form.get('password')

            data = {}
            with open("users.json", "r") as read_file:
                data = json.load(read_file)


            for user in data['accounts']:
                if user['username'] == username and user['password'] == password:
                       resp = make_response(redirect('/'))
                       rand_token = str(uuid4())
                       resp.set_cookie('token', rand_token)
                       user['token'] = rand_token
                       with open("users.json", "w") as out_file:
                            json.dump(data, out_file)
                       return resp

        return render_template("pages-login.html")

#Home page, dashboard page
@app.route("/", methods=['GET', 'POST'])
def home():
    token = request.cookies.get('token')
    if (os.path.isfile('users.json')):
        user = getUserFromToken(token)
        if user:
            updateRevenue()

            user = getUserFromToken(token) #get most recent data

            tickerInfo = []
            tickercount = 0
            for ticker, amount in user['tickers'].items():
                tickercount += amount
                tickerInfo.append({"value" : amount, "name" : ticker})

            if request.method == 'POST':
                generateAccountReport(user)
                print("hj")
                return send_file(user['username'] + "_report.csv", as_attachment=True)

            return render_template("index.html", user=user['username'], 
                revenue=user['revenue'],
                tickers=tickerInfo,
                tickerCount=tickercount,
                trades=user['trades'],
                percentChange=user['percentChange'],
                tradeHistory=user['tradeHistory'])
    return redirect("/login")

@app.route("/index.html")
def home2():
    return redirect("/")

#Profile page
@app.route("/profile")
def profilePage():
    token = request.cookies.get('token')
    if (os.path.isfile('users.json')):
        user = getUserFromToken(token)
        if user:
            return render_template("users-profile.html", user=getUserFromToken(token)['username'])
    return redirect("/login")

#404 Page
@app.route("/404")
def errorPage():
    token = request.cookies.get('token')
    if (os.path.isfile('users.json')):
        user = getUserFromToken(token)
        if user:
            return render_template("pages-error-404.html", user=getUserFromToken(token)['username'])
    return redirect("/login")

#Contact Page
@app.route("/contact")
def contactPage():
    token = request.cookies.get('token')
    if (os.path.isfile('users.json')):
        user = getUserFromToken(token)
        if user:
            return render_template("pages-contact.html", user=getUserFromToken(token)['username'])
    return redirect("/login")

@app.route("/buy-sell", methods=['GET', 'POST'])
def buysell():
    token = request.cookies.get('token')
    if (os.path.isfile('users.json')):
        user = getUserFromToken(token)
        if user:
            if request.method == 'POST':
                if (request.form.get('tickerNameBuy') != ''):
                    buyTicker(request.form.get('tickerNameBuy'), request.form.get('tickerAmountBuy'))

                if (request.form.get('tickerNameSell') != ''):
                    sellTicker(request.form.get('tickerNameSell'), request.form.get('tickerAmountSell'))

            user = getUserFromToken(token)
            tickerInfo = []
            for ticker, amount in user['tickers'].items():
                tickerInfo.append({"value" : amount, "name" : ticker})
            return render_template("pages-buy-sell.html", user=getUserFromToken(token)['username'], tickers=tickerInfo)
    return redirect("/login")


#Models Page
@app.route('/models', methods=['GET', 'POST'])
def trading_models():
    ran_bool = 0
    ran_mlp_bool = 0
    prophet_ranbool = 0
    ran_comparison = 0
    alpaca = Alpaca()
    s = Strategy(alpaca)
    csv_for_ml = ""
    token = request.cookies.get('token')
    if (os.path.isfile('users.json')):
        user = getUserFromToken(token)
        if user:
            if request.method == "POST":
            #ticker = request.form.get('nick')
                if request.form['submit_button1'] == 'submit_it':
                    ticker = request.form['textinfo']
                    amount = int(request.form['totalamount'])
                    resolution = request.form['resolution']
                    backtest = request.form['backtest']
                    ran_bool = 1
                    # creating an init allows us to run the same function for
                    # different tickers and resolutions
                    s.add_price_event(price_event_xgboost, ticker, resolution=resolution, init=init_xgboost)

                    
                    variable = s.backtest(backtest, {'USD': amount})


                    filenames = find_csv_filenames("price_caches/")
    
                    for name in filenames:
                        if name.find(ticker) != -1:
                            csv_for_ml = name
                   # print(csv_for_ml)
                    

                    csv_dir = "price_caches/" + csv_for_ml
                    target_dir = "assets/img/"
                    csv_target_dir = target_dir + csv_for_ml


                    shutil.copyfile("/home/nicholas/TheCapstone/Blankly/Flask/" + csv_dir, "/home/nicholas/TheCapstone/Blankly/Flask/" + csv_target_dir)


                    #variable.figures[0]
                    #script, div = components(variable.figures[0])
                    #script2, div2 = components(variable.figures[1])
                    #script3, div3 = components(variable.figures[2])
                    print(variable.figures)
                    script, div = components(variable.figures)
                    global history
                    #print (script)
                    metrics = variable.get_metrics()
                    global has_bought_xgboost
                    if has_bought_xgboost:
                        str_bought = 'True'
                    else:
                        str_bought = 'False'
                    print("--------------------------------" + str_bought + '---------------------------------------')
                    metrics = { x.translate({32:None}) : y
                         for x, y in metrics.items()}
                    return render_template('pages-models.html', user=user['username'],ran_mlp_bool=ran_mlp_bool, csv_dir=csv_target_dir, ran_comparison=ran_comparison, prophet_ranbool=prophet_ranbool, has_bought=has_bought_xgboost, script=script, div=div, metrics=metrics, ran_bool=ran_bool, strategy='XGBOOST')
                if request.form['submit_button1'] == 'submit_it2':
                    ticker = request.form['textinfo2']
                    amount = int(request.form['totalamount2'])
                    resolution  = request.form['resolution2']
                    backtest = request.form['backtest2']
                    ran_bool = 1
                    ran_mlp_bool = 0
    
                    
                    # different tickers and resolutions
                    
                    s.add_price_event(price_event_mlp, ticker, resolution=resolution, init=init_mlp)

                    variable = s.backtest(backtest, {'USD': amount})
                    filenames = find_csv_filenames("price_caches/")

                    for name in filenames:
                        if name.find(ticker) != -1:
                            csv_for_ml = name
                   # print(csv_for_ml)
                    

                    csv_dir = "price_caches/" + csv_for_ml
                    target_dir = "assets/img/"
                    csv_target_dir = target_dir + csv_for_ml


                    shutil.copyfile("/home/nicholas/TheCapstone/Blankly/Flask/" + csv_dir, "/home/nicholas/TheCapstone/Blankly/Flask/" + csv_target_dir)
                    # creating an init allows us to run the same function for

                    #use_prophet(ticker)
                    global has_bought_mlp
                    if has_bought_mlp:
                        
                        str_bought = 'True'
                    else:
                        str_bought = 'False'
                    print("--------------------------------" + str_bought + '---------------------------------------')
                    script, div = components(variable.figures)
                    #variable.figures[2]
                    global history
                    metrics = variable.get_metrics()
                    metrics = { x.translate({32:None}) : y
                         for x, y in metrics.items()}
                    return render_template('pages-models.html', user=user['username'],ran_mlp_bool=ran_mlp_bool, csv_dir=csv_target_dir, ran_comparison=ran_comparison, prophet_ranbool=prophet_ranbool, has_bought=has_bought_mlp, script=script, div=div, metrics=metrics, ran_bool=ran_bool, strategy='MLP')

                if request.form['submit_button1'] == 'submit_it3':
                    ticker = request.form['textinfo3']
                    amount = int(request.form['totalamount3'])
                    resolution  = request.form['resolution3']
                    backtest = request.form['backtest3']
                    ran_comparison = 1
                    
                    ran_bool = 0
                    prophet_ranbool = 0

                    
                    s.add_price_event(price_event_mlp, ticker, resolution=resolution, init=init_mlp)
                    
                    variable_mlp = s.backtest(backtest, {'USD': amount})

                    s.add_price_event(price_event_xgboost, ticker, resolution=resolution, init=init_xgboost)

                    variable_xgboost = s.backtest(backtest, {'USD': amount})  

                    filenames = find_csv_filenames("price_caches/")
    
                    for name in filenames:
                        if name.find(ticker) != -1:
                            csv_for_ml = name
                    print(csv_for_ml)
                    

                    csv_dir = "price_caches/" + csv_for_ml
                    target_dir = "assets/img/"
                    csv_target_dir = target_dir + csv_for_ml


                    shutil.copyfile("/home/nicholas/TheCapstone/Blankly/Flask/" + csv_dir, "/home/nicholas/TheCapstone/Blankly/Flask/" + csv_target_dir)


                    script1, div1 = components(variable_mlp.figures)
                    script2, div2 = components(variable_xgboost.figures)

                    return render_template('pages-models.html', user=user['username'],ran_mlp_bool=ran_mlp_bool, csv_dir=csv_target_dir, ran_comparison=ran_comparison, prophet_ranbool=prophet_ranbool, script1=script1, div1=div1, script2=script2, div2=div2, ran_bool=ran_bool, strategy='Comparison')
 

                if request.form['submit_button1'] == 'submit_it4':
                    ticker = request.form['textinfo4']
                    ran_bool = 0
                    prophet_ranbool = 1
                    s.add_price_event(price_event_mlp, ticker, resolution='1d', init=init_mlp)
                    
                    variable = s.backtest('1y', {'USD': 1000})

                    

                    from fbprophet import Prophet



                    filenames = find_csv_filenames("price_caches/")
                    csv_for_ml = ''
                    for name in filenames:
                        if name.find(ticker) != -1:
                            csv_for_ml = name
                        

                        
                    new_db = pd.read_csv('price_caches/' + csv_for_ml)
                    #drop_cols = ['volume', 'open', 'low', 'high']
                    print(new_db)
                    for i in range(len(new_db['time'])):
                        new_db['time'][i] = datetime.datetime.fromtimestamp(new_db['time'][i]).strftime("%B %d, %Y")
                    print(new_db)
                    df = new_db[["time","close"]] 
                # Rename the features: These names are required for the model fitting
                    df = new_db.rename(columns = {"time":"ds","close":"y"}) 
                    #train_vals = db.drop(drop_cols, 1)
                    #print(train_vals)
                    #db = db.rename(columns = {"time":"ds","close":"y"}) 
                    
                    #split_idx = int(db.shape[0])

                    #train_df  = db.loc[:split_idx].copy()
                    #train_vals = train_df.drop(drop_cols, 1)
                    

                    # The Prophet class (model)
                    fbp = Prophet(daily_seasonality = True) 
                    # Fit the model 
                    fbp.fit(df)
                    # We need to specify the number of days in future
                    # We'll be predicting the full 2021 stock prices
                    fut = fbp.make_future_dataframe(periods=100) 
                    forecast = fbp.predict(fut)

                    from fbprophet.plot import plot_plotly, plot_components_plotly
                    fig1 = fbp.plot(forecast, xlabel='Date', ylabel='Price')

                    #fig = use_prophet(ticker)
                    if os.path.isfile('assets/img/new_plot.png'):
                        os.remove("assets/img/new_plot.png") 
                    plt.savefig('assets/img/new_plot.png')
                    

                    return render_template('pages-models.html', user=user['username'],ran_mlp_bool=ran_mlp_bool, ran_comparison=ran_comparison, url = 'assets/img/new_plot.png', ran_bool = ran_bool, prophet_ranbool=prophet_ranbool, ticker=ticker)


                    


            return render_template('pages-models.html', user=user['username'],ran_bool=ran_bool)
    return redirect("/login")

########## Pages End ##########

########## Token login Begins ##########

#user login token helper
def getUserFromToken(token):
    if (os.path.isfile('users.json')):
        with open("users.json", "r") as read_file:
            data = json.load(read_file)
            for user in data['accounts']:
             if ('token' in user.keys()):
              if token == user['token'] and token != "":
                  return user
    else:
        return "Sign in"

#signout button
@app.route('/signout')
def logout():
    token = request.cookies.get('token')
    with open("users.json", "r") as read_file:
        data = json.load(read_file)
        for user in data['accounts']:
         if ('token' in user.keys()):
          if token == user['token']:
            resp = make_response(redirect('/login'))
            resp.set_cookie('token', "")
            return resp

#register button
@app.route("/register",  methods=['GET', 'POST'])
def register():
    message = ''

    if request.method == 'POST':
        user = {
            "username" : request.form.get('username'),
            "email" : request.form.get('email'),
            "password" : request.form.get('password'),
            "revenue" : "100000",
            "tickers" : {"AAPL" : 55, "TSLA": 64},
            "tradeHistory" : {},
            "trades" : 0,
            "percentChange" : 1
        }

        if (os.path.isfile('users.json')):
         database = json.load(open("users.json", "r"))
         with open("users.json", "r") as outfile:
            for users in database['accounts']:
                if (users['username'] == user['username']):
                    return
         os.remove("users.json")
         database['accounts'].append(user)
         with open("users.json", "w") as outfile:
            json.dump(database, outfile)

            return redirect("/login")

        else:
            with open("users.json", "w") as outfile:
                database = {"accounts" : [user]}
                json.dump(database, outfile)
            return redirect("/login")

    return render_template('pages-register.html', message=message)


@app.route('/reports')
def reports():
    print("sadasd")
    token = request.cookies.get('token')
    user = getUserFromToken(token)
    generateAccountReport(user)

    return send_file("Reports/" + user['username'] + "_Report.csv", as_attachment=True)



########## Token login Ends ##########

########## Blankly Dependencies Begins ##########

def init_xgboost(symbol, state: StrategyState):
    interface: Interface = state.interface
    resolution = state.resolution
    variables = state.variables
    #x, y = make_classification(n_samples=500, n_features=3, n_informative=3, n_redundant=0, random_state=1)
    #x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, random_state=1)
    # initialize the historical data
    #variables['model'] = xgboost.XGBClassifier().fit(x_train, y_train)
    #variables['model'] = MLPClassifier(random_state=1, max_iter=10000).fit(x_train, y_train)
    variables['history'] = interface.history(symbol, 300, resolution, return_as='list')['close']
    variables['high'] = interface.history(symbol, 300, resolution, return_as='list')['high']
    variables['close'] = interface.history(symbol, 300, resolution, return_as='list')['close']

    variables['low'] = interface.history(symbol, 300, resolution, return_as='list')['low']

    variables['volume'] = interface.history(symbol, 300, resolution, return_as='list')['volume']

    variables['open'] = interface.history(symbol, 300, resolution, return_as='list')['open']
    variables['time'] = interface.history(symbol, 300, resolution, return_as='list')['time']

    dataDic = {'time': variables['time'], 'close': variables['close'], 'high' : variables['high'], 'low' : variables['low'], 'volume' : variables['volume'], 'open' : variables['open']}

    db = pd.DataFrame(dataDic)
    db['RSI'] = calc_rsi_forRegression(db).fillna(0)
    db['SMA_50'] = db['close'].rolling(50).mean().shift()
    db['SMA_100'] = db['close'].rolling(100).mean().shift()
    scaler = MinMaxScaler()
    #print(db)
    #b['RSI'] = db['RSI'].reshape(-1,1)
    #db['RSI'] = scaler.fit_transform(db['RSI'])
    print(db)
    #data = db.values[]
    # perform a robust scaler transform of the dataset
    trans = MinMaxScaler()
    data = trans.fit_transform(db)
    # convert the array back to a dataframe
    db = pd.DataFrame(data)
    # summarize
    print(db)
    
    #print(db)

    #0- high, 1- close, 2-low 3-volume 4-time, 5-open
    drop_cols = [0,1, 2, 3, 4, 5]
    #y = db[1].copy
    x = db.drop(drop_cols, 1)

    ind = list(db.columns).index(1)
    y = []
    for i in range(db.shape[0]-1):
        if (db[1][i+1]-db[1][i])>0:
            y.append(1)
        else:
            y.append(0)
    y.append(0)
    len(y)
    print(x)
    len(db[0])

    print(y)
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)



    test_size  = 0.15
    valid_size = 0.15
    #print(db.time)
    test_split_idx  = int(db.shape[0] * (1-test_size))
    valid_split_idx = int(db.shape[0] * (1-(valid_size+test_size)))

    train_df  = db.loc[:valid_split_idx].copy()
    valid_df  = db.loc[valid_split_idx+1:test_split_idx].copy()
    test_df   = db.loc[test_split_idx+1:].copy()

    
    #print(train_df)
    #train_df = train_df.drop(drop_cols, 1)
    ##valid_df = valid_df.drop(drop_cols, 1)
    #test_df  = test_df.drop(drop_cols, 1)

    #y_train = train_df['close'].copy()
    #X_train = train_df.drop(['close'], 1)

    #y_valid = valid_df['close'].copy()
    #X_valid = valid_df.drop(['close'], 1)

    #y_test  = test_df['close'].copy()
    #X_test  = test_df.drop(['close'], 1)
    eval_set = [(x_train, y_train), (x_test, y_test)]
    print(y_train)
    model = xgboost.XGBClassifier( n_estimators = 300, learning_rate = 0.01, max_depth = 10, gamma = 0.001, eval_set=eval_set ,objective='binary:logistic', verbose=False)
    #clf = GridSearchCV(model, parameters)
    model.fit(x_train, y_train, eval_set=eval_set, verbose=False)
    
    variables['model'] = model

    
    variables['has_bought'] = False

def price_event_xgboost(price, symbol, state: StrategyState):
    interface: Interface = state.interface
    variables = state.variables

    variables['history'].append(price)
    model = variables['model']
    scaler = MinMaxScaler()
    rsi_values = rsi(variables['history'], period=14).reshape(-1, 1)
    rsi_value = scaler.fit_transform(rsi_values)[-1]

    ma_values = sma(variables['history'], period=50).reshape(-1, 1)
    ma_value = scaler.fit_transform(ma_values)[-1]

    ma100_values = sma(variables['history'], period=100).reshape(-1, 1)
    ma100_value = scaler.fit_transform(ma100_values)[-1]
    value = np.array([rsi_value, ma_value, ma100_value]).reshape(1, 3)
    prediction = model.predict_proba(value)[0][1]
    #print(value - prediction)
    #print(prediction)
    # comparing prev diff with current diff will show a cross
    if prediction > 0.4 and not variables['has_bought']:
        interface.market_order(symbol, 'buy', trunc(interface.cash/price, 8))
        variables['has_bought'] = True
    elif prediction <= 0.4 and variables['has_bought']:
        # truncate is required due to float precision
        interface.market_order(symbol, 'sell', interface.account[state.base_asset]['available'])
        variables['has_bought'] = False
    
    global has_bought_xgboost
    has_bought_xgboost = variables['has_bought']

from os import listdir

def find_csv_filenames( path_to_dir, suffix=".csv" ):
    filenames = listdir(path_to_dir)
    return [ filename for filename in filenames if filename.endswith( suffix ) ]

def use_prophet(symbol):
    from fbprophet import Prophet
    
    filenames = find_csv_filenames("price_caches/")
    
    for name in filenames:
        if name.find(symbol) != -1:
            csv_for_ml = name
        

        
    new_db = pd.read_csv('price_caches/' + csv_for_ml)
    #drop_cols = ['volume', 'open', 'low', 'high']
    #print(db)
    for i in range(len(new_db['time'])):
        new_db['time'][i] = datetime.datetime.fromtimestamp(new_db['time'][i]).strftime("%B %d, %Y")
    print(new_db)
    df = new_db[["time","close"]] 
# Rename the features: These names are required for the model fitting
    df = new_db.rename(columns = {"time":"ds","close":"y"}) 
    #train_vals = db.drop(drop_cols, 1)
    #print(train_vals)
    #db = db.rename(columns = {"time":"ds","close":"y"}) 
    
    #split_idx = int(db.shape[0])

    #train_df  = db.loc[:split_idx].copy()
    #train_vals = train_df.drop(drop_cols, 1)
    

    # The Prophet class (model)
    fbp = Prophet(daily_seasonality = True) 
    # Fit the model 
    fbp.fit(df)
    # We need to specify the number of days in future
    # We'll be predicting the full 2021 stock prices
    fut = fbp.make_future_dataframe(periods=100) 
    forecast = fbp.predict(fut)

    from fbprophet.plot import plot_plotly, plot_components_plotly
    fig1 = fbp.plot(forecast)
    return fig1
'''
    # A better plot than the simple matplotlib
    print(forecast)
    fig = make_subplots(rows=2, cols=1)
    fig.add_trace(go.Scatter(x=forecast.ds, y=forecast.yhat,
                         name='Truth',
                         marker_color='LightSkyBlue'), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=forecast.ds,
                         y=forecast.trend,
                         name='Prediction',
                         marker_color='MediumPurple'), row=1, col=1)                     
    fig.show()
    '''
    

def init_mlp(symbol, state: StrategyState):
    interface: Interface = state.interface
    resolution = state.resolution
    variables = state.variables
    x, y = make_classification(n_samples=500, n_features=3, n_informative=3, n_redundant=0, random_state=1)
    #print(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, random_state=1)
    # initialize the historical data
    #variables['model'] = xgboost.XGBClassifier().fit(x_train, y_train)
    variables['model'] = MLPClassifier(random_state=1, max_iter=10000).fit(x_train, y_train)
    new_model = xgboost.XGBRegressor()
    variables['history'] = interface.history(symbol, 3000, resolution, return_as='list')['close']
    variables['high'] = interface.history(symbol, 3000, resolution, return_as='list')['high']
    variables['close'] = interface.history(symbol, 3000, resolution, return_as='list')['close']

    variables['low'] = interface.history(symbol, 3000, resolution, return_as='list')['low']

    variables['volume'] = interface.history(symbol, 3000, resolution, return_as='list')['volume']

    variables['open'] = interface.history(symbol, 3000, resolution, return_as='list')['open']
    variables['time'] = interface.history(symbol, 3000, resolution, return_as='list')['time']
    filenames = find_csv_filenames("price_caches/")
    for name in filenames:
        if name.find(symbol) != -1:
            csv_for_ml = name
        
    '''
        
    db = pd.read_csv('price_caches/' + csv_for_ml)
    dataDic = {'time': variables['time'], 'close': variables['close'], 'high' : variables['high'], 'low' : variables['low'], 'volume' : variables['volume'], 'open' : variables['open']}
    #db = pd.DataFrame(dataDic)
    for i in range(len(db['time'])):
        db['time'][i] = datetime.datetime.fromtimestamp(db['time'][i]).strftime("%B %d, %Y")
    df_close = db[['time', 'close']].copy()
    df_close = df_close.set_index('time')
    df_close.head()
    #print(db)
    db['RSI'] = calc_rsi_forRegression(db).fillna(0)
    #rint(db['RSI'])
    #creating macd signals
    EMA_12 = pd.Series(db['close'].ewm(span=12, min_periods=12).mean())
    EMA_26 = pd.Series(db['close'].ewm(span=26, min_periods=26).mean())
    db['MACD'] = pd.Series(EMA_12 - EMA_26).fillna(0)
    db['MACD_signal'] = pd.Series(db.MACD.ewm(span=9, min_periods=9).mean()).fillna(0)

    db['EMA_9'] = db['close'].ewm(9).mean().shift()
    db['SMA_5'] = db['close'].rolling(5).mean().shift()
    db['SMA_10'] = db['close'].rolling(10).mean().shift()
    db['SMA_15'] = db['close'].rolling(15).mean().shift()
    db['SMA_30'] = db['close'].rolling(30).mean().shift()

    #print(db)
    old_time = db['time']
    db['close'] = db['close'].shift(-1)
    db = db.iloc[33:] # Because of moving averages and MACD line
    db = db[:-1]      # Because of shifting close price

    db.index = range(len(db))

    test_size  = 0.15
    valid_size = 0.15
    #print(db.time)
    test_split_idx  = int(db.shape[0] * (1-test_size))
    valid_split_idx = int(db.shape[0] * (1-(valid_size+test_size)))

    train_df  = db.loc[:valid_split_idx].copy()
    valid_df  = db.loc[valid_split_idx+1:test_split_idx].copy()
    test_df   = db.loc[test_split_idx+1:].copy()

    drop_cols = ['volume', 'open', 'low', 'high', 'time']
    print(train_df)
    train_df = train_df.drop(drop_cols, 1)
    valid_df = valid_df.drop(drop_cols, 1)
    test_df  = test_df.drop(drop_cols, 1)

    y_train = train_df['SMA_30'].copy()
    X_train = train_df.drop(['SMA_30'], 1)

    y_valid = valid_df['SMA_30'].copy()
    X_valid = valid_df.drop(['SMA_30'], 1)

    y_test  = test_df['SMA_30'].copy()
    X_test  = test_df.drop(['SMA_30'], 1)
    print(x_train)
    parameters = {
    'n_estimators': [100, 200, 300, 400],
    'learning_rate': [0.001, 0.005, 0.01, 0.05],
    'max_depth': [8, 10, 12, 15],
    'gamma': [0.001, 0.005, 0.01, 0.02],
    'random_state': [42]
    }

    eval_set = [(X_train, y_train), (X_valid, y_valid)]
    model = xgboost.XGBRegressor( n_estimators = 300, learning_rate = 0.01, max_depth = 10, gamma = 0.001, eval_set=eval_set ,objective='reg:squarederror', verbose=False)
    #clf = GridSearchCV(model, parameters)
    model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
    #xgboost.plot_importance(model)
    #model.summary()
    y_pred = model.predict(X_test)
    #y_predNone = model.predict()
    #print(y_pred)
    #print(train_df['close'])
    #print(f'y_true = {np.array(y_test)[:5]}')
    #print(f'y_pred = {y_pred[:5]}')
    #print(f'mean_squared_error = {mean_squared_error(y_test, y_pred)}')
    #print(train_df)
    #print(variables['high'])
    #print(variables['history'])


    predicted_prices = db.loc[test_split_idx+1:].copy()
    predicted_prices['SMA_30'] = y_pred
    #predicted_prices[]
    #date_1 = datetime.datetime.strptime(start_date, "%m/%d/%y")
    #list_append = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    #predicted_prices['time'].append(list_append)
    print(predicted_prices['time'])
    #end_date = date_1 + datetime.timedelta(days=10)
    print(len(predicted_prices['close']))
    
    print(db['time'].iloc[-1])
    print(predicted_prices['close'].iloc[-1])
    print(predicted_prices['time'].iloc[-1])
    
    fig = make_subplots(rows=2, cols=1)
    fig.add_trace(go.Scatter(x=db.time, y=db.SMA_30,
                         name='Truth',
                         marker_color='LightSkyBlue'), row=1, col=1)
    print(db.time)
    fig.add_trace(go.Scatter(x=predicted_prices.time,
                         y=predicted_prices.SMA_30,
                         name='Prediction',
                         marker_color='MediumPurple'), row=1, col=1)

    fig.add_trace(go.Scatter(x=predicted_prices.time,
                         y=y_test,
                         name='Truth',
                         marker_color='LightSkyBlue',
                         showlegend=False), row=2, col=1)

    fig.add_trace(go.Scatter(x=predicted_prices.time,
                         y=y_pred,
                         name='Prediction',
                         marker_color='MediumPurple',
                         showlegend=False), row=2, col=1)

    #fig.show()



    dti = pd.date_range("2022-04-20 00:30:00", periods=1000, freq="H")
    df_future_dates = pd.DataFrame(dti, columns = ['time'])
    
    
    df_future_dates['RSI'] = np.nan
    df_future_dates['MACD'] = np.nan
    df_future_dates['MACD_signal'] = np.nan
    df_future_dates['EMA_9'] = np.nan
    df_future_dates['SMA_5'] = np.nan
    df_future_dates['SMA_10'] = np.nan
    df_future_dates['SMA_15'] = np.nan
    df_future_dates['SMA_30'] = np.nan
    df_future_dates['time'] = pd.to_datetime(df_future_dates['time'], format='%Y-%m-%d %H:%M:%S')
    date = df_future_dates['time']
    df_future_dates = df_future_dates.drop(['time'], 1)


    

    
    df_future_dates_copy = df_future_dates.copy()
    #testX_future, testY_future = create_features(df_future_dates, target_variable='Irr')
    
    testX_future = df_future_dates
    #testY_future = df_future_dates.drop(drop_cols, 1)


    #xgb = xgboost.XGBRegressor(objective= 'reg:linear', n_estimators=1000)
    

    ## Now here I have used train and test from above
    

    predicted_results_future = model.predict(testX_future)
    import matplotlib.pyplot as plt
    fig1 = make_subplots(rows=1, cols=1)
    fig1.add_trace(go.Scatter(x=date, y=predicted_results_future,
                         name='predicted',
                         marker_color='LightSkyBlue'), row=1, col=1)
    #fig1.show()

    # Graph 
    plt.figure(figsize=(13,8))
    plt.plot(list(predicted_results_future))
    plt.title("Predicted")
    plt.ylabel("Price")
    plt.legend(('predicted'))
   # plt.show()
    '''
    
    variables['has_bought'] = False

def price_event_mlp(price, symbol, state: StrategyState):
    global has_bought_mlp
    interface: Interface = state.interface
    variables = state.variables
    #Creating a pandas dataframe from all the values 
    
    






    variables['history'].append(price)
    model = variables['model']
    scaler = MinMaxScaler()
    #print(variables['history'])
    rsi_values = rsi(variables['history'], period=14).reshape(-1, 1)
    #print(rsi_values)
    rsi_value = scaler.fit_transform(rsi_values)[-1]
    #print(model)
    ma_values = sma(variables['history'], period=50).reshape(-1, 1)
    ma_value = scaler.fit_transform(ma_values)[-1]

    ma100_values = sma(variables['history'], period=100).reshape(-1, 1)
    ma100_value = scaler.fit_transform(ma100_values)[-1]
    value = np.array([rsi_value, ma_value, ma100_value]).reshape(1, 3)
    prediction = model.predict_proba(value)[0][1]
    #print(value)
    #print(prediction)
    # comparing prev diff with current diff will show a cross
    if prediction > 0.4 and not variables['has_bought']:
        interface.market_order(symbol, 'buy', trunc(interface.cash/price, 8))
        variables['has_bought'] = True
    elif prediction <= 0.4 and variables['has_bought']:
        # truncate is required due to float precision
        interface.market_order(symbol, 'sell', interface.account[state.base_asset]['available'])
        variables['has_bought'] = False

   
    has_bought_mlp = variables['has_bought']


def calc_rsi_forRegression(dataframe):
    n = 14
    close = dataframe['close']
    #delta = close.diff()
    delta = close.diff()
    delta = delta[1:]
    pricesUp = delta.copy()
    pricesDown = delta.copy()
    pricesUp[pricesUp < 0] = 0
    pricesDown[pricesDown > 0] = 0
    rollUp = pricesUp.rolling(n).mean()
    rollDown = pricesDown.abs().rolling(n).mean()
    rs = rollUp / rollDown
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi
########## Blankly Dependencies Ends ##########

#import yfinance as yf
import requests
import time
import datetime
import random

from bs4 import BeautifulSoup

def stock_price(symbol):
    url = f"https://finance.yahoo.com/quote/{symbol}/"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    class_ = "My(6px) Pos(r) smartphone_Mt(6px) W(100%)"
    return soup.find("div", class_=class_).find("fin-streamer").text

def updateRevenue():
    if (os.path.isfile('users.json')):
        token = request.cookies.get('token')
        #load database
        data = {}
        with open("users.json", "r") as read_file:
            data = json.load(read_file)
        user = getUserFromToken(token)

        newRevenue = 0
        for users in data['accounts']:
            if user == users:
                for ticker, amount in user['tickers'].items():
                    newRevenue += round(float(stock_price(ticker).replace(',', ""))  * int(amount), 2)

                users['percentChange'] = round((newRevenue - float(users['revenue']))/(float(users['revenue'])) * 100, 2)

                users['revenue'] = round(newRevenue, 2)

        with open("users.json", "w") as out_file:
            json.dump(data, out_file)

    return

def buyTicker(ticker, amount):
    token = request.cookies.get('token')
    if (os.path.isfile('users.json')):
        #load database
        data = {}
        with open("users.json", "r") as read_file:
            data = json.load(read_file)
        user = getUserFromToken(token)

        for users in data['accounts']:
            if user == users:
                if (ticker in user['tickers'].keys()):
                 users['tickers'][ticker] += int(amount)
                else:
                 users['tickers'][ticker] = int(amount)
                users['trades'] += 1
                now = datetime.datetime.now()
                dateString = str(now.month) + " " + str(now.day) + " " + str(now.year) + " " + str(now.hour) + ":" + str(now.minute) + ":" + str(now.second)
                price = round(float(stock_price(ticker).replace(',', ""))  * int(amount), 2)
                users['tradeHistory'][int(time.time())] = {"type" : "BUY", "amount" : amount, "ticker" : ticker, "date" : dateString, "price" : price, "number" : random.randint(0,10000)}
                

        with open("users.json", "w") as out_file:
            json.dump(data, out_file)
    return

def sellTicker(ticker, amount):
    token = request.cookies.get('token')
    if (os.path.isfile('users.json')):
        #load database
        data = {}
        with open("users.json", "r") as read_file:
            data = json.load(read_file)
        user = getUserFromToken(token)

        for users in data['accounts']:
            if user == users:
                if (ticker in user['tickers'].keys()):
                 if (int(amount) > int(users['tickers'][ticker])):
                    amount = users['tickers'][ticker]
                 users['tickers'][ticker] -= int(amount)
                 if (users['tickers'][ticker] <= 0):
                     users['tickers'].pop(ticker, None)
                users['trades'] += 1
                now = datetime.datetime.now()
                dateString = str(now.month) + " " + str(now.day) + " " + str(now.year) + " " + str(now.hour) + ":" + str(now.minute) + ":" + str(now.second)
                price = round(float(stock_price(ticker).replace(',', ""))  * int(amount), 2)
                users['tradeHistory'][int(time.time())] = {"type" : "SELL", "amount" : amount, "ticker" : ticker, "date" : dateString, "price" : price, "number" : random.randint(0,10000)}

        with open("users.json", "w") as out_file:
            json.dump(data, out_file)
    return

######### Stock buying and selling end #######


######### Reporting Analysis Begin ##########

def generateAccountReport(user):
    print("hello")
    with open("Reports/" + user['username'] + "_Report.csv", "w") as outfile:
        outfile.write("Current Net\n")
        outfile.write(str(user['revenue']) + "\n\n")

        outfile.write("Total Trades\n")
        outfile.write(str(user['trades']) + "\n\n")

        outfile.write("Owned Tickers, Amount\n")

        total = 0
        for ticker in user['tickers']:
            outfile.write(ticker + "," + str(user['tickers'][ticker]) + "\n")
            total += user['tickers'][ticker]
        outfile.write("\nTotal," + str(total) + "\n\n")

        outfile.write("Trade History\n")
        outfile.write("Trade #,Date, Algorithm, Ticker, Amount, Price, Buy/Sell\n")
        for trade in user['tradeHistory']:
            print(trade)
            outfile.write(str(user['tradeHistory'][trade]['number']) + "," + user['tradeHistory'][trade]['date'] + "," + "Manual" + "," + 
            user['tradeHistory'][trade]['ticker'] + "," + user['tradeHistory'][trade]['amount'] + "," + 
            str(user['tradeHistory'][trade]['price']) + "," + user['tradeHistory'][trade]['type'] + "\n")
        outfile.close()

    return

######### Reporting Analysis End ##########

if __name__ == "__main__":
    app.run(debug=True)
