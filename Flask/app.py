from sqlite3 import TimeFromTicks
from flask import Flask, redirect, url_for, render_template, request, send_from_directory, make_response, send_file
import json
import numpy as np
import os
from uuid import uuid4

########## Blankly Begin ##########

import blankly
alpaca = blankly.Alpaca()
from blankly import trunc
from blankly import Strategy, StrategyState, Interface
from blankly import CoinbasePro, Alpaca
from blankly.indicators import rsi, sma

########## Blankly End ##########

########## Machine Learning Begins ##########

# You may need to "pip install scikit-learn" if you do not have this installed
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import xgboost

########## Machine Learning End ##########


app = Flask(__name__,  static_url_path='')

global history


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
    alpaca = Alpaca()
    s = Strategy(alpaca)

    token = request.cookies.get('token')
    if (os.path.isfile('users.json')):
        user = getUserFromToken(token)
        if user:
            if request.method == "POST":
            #ticker = request.form.get('nick')
                if request.form['submit_button1'] == 'submit_it':
                    ticker = request.form['textinfo']
                    amount = int(request.form['totalamount'])

                    ran_bool = 1
                    # creating an init allows us to run the same function for
                    # different tickers and resolutions
                    s.add_price_event(price_event_xgboost, ticker, resolution='1d', init=init_xgboost)

                    variable = s.backtest('2y', {'USD': amount})
                    #variable.figures[0]
                    #script, div = components(variable.figures[0])
                    #script2, div2 = components(variable.figures[1])
                    #script3, div3 = components(variable.figures[2])
                    script, div = components(variable.figures)
                    global history
                    #print (script)
                    metrics = variable.get_metrics()
                    metrics = { x.translate({32:None}) : y
                         for x, y in metrics.items()}
                    return render_template('pages-models.html', script=script, div=div, metrics=metrics, ran_bool=ran_bool, strategy='XGBOOST')
                if request.form['submit_button1'] == 'submit_it2':
                    ticker = request.form['textinfo2']
                    amount = int(request.form['totalamount2'])
                    ran_bool = 1

                    # creating an init allows us to run the same function for
                    # different tickers and resolutions
                    s.add_price_event(price_event_mlp, ticker, resolution='1d', init=init_mlp)

                    variable = s.backtest('2y', {'USD': amount})
                    script, div = components(variable.figures)
                    #variable.figures[2]
                    global history
                    metrics = variable.get_metrics()
                    metrics = { x.translate({32:None}) : y
                         for x, y in metrics.items()}
                    return render_template('pages-models.html', script=script, div=div, metrics=metrics, ran_bool=ran_bool, strategy='MLP')
            return render_template('pages-models.html', ran_bool=ran_bool)
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
    x, y = make_classification(n_samples=500, n_features=3, n_informative=3, n_redundant=0, random_state=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, random_state=1)
    # initialize the historical data
    variables['model'] = xgboost.XGBClassifier().fit(x_train, y_train)
    #variables['model'] = MLPClassifier(random_state=1, max_iter=10000).fit(x_train, y_train)
    variables['history'] = interface.history(symbol, 300, resolution, return_as='list')['close']
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
    #print(prediction)
    # comparing prev diff with current diff will show a cross
    if prediction > 0.4 and not variables['has_bought']:
        interface.market_order(symbol, 'buy', trunc(interface.cash/price, 8))
        variables['has_bought'] = True
    elif prediction <= 0.4 and variables['has_bought']:
        # truncate is required due to float precision
        interface.market_order(symbol, 'sell', interface.account[state.base_asset]['available'])
        variables['has_bought'] = False

    global history
    history = variables['history']



def init_mlp(symbol, state: StrategyState):
    interface: Interface = state.interface
    resolution = state.resolution
    variables = state.variables
    x, y = make_classification(n_samples=500, n_features=3, n_informative=3, n_redundant=0, random_state=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, random_state=1)
    # initialize the historical data
    #variables['model'] = xgboost.XGBClassifier().fit(x_train, y_train)
    variables['model'] = MLPClassifier(random_state=1, max_iter=10000).fit(x_train, y_train)
    variables['history'] = interface.history(symbol, 300, resolution, return_as='list')['close']
    variables['has_bought'] = False

def price_event_mlp(price, symbol, state: StrategyState):
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
    #print(prediction)
    # comparing prev diff with current diff will show a cross
    if prediction > 0.4 and not variables['has_bought']:
        interface.market_order(symbol, 'buy', trunc(interface.cash/price, 8))
        variables['has_bought'] = True
    elif prediction <= 0.4 and variables['has_bought']:
        # truncate is required due to float precision
        interface.market_order(symbol, 'sell', interface.account[state.base_asset]['available'])
        variables['has_bought'] = False

    global history
    history = variables['history']

########## Blankly Dependencies Ends ##########



######### Stock buying and selling begin #######

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
                 if (amount > users['tickers'][ticker]):
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
