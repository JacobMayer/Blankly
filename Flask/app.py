from flask import Flask, redirect, url_for, render_template, request, send_from_directory, make_response
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
    return render_template("pages-faq.html", user=getUserFromToken(token))

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
@app.route("/")
def home():
    token = request.cookies.get('token')
    if (os.path.isfile('users.json')):
        user = getUserFromToken(token)
        if user:
            return render_template("index.html", user=getUserFromToken(token))
    return redirect("/login")

#Profile page
@app.route("/profile")
def profilePage():
    token = request.cookies.get('token')
    if (os.path.isfile('users.json')):
        user = getUserFromToken(token)
        if user:
            return render_template("users-profile.html", user=getUserFromToken(token))
    return redirect("/login")

#404 Page
@app.route("/404")
def errorPage():
    token = request.cookies.get('token')
    if (os.path.isfile('users.json')):
        user = getUserFromToken(token)
        if user:
            return render_template("pages-error-404.html", user=getUserFromToken(token))
    return redirect("/login")

#Contact Page
@app.route("/contact")
def contactPage():
    token = request.cookies.get('token')
    if (os.path.isfile('users.json')):
        user = getUserFromToken(token)
        if user:
            return render_template("pages-contact.html", user=getUserFromToken(token))
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

                    ran_bool = 1
                    # creating an init allows us to run the same function for
                    # different tickers and resolutions
                    s.add_price_event(price_event, ticker, resolution='1d', init=init)

                    variable = s.backtest('2y', {'USD': 10000})
                    global history
                    metrics = variable.get_metrics()
                    metrics = { x.translate({32:None}) : y
                         for x, y in metrics.items()}
                    return render_template('pages-models.html', metrics=metrics, ran_bool=ran_bool, strategy='XGBOOST')
                if request.form['submit_button1'] == 'submit_it2':
                    ticker = request.form['textinfo2']
                    ran_bool = 1

                    # creating an init allows us to run the same function for
                    # different tickers and resolutions
                    s.add_price_event(price_event, ticker, resolution='1d', init=init)

                    variable = s.backtest('2y', {'USD': 10000})
                    #variable.figures[2]
                    global history
                    metrics = variable.get_metrics()
                    metrics = { x.translate({32:None}) : y
                         for x, y in metrics.items()}
                    return render_template('pages-models.html', metrics=metrics, ran_bool=ran_bool, strategy='RSI')
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
                  return user['username']
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
            "password" : request.form.get('password')
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

########## Token login Ends ##########

########## Blankly Dependencies Begins ##########

def init(symbol, state: StrategyState):
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

def price_event(price, symbol, state: StrategyState):
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



if __name__ == "__main__":
    app.run(debug=True)
