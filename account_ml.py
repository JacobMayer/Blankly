# main.py

from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.properties import ObjectProperty
from kivy.uix.popup import Popup
from kivy.uix.label import Label
from database import DataBase
from kivy.lang import Builder

#from asyncio.windows_events import NULL
import numpy as np
from blankly import trunc
from blankly import Strategy, StrategyState, Interface
from blankly import CoinbasePro, Alpaca
from blankly.indicators import rsi, sma
# You may need to "pip install scikit-learn" if you do not have this installed
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import xgboost

global history

Builder.load_string("""
<CreateAccountWindow>:
    name: "create"

    namee: namee
    email: email
    password: passw

    FloatLayout:
        cols:1

        FloatLayout:
            size: root.width, root.height/2

            Label:
                text: "Create an Account"
                size_hint: 0.8, 0.2
                pos_hint: {"x":0.1, "top":1}
                font_size: (root.width**2 + root.height**2) / 14**4

            Label:
                size_hint: 0.5,0.12
                pos_hint: {"x":0, "top":0.8}
                text: "Name: "
                font_size: (root.width**2 + root.height**2) / 14**4

            TextInput:
                pos_hint: {"x":0.5, "top":0.8}
                size_hint: 0.4, 0.12
                id: namee
                multiline: False
                font_size: (root.width**2 + root.height**2) / 14**4

            Label:
                size_hint: 0.5,0.12
                pos_hint: {"x":0, "top":0.8-0.13}
                text: "Email: "
                font_size: (root.width**2 + root.height**2) / 14**4

            TextInput:
                pos_hint: {"x":0.5, "top":0.8-0.13}
                size_hint: 0.4, 0.12
                id: email
                multiline: False
                font_size: (root.width**2 + root.height**2) / 14**4

            Label:
                size_hint: 0.5,0.12
                pos_hint: {"x":0, "top":0.8-0.13*2}
                text: "Password: "
                font_size: (root.width**2 + root.height**2) / 14**4

            TextInput:
                pos_hint: {"x":0.5, "top":0.8-0.13*2}
                size_hint: 0.4, 0.12
                id: passw
                multiline: False
                password: True
                font_size: (root.width**2 + root.height**2) / 14**4

        Button:
            pos_hint:{"x":0.3,"y":0.25}
            size_hint: 0.4, 0.1
            font_size: (root.width**2 + root.height**2) / 17**4
            text: "Already have an Account? Log In"
            on_release:
                root.manager.transition.direction = "left"
                root.login()

        Button:
            pos_hint:{"x":0.2,"y":0.05}
            size_hint: 0.6, 0.15
            text: "Submit"
            font_size: (root.width**2 + root.height**2) / 14**4
            on_release:
                root.manager.transition.direction = "left"
                root.submit()


<LoginWindow>:
    name: "login"

    email: email
    password: password

    FloatLayout:

        Label:
            text:"Email: "
            font_size: (root.width**2 + root.height**2) / 13**4
            pos_hint: {"x":0.1, "top":0.9}
            size_hint: 0.35, 0.15

        TextInput:
            id: email
            font_size: (root.width**2 + root.height**2) / 13**4
            multiline: False
            pos_hint: {"x": 0.45 , "top":0.9}
            size_hint: 0.4, 0.15

        Label:
            text:"Password: "
            font_size: (root.width**2 + root.height**2) / 13**4
            pos_hint: {"x":0.1, "top":0.7}
            size_hint: 0.35, 0.15

        TextInput:
            id: password
            font_size: (root.width**2 + root.height**2) / 13**4
            multiline: False
            password: True
            pos_hint: {"x": 0.45, "top":0.7}
            size_hint: 0.4, 0.15

        Button:
            pos_hint:{"x":0.2,"y":0.05}
            size_hint: 0.6, 0.2
            font_size: (root.width**2 + root.height**2) / 13**4
            text: "Login"
            on_release:
                root.manager.transition.direction = "up"
                root.loginBtn()

        Button:
            pos_hint:{"x":0.3,"y":0.3}
            size_hint: 0.4, 0.1
            font_size: (root.width**2 + root.height**2) / 17**4
            text: "Don't have an Account? Create One"
            on_release:
                root.manager.transition.direction = "right"
                root.createBtn()


<MainWindow>:
    n: n
    email: email
    created:created

    FloatLayout:
        Label:
            id: n
            pos_hint:{"x": 0.1, "top":0.9}
            size_hint:0.8, 0.2
            text: "Account Name: "

        Label:
            id: email
            pos_hint:{"x": 0.1, "top":0.7}
            size_hint:0.8, 0.2
            text: "Email: "

        Label:
            id: created
            pos_hint:{"x": 0.1, "top":0.5}
            size_hint:0.8, 0.2
            text: "Created: "
        Button:
            pos_hint:{"x": 0.2, "top": 0.5}
            size_hint:0.6,0.2
            text: "Volatility"
            on_release:
                root.popup_volatility()

        Button:
            pos_hint:{"x":0.2, "y": 0.1}
            size_hint:0.6,0.2
            text: "Log Out"
            on_release:
                app.root.current = "login"
                root.manager.transition.direction = "down"
""")
class CreateAccountWindow(Screen):
    namee = ObjectProperty(None)
    email = ObjectProperty(None)
    password = ObjectProperty(None)

    def submit(self):
        if self.namee.text != "" and self.email.text != "" and self.email.text.count("@") == 1 and self.email.text.count(".") > 0:
            if self.password != "":
                db.add_user(self.email.text, self.password.text, self.namee.text)

                self.reset()

                sm.current = "login"
            else:
                invalidForm()
        else:
            invalidForm()

    def login(self):
        self.reset()
        sm.current = "login"

    def reset(self):
        self.email.text = ""
        self.password.text = ""
        self.namee.text = ""

#class AppWindow(Screen):
#    def results_button(self):


class LoginWindow(Screen):
    email = ObjectProperty(None)
    password = ObjectProperty(None)

    def loginBtn(self):
        if db.validate(self.email.text, self.password.text):
            MainWindow.current = self.email.text
            self.reset()
            sm.current = "main"
        else:
            invalidLogin()

    def createBtn(self):
        self.reset()
        sm.current = "create"

    def reset(self):
        self.email.text = ""
        self.password.text = ""


class MainWindow(Screen):
    n = ObjectProperty(None)
    created = ObjectProperty(None)
    email = ObjectProperty(None)
    current = ""

    def logOut(self):
        sm.current = "login"
    
    def popup_volatility(self):
        result_volatility()

    def on_enter(self, *args):
        password, name, created = db.get_user(self.current)
        self.n.text = "Account Name: " + name
        self.email.text = "Email: " + self.current
        self.created.text = "Created On: " + created


class WindowManager(ScreenManager):
    pass


def invalidLogin():
    pop = Popup(title='Invalid Login',
                  content=Label(text='Invalid username or password.'),
                  size_hint=(None, None), size=(400, 400))
    pop.open()

def result_volatility():
    global history
    metrics = history.get_metrics()

    pop = Popup(title='Models Predictions(Volatility)', content=Label(text=str(metrics['Volatility'])), size_hint=(None, None), size=(400, 400))
    pop.open()


def invalidForm():
    pop = Popup(title='Invalid Form',
                  content=Label(text='Please fill in all inputs with valid information.'),
                  size_hint=(None, None), size=(400, 400))

    pop.open()


#kv = Builder.load_file("my.kv")

sm = WindowManager()
db = DataBase("app_users.txt")

screens = [LoginWindow(name="login"), CreateAccountWindow(name="create"),MainWindow(name="main")]
for screen in screens:
    sm.add_widget(screen)

sm.current = "login"


class MyMainApp(App):
    def build(self):
        return sm



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

if __name__ == "__main__":
    alpacha = Alpaca()
    s = Strategy(alpacha)
    # creating an init allows us to run the same function for
    # different tickers and resolutions
    s.add_price_event(price_event, 'AAPL', resolution='1d', init=init)
    # s.add_price_event(price_event, 'AAPL', resolution='1d', init=init)
    #global history
    history = s.backtest('2y', {'USD': 10000})
    #print(history)
    MyMainApp().run()









#if __name__ == "__main__":
 #   MyMainApp().run()