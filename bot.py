from blankly import Strategy, StrategyState, Interface
from blankly import Alpaca
from blankly.utils import trunc
from blankly.indicators import macd, sma, rsi
import streamlit as st
import pandas as pd
from bokeh.plotting import figure, output_file, show
from bokeh.models import ColumnDataSource, NumeralTickFormatter, HoverTool, DaysTicker, DatetimeTickFormatter, graphs



SHORT_PERIOD = 12
LONG_PERIOD = 26
SIGNAL_PERIOD = 9


def init(symbol, state: StrategyState):
    interface: Interface = state.interface
    resolution: float = state.resolution
    variables = state.variables
    # initialize the historical data
    variables['history'] = interface.history(symbol, 800, 
        resolution,
        return_as='list')['close']
    variables['short_period'] = SHORT_PERIOD
    variables['long_period'] = LONG_PERIOD
    variables['signal_period'] = SIGNAL_PERIOD
    variables['has_bought'] = False


def macd_strat(var, inter, symbol, price):
    variables = var
    interface = inter
    macd_res, macd_signal, macd_histogram = macd(variables['history'], 
                                                 short_period=variables['short_period'],
                                                 long_period=variables['long_period'],
                                                 signal_period=variables['signal_period'])

    slope_macd = (macd_res[-1] - macd_res[-5]) / 5  # get the slope of the last 5 MACD_points
    prev_macd = macd_res[-2]
    curr_macd = macd_res[-1]
    curr_signal_macd = macd_signal[-1]

    # We want to make sure this works even if curr_macd does not equal the signal MACD
    is_cross_up = slope_macd > 0 and curr_macd >= curr_signal_macd > prev_macd

    is_cross_down = slope_macd < 0 and curr_macd <= curr_signal_macd < prev_macd
    if is_cross_up and not variables['has_bought']:
        # buy with all available cash
        interface.market_order(symbol, 'buy', int(interface.cash/price))
        variables['has_bought'] = True
    elif is_cross_down and variables['has_bought']:
        # sell all of the position
        interface.market_order(symbol, 'sell', int(interface.account[symbol].available))
        variables['has_bought'] = False

def golden_cross(var, inter, symbol, price):
    variables = var
    interface = inter
    
    sma200 = sma(variables['history'], period=200)
    # match up dimensions
    sma50 = sma(variables['history'], period=50)[-len(sma200):]
    diff = sma200 - sma50
    slope_sma50 = (sma50[-1] - sma50[-5]) / 5 # get the slope of the last 5 SMA50 Data Points
    prev_diff = diff[-2]
    curr_diff = diff[-1]
    is_cross_up = slope_sma50 > 0 and curr_diff >= 0 and prev_diff < 0
    is_cross_down = slope_sma50 < 0 and curr_diff <= 0 and prev_diff > 0
    # comparing prev diff with current diff will show a cross
    if is_cross_up and not variables['has_bought']:
        interface.market_order(symbol, 'buy', int(interface.cash/price))
        variables['has_bought'] = True
    elif is_cross_down and variables['has_bought']:
        # use strategy.base_asset if on CoinbasePro or Binance
        # truncate here to fix any floating point errors
        interface.market_order(symbol, 'sell', int(interface.account[symbol].available))
        variables['has_bought'] = False
    
def rsi(var, inter, symbol, price):
    variables = var
    interface = inter
    rsi = blankly.indicators.rsi(state.variables['history'])
    if rsi[-1] < 30 and not state.variables['owns_position']:
        # Dollar cost average buy
        buy = int(state.interface.cash/price)
        state.interface.market_order(symbol, side='buy', size=buy)
        state.variables['owns_position'] = True
    elif rsi[-1] > 70 and state.variables['owns_position']:
        # Dollar cost average sell
        curr_value = int(state.interface.account[state.base_asset].available)
        state.interface.market_order(symbol, side='sell', size=curr_value)
        state.variables['owns_position'] = False
    
def price_event(price, symbol, state: StrategyState):
    interface: Interface = state.interface
    # allow the resolution to be any resolution: 15m, 30m, 1d, etc.
    variables = state.variables

    variables['history'].append(price)
    
    macd_strat(variables, interface, symbol, price)
<<<<<<< HEAD
 #   golden_cross(variables, interface, symbol, price)
 #   rsi(variables, interface, symbol, price)
=======
    golden_cross(variables, interface, symbol, price)
    rsi(variables, interface, symbol, price)
>>>>>>> 9b59e3493fcd4d3dede217f1d63400089d365120


# UI creation
# initial version by Justin Carlson
# static homepage stuff for now

alpaca = Alpaca()
s = Strategy(alpaca)

def addStockEvent(stock, resolution, init):
    s.add_price_event(price_event, stock, resolution, init)
    return

def buildChartData(amount, time, interface):

    backTestCallback = s.backtest(initial_values={'USD': amount}, to=time)

    backTestAcctInfo = backTestCallback.get_account_history()

    time = backTestAcctInfo['time'].tolist()
    value = backTestAcctInfo['USD'].tolist()

    infoFigure = figure(
     title='Account Value (USD)',
     x_axis_label='Time',
     x_axis_type='datetime',
     y_axis_label='Value (USD)')

    infoFigure.line(time, value, legend_label='Trend', line_width=2)

    return infoFigure

def buildDashboard():
   
    menu, graphs = st.columns([2,1])
  
    with menu:
        st.write(""" # PPI \n""")
        st.sidebar.button(label="Dashboard", help="Dashboard")

    with graphs:
        #build charts and puts on webpage of all stock events
        st.bokeh_chart(buildChartData(10000, '2y', graphs))
    
#in the future this will be iterated upon based on user requests
addStockEvent('TSLA', '1d', init)

#build user interface on webpage
buildDashboard()
