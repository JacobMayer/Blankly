from blankly import Strategy, StrategyState, Interface
from blankly import Alpaca
from blankly.utils import trunc
from blankly.indicators import macd

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


def price_event(price, symbol, state: StrategyState):
    interface: Interface = state.interface
    # allow the resolution to be any resolution: 15m, 30m, 1d, etc.
    variables = state.variables

    variables['history'].append(price)
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


alpaca = Alpaca()
s = Strategy(alpaca)
s.add_price_event(price_event, 'TSLA', resolution='1d', init=init)
s.backtest(initial_values={'USD': 10000}, to='2y')


