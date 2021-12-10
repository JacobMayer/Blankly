
# Blankly
Bot that trained using Rienforcement learning

Our Trading bot uses the public use library 'Blankly' and we have implemented its features into a full fledged trading bot


==== How to use our bot ====

Step 1: Download all files from this branch

Step 2: Run any of the python files for the desired strategy

Step 3: An HTML Webpage with graphs will appear, 
along with data in the console for valuable metrics such as Compound Annual Growth, Cumulative Returns, etc.

Step 4: To edit the Symbol/Ticker (i.e. MSFT, SPY, TSLA, NFLX) change s.add_price_event and the resolution you'd like (i.e. 1m, 5m, 30m, 1d)
You can even add additional s.add_price_event to backtest multiple tickers at one time

Step 5: You can change s.backtest to change initial account values, and the time frame for the data (i.e. 2y, 1y, 5y) or enter a specific time frame using 


### Bot on its own training on Bitcoin data
Must have python3.7+ for Blankly to work...

To run bot clone this branch:
```
cd 
git clone https://github.com/JacobMayer/Blankly.git
```
Then type the following:
```
cd ~/Blankly
python xgboost.py
```

The image bellow is the data split into training, validation, and testing.

![image](https://user-images.githubusercontent.com/78880630/145518043-63f066bd-06ac-4f7e-b716-35d5978aa7ee.png)


The first five numerical predicted values versus the truth values:
```
truth values      = [47706.12 48960.79 46942.22 49058.67 48902.4 ]
prediction values = [48196.266 49143.19  49143.19  49143.19  47668.664]
```

The final graph gives predicted trendline and the truth trendline:

![image](https://user-images.githubusercontent.com/78880630/145519015-52c775b6-4e44-4f1a-90b3-a1d3461d7e73.png)
