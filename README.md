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



### Running bot without API for quick developmental changes: 

# Trading_bot_rienforcement
System:
- Ubuntu 16.04
- GPU (GeForce GTX 1080 Ti)


Instructions:

In order to run the code:

Make sure to clone the stable baselines repository (this is a repo that holds popular reinforcement learning algorithms, such as DeepQ, HER, A2C, etc):
```
cd
git clone https://github.com/hill-a/stable-baselines.git
```

First clone the Github Repository using terminal commands.

```
git clone https://github.com/nickhward/Trading_bot_rienforcement.git
```

Run the setup script to install required dependencies.
```
cd ~/Trading_bot_rienforcement
pip3 install -e .
```


To run the reinforcement learning agent type the command in the same directory as `pip3 install -e.'
```
python3 bot_training.py
```

You should see initial values like such running over and over until it reaches 1000000 timesteps:

![image](https://user-images.githubusercontent.com/78880630/138394972-58f1b4cb-6bef-4cd1-8584-4de2dcea3dbc.png)

And a graph that looks similar to this when it is finished running through the timesteps:
![image](https://user-images.githubusercontent.com/78880630/145345019-99020b99-f858-47ed-8afb-ffd0259384cc.png)
