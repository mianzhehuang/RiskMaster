# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 09:52:10 2022

@author: huang
"""

import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import colors
from matplotlib.ticker import PercentFormatter



def vixchange_RSI_func(RSI, previous_up, previous_down):
    RS = 100 / (100 -RSI) - 1
    vixchange = RS * previous_down - previous_up
    if vixchange>=0:
        return vixchange
    else:
        vixchange = previous_up/RS - previous_down
        return -vixchange

start = '1991-01-01'
end = '2022-07-19'
window = 10
result = pd.DataFrame()
strike_distance = pd.read_csv(r"C:\Users\huang\Dropbox\Algo Force\Signal\Strike Distances.csv")
spy_hist = yf.download('SPY', start= start, end=end)
vix_hist = yf.download('^VIX', start= start, end=end)


result = pd.concat([spy_hist[["Adj Close"]], vix_hist[["Adj Close"]]], axis=1)
result.columns = ["SPY", "VIX"]
result["VIX MOVE"] = result["VIX"]/result["VIX"].shift(1) -1
result = result.dropna()
result["VIX UP"] = result["VIX MOVE"].apply(lambda x: x if x>=0 else 0)
result["VIX DOWN"] = result["VIX MOVE"].apply(lambda x: abs(x) if x<=0 else 0)
result["AVG UP"] = result["VIX UP"].rolling(window).mean()
result["AVG DOWN"] = result["VIX DOWN"].rolling(window).mean()
result["RS"] = result["AVG UP"] / result["AVG DOWN"]
result["RSI"] = 100 - 100 / (1+result["RS"]) 

result["UP_PREVIOUS"] = result["AVG UP"] * window - result["VIX UP"]
result["DOWN_PREVIOUS"] = result["AVG DOWN"] * window - result["VIX DOWN"]
result["VIX CHANGE RSI70"] = result.apply(lambda x: vixchange_RSI_func(70, x["UP_PREVIOUS"], x["DOWN_PREVIOUS"]), axis=1)
result["VIX CHANGE RSI50"] = result.apply(lambda x: vixchange_RSI_func(50, x["UP_PREVIOUS"], x["DOWN_PREVIOUS"]), axis=1)
result["VIX CHANGE RSI30"] = result.apply(lambda x: vixchange_RSI_func(30, x["UP_PREVIOUS"], x["DOWN_PREVIOUS"]), axis=1)
#result["RS_RSI70"] = (result["VIX CHANGE RSI70"] + result["UP_PREVIOUS"])/ result["DOWN_PREVIOUS"] 
#result["RS_RSI70_N"] =  ( result["UP_PREVIOUS"])/(abs(result["VIX CHANGE RSI70"]) + result["DOWN_PREVIOUS"])   
result["VIX_RSI70"] = result["VIX"].shift(1) * (result["VIX CHANGE RSI70"] + 1)
result["VIX_RSI50"] = result["VIX"].shift(1) * (result["VIX CHANGE RSI50"] + 1)
result["VIX_RSI30"] = result["VIX"].shift(1) * (result["VIX CHANGE RSI30"] + 1)

result[["VIX", "VIX_RSI30", "VIX_RSI50", "VIX_RSI70"]][-60:].to_csv(r"VIX_momentum_0618.csv")
#result.to_csv(r"spxvsvix_hist.csv")