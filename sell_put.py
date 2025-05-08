# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 16:39:04 2020

@author: huang
"""

import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import colors
from matplotlib.ticker import PercentFormatter

start = '2000-01-01'
end = '2021-01-07'
spy_hist = yf.download('SPY', start= start, end=end)
vix_hist = yf.download('^VIX', start= start, end=end)

#spy_ret_hist3= (spy_hist['Low'].shift(-3) /spy_hist['Adj Close'] - 1).dropna()
#spy_ret_hist2= (spy_hist['Low'].shift(-2) /spy_hist['Adj Close'] - 1).dropna()
#spy_ret_hist1= (spy_hist['Low'].shift(-1) /spy_hist['Adj Close'] - 1).dropna()
#spy_ret_hist = min()
n = 15
margin = 500 
max_loss = 0.6
premium = 40 
loss = -0.105
gain = 0.073
vix_low = 20
vix_high = 30
#get the past n days lwo price
for i in range(1,n):
    spy_hist[i] = spy_hist['Low'].shift(-i)

spy_hist[str(n)+'d_max_loss'] = spy_hist.apply(lambda x: min([x[i]/x['High']-1 for i in range(1,n)]), axis=1)
spy_ret_hist_loss = spy_hist[str(n)+'d_max_loss']


#get the past n days high price
for i in range(1,n):
    spy_hist[i] = spy_hist['High'].shift(-i)

spy_hist[str(n)+'d_max_gain'] = spy_hist.apply(lambda x: max([x[i]/x['Low']-1 for i in range(1,n)]), axis=1)
spy_ret_hist_gain = spy_hist[str(n)+'d_max_gain']

result = pd.concat([spy_ret_hist_loss, spy_ret_hist_gain, vix_hist['Adj Close']], axis=1)
result.columns = [str(n)+'d_spy_max_loss', str(n)+'d_spy_max_gain', 'VIX']
result = result.dropna()


spy_ret_vix = result[result['VIX']>=vix_low]
spy_ret_vix = spy_ret_vix[spy_ret_vix['VIX']<=vix_high]
up_loss_prob = sum(spy_ret_vix[str(n)+'d_spy_max_gain']>gain)/len(spy_ret_vix[str(n)+'d_spy_max_gain'])
down_loss_prob = sum(spy_ret_vix[str(n)+'d_spy_max_loss']<loss)/len(spy_ret_vix[str(n)+'d_spy_max_loss'])

#loss_list = np.arange(-0.11, -0.02, 0.01)
#loss_prob = []
#for loss in loss_list:
#    loss_prob.append(sum(spy_ret_vix20['SPY Return']<loss)/len(spy_ret_vix20['SPY Return']))
#win_rate = sum(spy_ret_vix20['SPY Return']<-0.08)/len(spy_ret_vix20['SPY Return'])

#plt.scatter(loss_list, loss_prob)
#bad_period = spy_ret_vix20[spy_ret_vix20['SPY Return']<-0.03]
b = premium / (margin*max_loss-premium)
p = 1-max(up_loss_prob, down_loss_prob)
def kari_formula(p, b):
    return (p*b- (1-p))/b

print(up_loss_prob)
print(down_loss_prob)
print(kari_formula(p,b)*margin/(margin*max_loss-premium))
    