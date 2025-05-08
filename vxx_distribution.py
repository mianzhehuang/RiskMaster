# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 20:52:13 2021

@author: huang
"""

import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import colors
from matplotlib.ticker import PercentFormatter

start = '1990-01-01'
end = '2021-01-31'
vix_hist = yf.download('^VIX', start= start, end=end)

fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)

vxx_hp = pd.read_excel(r"C:\Users\huang\Dropbox\Algo Force\VXX_HP.xlsx", sheet_name="VXX HP")
vxx_hp.set_index('Date', inplace=True)
vxx_hist= (vxx_hp['Close'] /vxx_hp['Close'].shift(20) - 1).dropna()
#axs[0].hist(ret_3d, bins=20)
vxx_hist_vix = pd.concat([vxx_hist, vix_hist['Adj Close']], axis=1)
vxx_hist_vix = vxx_hist_vix.dropna()
vxx_hist_vix .columns = ['VXX Ret', 'VIX Level']

vxx_hist_vix_filter = vxx_hist_vix[vxx_hist_vix['VIX Level']>=0]
vxx_hist_vix_filter =  vxx_hist_vix_filter[vxx_hist_vix_filter['VIX Level']<20]

vxx_hist = vxx_hist_vix_filter['VXX Ret']
axs[0].hist(vxx_hist, bins=100)
axs[1].hist(vxx_hist_vix['VXX Ret'],bins=100)
#xticks(range(-1,1))

def vxx_cal(vxx_current, up_strike, down_strike, premium, histogram,step):
    #down_list = [0, -0.05, -0.1, -0.15, -0.02, -0.03, -0.04, -0.05]
    start = up_strike/vxx_current - 1
    end = down_strike/vxx_current - 1
    down_list = np.arange(start, end + (end-start)/step, (end-start)/step)
    exp_ret = 0 
    for i in range(1, len(down_list)):
        p = (sum(histogram<down_list[i-1]) - sum(histogram<down_list[i]))/len(histogram)
        exp_ret += max(p * (up_strike - (1+(down_list[i-1]+down_list[i])/2)*vxx_current),0)
        
    return exp_ret #- premium

vxx_current = 16.38
up_strike = 16
down_strike = 15.5
premium = 0.32 + 0.0072 + 0.0071 
step = 100
value = vxx_cal(vxx_current, up_strike, down_strike, premium, vxx_hist,step)
print("Option expectation return :", value)
print("Option expectation return %", value/premium)