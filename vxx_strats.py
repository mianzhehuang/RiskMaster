# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 10:57:42 2021

@author: huang
"""

import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import colors
from matplotlib.ticker import PercentFormatter



def max_gain(df_orig, window):
    df = df_orig
    for i in range(1,window):
        df[i] = df['High'].shift(-i)
    df = df.dropna()
    df[str(window)+'d_max_gain'] = df.apply(lambda x: max([x[i]/x['Low']-1 for i in range(1, window)]), axis=1)
    return df[str(window)+'d_max_gain']

#%%


start = '1990-01-01'
end = '2021-01-31'
vix_hist = yf.download('^VIX', start= start, end=end)

#fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)

vxx_hp = pd.read_excel(r"C:\Users\huang\Dropbox\Algo Force\VXX_HP.xlsx", sheet_name="VXX HP")
vxx_hp.set_index('Date', inplace=True)
vxx_hist_prior = (vxx_hp['Close'] /vxx_hp['Close'].shift(3) - 1).dropna()
vxx_hist_post = (vxx_hp['Close'].shift(-15) /vxx_hp['Close'] - 1).dropna()

vxx_hist_trend = pd.concat([vxx_hist_post, vxx_hist_prior], axis=1)
vxx_hist_trend.columns = ['vxx_ret_after15','vxx_ret_before3']
vxx_hist_trend.dropna(inplace=True)

#%%
#################################################################################
#######Probability vs conditional probability####################################
p_drop10pct = sum(vxx_hist_trend['vxx_ret_after15']<-0.15) / len(vxx_hist_trend['vxx_ret_after15'])
vxx_hist_trend_sub = vxx_hist_trend[vxx_hist_trend['vxx_ret_before3']>0.1]
p_drop10pct_cond = sum(vxx_hist_trend_sub['vxx_ret_after15']<-0.15) / len(vxx_hist_trend_sub['vxx_ret_after15'])
print(p_drop10pct, ",", p_drop10pct_cond)

#%%#############################################################################
#####Strategy###################################################################


