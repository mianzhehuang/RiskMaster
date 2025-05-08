# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 13:04:20 2020

@author: huang
"""

#def cci(ts):
#    ts['typical'] = ts['open'] + ts['high'] + ts['close']
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import colors
from matplotlib.ticker import PercentFormatter

def cmo(df, window):
    #df['Su'] = df['High'].rolling(window).sum()
    #df['Sd'] = df['Low'].rolling(window).sum()
    #df['CMO'] = 100 * (df['Su'] - df['Sd'])/(df['Su'] + df['Sd'])
    df['diff'] = df['Adj Close'] - df['Adj Close'].shift(1)
    df['udiff'] = df['diff'].apply(lambda x: 0 if x<0 else x)
    df['ddiff'] = df['diff'].apply(lambda x: 0 if x>=0 else abs(x))
    df['Su'] = df['udiff'].rolling(window).sum()
    df['Sd'] = df['ddiff'].rolling(window).sum()
    df['CMO'] = 100 * (df['Su'] - df['Sd'])/(df['Su'] + df['Sd'])
    return df

def cci(df, window):
    df['typical ind'] = (df['Open'] + df['High'] + df['Close'])/3
    df['typical'] = df['typical ind'].rolling(window).sum()
    df['MA'] = df['typical'].rolling(window).sum()/window
    df['tp - MA'] = df.apply(lambda x: abs(x['typical'] - x['MA']), axis=1)
    df['MD'] = df['tp - MA'].rolling(window).sum()/window
    df['cci'] = (df['typical'] - df['MA'])/(0.015 * df['MD'])
    return df

def adx(df, window):
    df['Current High - Current Low'] = df['High'] - df['Low']
    df['Current High - Previous Close'] = df['High'] - df['Close'].shift(1)
    df['Current Low - Previous Close'] = df['Low'] - df['Close'].shift(1)
    df['TR'] = df.apply(lambda x: np.max([x['Current High - Current Low'],
                                      x['Current High - Previous Close'],
                                      x['Current Low - Previous Close']]), axis=1)
    df['ATR'] = df['TR'].rolling(window).mean()
    df['+DM'] = df['High'] - df['High'].shift(1)
    df['-DM'] = df['Low'].shift(1) - df['Low']
    df['+DX'] = df.apply(lambda x: x['+DM'] if x['+DM']>x['-DM'] and x['+DM']>0 else 0, axis=1)
    df['-DX'] = df.apply(lambda x: x['-DM'] if x['-DM']>x['+DM'] and x['-DM']>0 else 0, axis=1)
    df['Smooth +DX'] = df['+DX'].rolling(window).mean()
    df['Smooth -DX'] = df['-DX'].rolling(window).mean()
    df['+DMI'] = df['Smooth +DX']/df['ATR'] * 100
    df['-DMI'] = df['Smooth -DX']/df['ATR'] * 100
    df['DX'] = df.apply(lambda x: abs(x['+DMI'] - x['-DMI']), axis=1)
    df['ADX'] = df['DX'].rolling(window).mean()
    return df


signal_map = {'cmo':cmo,
              'adx':adx,
              'cci':cci}
#lst = ['TSLA', 'SPLK', 'ICLN']
qqq_holding = pd.read_csv(r"C:\Users\huang\Dropbox\Carrick Huang\Investment\QQQ_holding.csv")
lst = list(qqq_holding['Holding Ticker'])
#tsla = yf.download("TSLA", start= "2020-01-01", end="2020-12-24")
#tsla_cmo = cmo(tsla, 14)
def get_signal(ticker,  start='2009-01-01', end='2009-12-31', signal='cmo', window = 14):
    ticker_hist = yf.download(ticker, start= start, end=end)
    signal_result = signal_map[signal](ticker_hist, window)
    return signal_result

for ticker in lst:
    try: 
        signal_result = get_signal(ticker,  start='2018-01-01', end='2021-06-02', signal='cci', window = 14)
        if list(signal_result['cci'])[-1]<0:
            print(ticker)
    except:
        pass
        
signal_result = get_signal('AAPL',  start='2018-01-01', end='2021-06-02', signal='cci', window = 14)

#%%
# create figure and axis objects with subplots()
fig,ax = plt.subplots()
# make a plot
ax.plot(signal_result.index, signal_result['Adj Close'], color="red")
# set x-axis label
ax.set_xlabel("Date",fontsize=14)
# set y-axis label
ax.set_ylabel("VXX",color="red",fontsize=14)

# twin object for two different y-axis on the sample plot
ax2=ax.twinx()
# make a plot with different y-axis using second axis object
ax2.plot(signal_result.index, signal_result['cci'],color="blue")
ax2.set_ylabel("cci",color="blue",fontsize=14)
plt.show()


#%%
signal_result['Spike'] = signal_result['cci'].apply(lambda x: 1 if x>0 else (np.nan if pd.isna(x) else 0))
signal_spike = signal_result['Spike'].dropna().values
count = 1 
period_list = []
normal_days = []
spike_days = []
for i in range(1, len(signal_spike)):
    if signal_spike[i-1] != signal_spike[i]:
        period_list.append([count, signal_spike[i-1]])
        if signal_spike[i-1]>0:
            spike_days.append(count)
        else:
            normal_days.append(count)
        count = 1
    else:
        count += 1

#%%    
fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)

spy_hist= (signal_result['Adj Close'] /signal_result['Adj Close'].shift(20) - 1).dropna()

#axs[0].hist(ret_3d, bins=20)
axs[0].hist(spy_hist, bins=100)
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
        
    return exp_ret

vxx_current = 16.67
up_strike = 16.5
down_strike = 14.5
premium = 0 
step = 100
value = vxx_cal(vxx_current, up_strike, down_strike, premium, vxx_hist,step)
    
        
    