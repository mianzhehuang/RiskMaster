# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 22:41:27 2022

@author: huang
"""

import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
import datetime
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def cmo(df, window):
    df['diff'] = df['Adj Close'] - df['Adj Close'].shift(1)
    df['udiff'] = df['diff'].apply(lambda x: 0 if x<0 else x)
    df['ddiff'] = df['diff'].apply(lambda x: 0 if x>=0 else abs(x))
    df['Su'] = df['udiff'].rolling(window).sum()
    df['Sd'] = df['ddiff'].rolling(window).sum()
    df['cmo_'+str(window)] = 100 * (df['Su'] - df['Sd'])/(df['Su'] + df['Sd'])
    return df['cmo_'+str(window)]

def cci(df, window):
    df['typical ind'] = (df['Open'] + df['High'] + df['Close'])/3
    df['typical'] = df['typical ind'].rolling(window).sum()
    df['MA'] = df['typical'].rolling(window).sum()/window
    df['tp - MA'] = df.apply(lambda x: abs(x['typical'] - x['MA']), axis=1)
    df['MD'] = df['tp - MA'].rolling(window).sum()/window
    df['cci_'+str(window)] = (df['typical'] - df['MA'])/(0.015 * df['MD'])
    return df['cci_'+str(window)]

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
    df['adx_'+str(window)] = df['DX'].rolling(window).mean()
    return df['adx_'+str(window)]
 

def sma_diff(df_orig, window):
    df = df_orig.copy()
    df['sma_'+str(window)] = (df['Adj Close'].rolling(window+1).sum() - df['Adj Close'])/window
    df['sma_'+str(window)+'_diff'] = df['Adj Close']/df['sma_'+str(window)] -1 
    return df['sma_'+str(window)+'_diff']

def sma(df_orig, window):
    df = df_orig.copy()
    df['sma_'+str(window)] = (df['Adj Close'].rolling(window+1).sum() - df['Adj Close'])/window
    #df['sma_'+str(window)+'_diff'] = df['Adj Close']/df['sma_'+str(window)] -1 
    return df['sma_'+str(window)]

def max_drop(df_orig, window):
    df = df_orig.copy()
    for i in range(1,window):
        df[i] = df['Low'].shift(-i)
    df = df.dropna()
    df[str(window)+'d_max_loss'] = df.apply(lambda x: min([x[i]/x['High']-1 for i in range(1,window)]), axis=1)
    return df[str(window)+'d_max_loss']

def max_gain(df_orig, window):
    df = df_orig
    for i in range(1,window):
        df[i] = df['High'].shift(-i)
    df = df.dropna()
    df[str(window)+'d_max_gain'] = df.apply(lambda x: max([x[i]/x['Low']-1 for i in range(1, window)]), axis=1)
    return df[str(window)+'d_max_gain']

def std_vol(df_orig, window):
    df = df_orig.copy()
    df['ret'] = df['Adj Close']/df['Adj Close'].shift(1) - 1
    df['shif_1'] = df['ret'].shift(1)
    df['std'] = df['shif_1'].rolling(window).std()
    df['vol_' + str(window)+'_day'] = df['std'] * np.sqrt(252)
    return df['vol_' + str(window)+'_day']



    

#%% test part 
### data processing 

start = '1990-01-01'
end = '2022-01-01'
ret_days = 1 # test spy xx days return
windows = 22
#ret_limit = 0.05 # test spy drop limit
result = pd.DataFrame()
strike_distance = pd.read_csv(r"C:\Users\huang\Dropbox\Algo Force\Signal\Strike Distances.csv")
vxx_distance = pd.read_excel(r"C:\Users\huang\Dropbox\Algo Force\VXX_HP.xlsx")
spy_hist = yf.download('SPY', start= start, end=end)
vix_hist = yf.download('^VIX', start= start, end=end)
vxx_hist = yf.download('VXX', start= '2021-01-01', end=end)
vxx_hist['ret'] = vxx_hist['Adj Close']/vxx_hist['Adj Close'].shift(ret_days) -1 
vxx_distance.set_index('Date',inplace=True) 
vxx_distance['ret'] = vxx_distance['Close'] / vxx_distance['Close'].shift(ret_days) - 1
vxx_full = pd.concat([vxx_distance['ret'], vxx_hist['ret']], axis=0)
vxx_full = pd.DataFrame(vxx_full)
vxx_full.columns = ['vxx_ret']
spy_hist['spy_ret'] = spy_hist['Adj Close'] / spy_hist['Adj Close'].shift(ret_days) - 1 
spy_vxx = pd.concat([vxx_full, pd.DataFrame(spy_hist['spy_ret'])], axis=1)
spy_vxx.dropna(inplace=True)

#spy_vxx_tail = spy_vxx[abs(spy_vxx['spy_ret'])>0.03]
#spy_vxx_tail.plot(x='spy_ret', y='vxx_ret',kind='scatter')


#%% get pattern
pattern_dict ={}
for i in range(len(spy_vxx)):
    if i+windows <= len(spy_vxx):
        spy_vxx_scenario = spy_vxx[i:i+windows] + 1
        spy_vxx_scenario = spy_vxx_scenario.cumprod()
        ind = spy_vxx_scenario.index[0]
        ind = ind.strftime('%Y-%m-%d')
        first_row = pd.DataFrame({'vxx_ret':[1], 'spy_ret':[1]})
        spy_vxx_scenario.reset_index(inplace=True)
        spy_vxx_scenario = spy_vxx_scenario[['vxx_ret', 'spy_ret']]
        spy_vxx_scenario = pd.concat([first_row, spy_vxx_scenario], ignore_index=True)
        spy_vxx_scenario.reset_index(inplace=True)
        spy_vxx_scenario = spy_vxx_scenario[['vxx_ret', 'spy_ret']]
        pattern_dict[ind] = spy_vxx_scenario
        pattern_dict[ind].plot()
    #pattern_dict[spy_vxx_scenario.index[0]] = 
    
    
    
    
