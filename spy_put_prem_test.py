# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 13:03:00 2021

@author: huang
"""

import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import colors
from matplotlib.ticker import PercentFormatter

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

def std_vol(df_orig, window):
    df = df_orig.copy()
    df['ret'] = df['Adj Close']/df['Adj Close'].shift(1) - 1
    df['shif_1'] = df['ret'].shift(1)
    df['std'] = df['shif_1'].rolling(window).std()
    df['vol_' + str(window)+'_day'] = df['std'] * np.sqrt(252)
    return df['vol_' + str(window)+'_day']

def vwap_delta(df_orig, window):
    df = df_orig.copy()
    df['typical ind'] = (df['Open'] + df['High'] + df['Close'])/3
    df['typical price volume'] = df['typical ind'] * df['Volume']
    df['tpvs'] = df['typical price volume'].rolling(window).sum()
    df['vs'] = df['Volume'].rolling(window).sum()
    df['vwap'] = df['tyvs']/df['vs']
    df['vwap_delta_'+str(window)] = df['vwap']/df['vwap'].shift(window)-1
    return df['vwap_delta_'+str(window)]
    



    

#%% test part 
### data processing 

start = '1990-01-01'
end = '2021-01-28'
ret_days = 22 # test spy xx days return
ret_limit = -0.06 # test spy drop limit
result = pd.DataFrame()

spy_hist = yf.download('SPY', start= start, end=end)
vix_hist = yf.download('^VIX', start= start, end=end)

#%% Construct the Y-variable
result = pd.concat([result, max_drop(spy_hist, ret_days)], axis=1) 
result['spy_ret_limit_loss_fl'] = result[str(ret_days)+'d_max_loss'].apply(lambda x: 1 if x>ret_limit else 0)

#%% Contruct the X-variable for SMA
result = pd.concat([result, sma_diff(spy_hist, 10)], axis=1) 
result = pd.concat([result, sma_diff(spy_hist, 20)], axis=1)
result = pd.concat([result, sma_diff(spy_hist, 100)], axis=1)

#%% Contruct the X-variable for VIX
vix_hist['vix_sma_20'] = sma(vix_hist, 20)
vix_hist.rename(columns={'Adj Close':'vix_current'}, inplace=True)

result = pd.concat([result, vix_hist[['vix_current', 'vix_sma_20']]], axis=1)


#%% Contruct the X-variable for SPY realized vol
result = pd.concat([result, std_vol(spy_hist, 20)], axis=1) 
result = pd.concat([result, std_vol(spy_hist, 100)], axis=1) 

#%% Construct CCI level
result = pd.concat([result, cci(spy_hist, 7)], axis=1) 
result = pd.concat([result, cci(spy_hist, 14)], axis=1) 
result = pd.concat([result, cci(spy_hist, 40)], axis=1) 

#%% Construct CCI level
result = pd.concat([result, cmo(spy_hist, 7)], axis=1) 
result = pd.concat([result, cmo(spy_hist, 14)], axis=1)  

#%% Contruct CCI ADX level
result = pd.concat([result, adx(spy_hist,7)], axis =1)
result = pd.concat([result, adx(spy_hist,14)], axis =1)


test_result = result[['vix_current', 'vix_sma_20',
                      'vol_20_day', 'vol_100_day',
                      'sma_10_diff', 'sma_20_diff', 'sma_100_diff',
                      'cci_7', 'cci_14', 'cci_40',
                      'cmo_7', 'cmo_14',
                      'adx_7', 'adx_14']]

result = result[['22d_max_loss', 'spy_ret_limit_loss_fl',
                 'vix_current', 'vix_sma_20',
                  'vol_20_day', 'vol_100_day',
                  'sma_10_diff', 'sma_20_diff', 'sma_100_diff',
                  'cci_7', 'cci_14', 'cci_40',
                  'cmo_7', 'cmo_14',
                  'adx_7', 'adx_14']]

result.dropna(inplace=True)
test_result.dropna(inplace=True)
result.to_excel("spy_test_result.xlsx")
test_result.to_excel("spy_predict_result.xlsx")
#result[['spy_ret_limit_loss_fl', 'vix_current', 'vix_sma_20',
#        'vol_20_day', 'vol_100_day',
#        'sma_10_diff', 'sma_20_diff', 'sma_100_diff',
#        'cci_7', 'cci_14', 'cci_40',
#        'cmo_7', 'cmo_14',
#        'adx_7', 'adx_14'
#        ]].to_excel("spy_test_result.xlsx")