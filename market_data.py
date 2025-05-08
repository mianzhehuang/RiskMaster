# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 05:21:40 2018

@author: huang
"""

import numpy as np, pandas as pd 
import matplotlib
from datetime import datetime as dt
import pandas_datareader.data as data
import sys 
#%matplotlib inline 

def api2df(stocks =None, api = 'quandl',start =None, end = None, price_type = 'AdjClose'):
    '''
    input: 
        'stocks' data type - list of tickers;
        'start' or 'end' data type - datetime;
    output:
        'dfs' DataFrame
    '''    
    dfs = pd.DataFrame()
    for stock in stocks:
        api_data = data.DataReader(stock, api, start, end, api_key='d_oZwp6fcNt6kNBczx2k')[price_type]
        #api_data = api_data.reset_index()
        #df = pd.DataFrame(data = list(api_data[price_type]),index = api_data['Date'],columns =[stock])#api_data['Symbol'][0]
        df = pd.DataFrame(api_data)
        df.columns = [stock]
        #pd.DataFrame(data=api_data, columns=[stock])
        dfs = pd.concat([df,dfs],axis = 1)
    
    return dfs
    
    
def price2ret(price, interval = 1,overlap = True, ret_type = 'log_ret'):
    '''
    input: 
        'price' data type - pandas.Series or DataFrame;
        'intervals' data type - string. Available: daily, weekly;
        'overlap' data type - boolean;
        'ret_type' log return or absolute return
    output:
        'ret' data type - Series
    '''
    price.sort_index(ascending= True,inplace = True) 
    ret_types = {'log_ret': np.log,
                'abs_ret': lambda x:x}
    if overlap:
        ret = ret_types.get(ret_type)(price).diff(interval)
    else:
        sub_date = pd.bdate_range(price.index[0],price.index[-1],freq = str(interval)+'B')
        sub_price = price.ix[sub_date]
        ret = ret_types.get(ret_type)(sub_price).diff()
        
    return ret 

#def corrMatrix(ret):
    
    
def main(argv):
    '''test config'''
    start = '2018-03-27'#dt(2018,3,27)
    end   = '2018-12-27'#dt(2018,4,27)
    #stocks = ['FB', 'AMZN', 'GOOG', 'NFLX', 'GLD']
    stocks = ['AAPL']
    stocks_data = api2df(stocks = stocks,start = start, end = end)
    #stocks_data = pdr.get_data_yahoo('FB')
    print(stocks_data)
    #ret = price2ret(stocks_data,interval = 5, overlap = False, ret_type = 'log_ret')
    #print(ret)
    #import calibration
    #vm = calibration.VasicekModel(ts = stocks_data, rettype = 'Normal')
    #print(vm.getCalibOLS())
    #paths = vm.project(path = 10, T = 21/260.0)
    #import matplotlib.pyplot as plt
    #plt.plot(paths.T)

    
    
if __name__ == "__main__":
    main(sys.argv[1:])

