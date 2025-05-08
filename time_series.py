# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 23:56:59 2018

@author: huang
"""
import numpy as np
import scipy.stats as stats
import market_data
import pandas as pd
import sys
import matrix 

def calcSimpleVaR(rets, alpha = 0.01, method = 'bootstrap'):
    if method == 'norm':
        return stats.norm.ppf(alpha, rets.mean(), rets.std())
    elif method == 'bootstrap':
        return rets.quantile(alpha)
    
def calcSimpleES(rets, alpha = 0.01):
    quantile  = stats.norm.ppf(alpha)
    return (rets.mean() - rets.std() * (stats.norm.pdf(quantile) / alpha)).values

        
def main(argv):
    from datetime import datetime as dt
    start = dt(2014,1,1)
    end   = dt(2017,1,1)    
    stocks = ['SPY']
    stocks_data = market_data.api2df(stocks = stocks,start = start, end = end)
    ret = market_data.price2ret(stocks_data,interval = 5, overlap = False, ret_type = 'log_ret')
    annual_ret = matrix.annualize(ret, datatype = 'logret', freqtype = 'weekly')
    print(calcSimpleVaR(ret, alpha = 0.05, method = 'norm'))
    print(calcSimpleES(ret, alpha = 0.05))
    

def deltaGammaVaR(alpha, t, mu, sigma, S=0, K=3, T=260.0):
    t = t/T
    z = stats.norm.ppf(alpha)
    z = z + S/6 *(z**2 - 1) + (K-3)/24 * (z**3 - 3*z) - S**2 / 36 * (2*z**3 - 5*z)
    VaR = -(mu * t + z * sigma * np.sqrt(t))
    return VaR
    
    

        

    
if __name__ == "__main__":
    main(sys.argv[1:])