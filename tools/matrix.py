# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 23:55:04 2018

@author: Wei
"""

import numpy as np, pandas as pd 
import matplotlib
import market_data
import scipy 
import sys
from numpy.linalg import inv
from scipy.sparse.linalg import lsmr
 
def _mean(X,w):
    '''Calculate the weighted mean'''
    return np.sum(X*w)/np.sum(w)

def _cov(X,Y,w):
    '''Calculate the weighted covariance'''
    return np.sum(w * (X - _mean(X, w)) * (Y - _mean(Y, w))) / np.sum(w)

def _corr(X,Y,w):
    '''Calculate the weighted correlation'''
    w = pd.Series(data = w, index = X.index)
    X_mask = X.isnull()
    Y_mask = Y.isnull()
    X = X.mask(Y_mask).dropna()
    Y = Y.mask(X_mask).dropna()
    w = w.mask(X_mask).dropna()
    w = w.mask(Y_mask).dropna()
    return _cov(X, Y, w) / np.sqrt(_cov(X, X, w) * _cov(Y, Y, w))    

def _w(decay, n):
    '''Calculatet the exponential weighted average returns'''
    if decay == 1:
        return [1.0]*n
    else:
        #w = np.sort((1. - decay)/(1. - decay ** n) * np.exp(np.array(range(n)) * np.log(decay)))
        w = [decay**i for i in range(n)]
    return w 

def annualize(data, datatype = None, interval = 5.0):
    bdays = 252.0 
    #freqs = {'daily':1.0, 'weekly':5.0, 'monthly':21.0}
    #freq  = freqs.get(freqtype)
    convertFunc = {'vol': lambda x: x*np.sqrt(bdays/interval),
                   'log_ret': lambda x: x * bdays/ interval}
    
    return convertFunc.get(datatype)(data)

def histVol(rets, weight = True, decay = 0.97, interval = 5.0):
    n = len(rets)
    if weight:
        w = _w(decay, n)
    else:
        w = _w(1, n)
    rets.sort_index(axis = 0, ascending = False,inplace = True)
    w = pd.Series(data = w, index = rets.index)
    vols = np.array([np.sqrt(_cov(rets[X],rets[X], w)) for X in rets.columns])
    vols_matrix = np.diag(vols)
    vols_matrix = pd.DataFrame(data = vols_matrix, index = rets.columns, columns = rets.columns)
    return annualize(vols_matrix, datatype = 'vol', interval = interval)

def alpha_beta(y_rets, x_rets, weight = False, decay = 1.0, info_type = None):
    #ensure the y_rets and x_rets have the same time series length
    rets = pd.concat([y_rets,x_rets],axis = 1) 
    rets.sort_index(axis = 0, ascending = False, inplace = True)
    n = len(rets)
    rets = rets.multiply(_w(decay, n), axis = 0)
    rets.dropna(axis = 0, inplace = True)
    #apply the linear regression
    if info_type == 'alpha_beta':
        import statsmodels.api as sm
        X = sm.add_constant(rets[x_rets.columns])
        mdl = sm.OLS(rets[y_rets.columns], X).fit()
        return mdl.summary()
    else:
        from sklearn.linear_model import LinearRegression
        mdl = LinearRegression().fit(rets[x_rets.columns], rets[y_rets.columns])
        return mdl
    
    
def histRet(rets, weight = True, decay = 0.97, ret_type = 'log_ret', interval = 5.0):
    n = len(rets)
    if weight:
        w = _w(decay, n)
    else:
        w = _w(1, n)
    rets.sort_index(axis = 0, ascending = False,inplace = True)
    w = pd.Series(data = w, index = rets.index)
    histrets = np.array([_mean(rets[X], w) for X in rets.columns])
    histrets = pd.Series(data = histrets, index = rets.columns)
    return annualize(histrets, datatype = ret_type, interval = interval)
        
    
def corrMatrix(rets,weight = True, decay = 0.97):
    n = len(rets)
    rets.sort_index(axis = 0, ascending = False,inplace = True)
    if weight:
        w = _w(decay, n)
        wCorr = [[_corr(rets[X],rets[Y], w) for X in rets] for Y in rets]
        wCorr = pd.DataFrame(data = wCorr, index = rets.columns, columns = rets.columns)
        return wCorr
    else:
        return rets.corr()

def cholMatrix(corrMatrix):
    index = corrMatrix.index
    cholMx = scipy.linalg.cholesky(corrMatrix.as_matrix(), lower=True)
    return pd.DataFrame(data = cholMx, index = index, columns = index)
    
def corrArray(minorRet, majorRet, weight = True, decay = 0.97):
    #ensure the minorRet and majorRet have the same time series length
    rets = pd.concat([minorRet,majorRet],axis = 1) 
    rets.sort_index(axis = 0, ascending = False, inplace = True)
    n = len(rets)
    if weight == False:
        decay = 1
    w = _w(decay, n)
    wCorr = [[_corr(rets[X],rets[Y], w) for Y in minorRet.columns] for X in majorRet.columns]
    wCorr = pd.DataFrame(data = wCorr, index = majorRet.columns, columns = minorRet.columns)
    return wCorr  
 
def wMinorMajor(minorRet, majorRet, weight= True, decay = 0.97):
    majorMatrix = corrMatrix(majorRet, weight = weight, decay = decay)
    minorArray  = corrArray(minorRet, majorRet, weight = weight, decay = decay)
    
    #extract the major and minor names:
    majorIndex = minorArray.index
    minorIndex = minorArray.columns
    index = np.append(majorIndex,minorIndex)
    
    majorChol   = cholMatrix(majorMatrix)
    '''Apply the Tikhonov regulariztion to ensure the quality of w'''
    w = lsmr(majorChol.as_matrix().T, minorArray.values.T) 
    w_temp = np.dot(minorArray.values.T, inv(majorChol.as_matrix().T))[0]
    resid = np.sqrt(1 - np.sum(w**2))
    w = np.append(w, resid)
    return pd.DataFrame(data = w, index = index, columns = minorIndex)


'''    
def main(argv):
    #test config
    from datetime import datetime as dt
    start = dt(2017,4,10)
    end   = dt(2018,4,10)
    #stocks = ['SPY', 'USO', 'EEM', 'AGG', 'GLD']
    stocks = ['SPY']
    stocks_data = market_data.api2df(stocks = stocks,start = start, end = end)
    minor = ['VIXY']
    minor_data = market_data.api2df(stocks = minor, start = start, end = end)
    #stocks_data = pdr.get_data_yahoo('FB')
    print(stocks_data)
    ret = market_data.price2ret(stocks_data,interval = 5, overlap = False, ret_type = 'log_ret')
    minorRet = market_data.price2ret(minor_data, interval = 5, overlap = False, ret_type = 'log_ret')
    alpha_beta_info = alpha_beta(minorRet, ret, weight = False, decay = 1.0)#, info_type = 'alpha_beta')
    print(alpha_beta_info.predict(np.log(-0.01+1)))
    
    print(wMinorMajor(minorRet, ret, weight= False))
    
    print(ret)
    print(corrMatrix(ret, weight = False))
    print(corrArray(minorRet, ret, weight = False))
    print(histVol(ret, weight = False))
    print(histRet(ret, weight = False))  
    print(cholMatrix(corrMatrix(ret, weight = False)))
    print(wMinorMajor(minorRet, ret, weight= False))
    print()
if __name__ == "__main__":
    main(sys.argv[1:])
'''