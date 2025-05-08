# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 21:02:49 2018

@author: huang
"""
import sys
from numpy.linalg import inv
import numpy as np
import pandas as pd
import matrix as mx
#from cvxopt import matrix 
#from cvxopt.bias import dot 
#from cvxopt.solvers import qp, options
import market_data
from scipy.optimize import minimize
from sklearn import linear_model

'''
def _constraints(n):
    # nxn matrix of 0s
    G = matrix(0.0, (n,n))
    # Convert G to negative identity matrix
    G[::n+1] = -1.0
    # nx1 matrix of 0s
    h = matrix(0.0, (n,1))
    # 1xn matrix of 1s
    A = matrix(1.0, (1,n))
    # scalar of 1.0
    b = matrix(1.0)   
    
    return G, h, A, b
'''
    
'''    
def mvOpt(ret, corr, vol, targetRet, rf = 0.0):
    #input:
    #    ret       - pandas Series, index are ticker names
    #    corr, vol - pandas DataFrame;
    #output:
    #    asset_weight - pandas series
    
    n = corr.shape[0]
    corr = matrix(corr.as_matrix())
    vol  = matrix(vol.as_matrix())
    excess_ret  = matrix(ret.values - rf)
    cov  = dot(dot(vol, corr),vol)
    
    #constraints
    G, h , A, b = _constraints(n)
    options['show_progress'] = False

    w = qp(targetRet*cov, -excess_ret, G, h, A, b)['x']
    
    return w
'''
def bisect(func, low, high, tol):
    def _samesign(low, high):
        return low * high >0
    
    assert not _samesign(func(low), func(high))
    mid = None 
    while(not mid or np.abs(func(mid))>tol):
        mid = (low + high) / 2.0
        if _samesign(func(low),func(mid)):
            low = mid
        else:
            high = mid
            
    return mid
    
def mvOpt(ret, corr, vol, targetRet, rf = 0.0, longOnly = True):
    ticker = ret.index
    n = corr.shape[0]
    corr = corr.as_matrix()
    vol  = vol.as_matrix()
    excess_ret  = ret.values - rf
    cov  = np.dot(np.dot(vol, corr),vol)    
    def _objective(w):
        return 1/2 * np.dot(w,np.dot(cov, w.T))
    
    def _constraint1(w):
        return np.dot(excess_ret, w.T) -targetRet
    
    def _constraint2(w):
        return np.sum(w) - 1
    
    w0 = np.array([1.0]*n).T
    con1 = {'type':'eq','fun':_constraint1}
    con2 = {'type':'eq','fun':_constraint2}
    cons = [con1,con2]
    if longOnly:
        b = (0,None)
        bnds = (b,)*n
    else:
        bnds = None 
    w_solve = minimize(_objective,w0,method = 'SLSQP', constraints = cons, bounds = bnds)['x']
    w_solve = pd.Series(data = w_solve, index = ticker)
    return w_solve

'''
def mktImpCov(indRets, mktRets):
    reg = linear_model.LinearRegression()
    def _residVariance(mktRets, indRet):
        
        result = reg.fit(mktRets, indRet)
        return result
    
    [_residVariance(mktRets, indRet[ind]) for ind in indRets.columns]
    return 
'''
def mktImp(indRet, mktRets, decay = 0.97, weight = True):
    reg = linear_model.LinearRegression()
    rets = pd.concat([indRet, mktRets], axis = 1)
    rets = rets.dropna()
    mktRetNames = rets.columns[1:]
    indRetNames = rets.columns[0]
    reg.fit(rets[mktRetNames], rets[indRetNames])
    
    '''output the result'''
    n = len(rets)
    if weight:
        w = np.array([decay**i for i in range(n)])
    else:
        w = np.array([1.0]*n)
    mktEstRet = mx._mean(rets[mktRetNames],w)
    
    return reg

    
def blRet(tau, histCov, pi, P, omega, Q):
    '''Black-Litterman returns estimation
    input:
        histCov: n by n; P: m by n (m points of view); omega: m by m; Q: m by 1
    '''
    
    histCov = histCov.as_matrix()
    P = P.as_matrix()
    omega = omega.as_matrix()
    
    def _blCov(tau, histCov, P, omega):
        return inv(inv(tau*histCov)+P.T.dot(inv(omega)).dot(P))
    
    def _blView(tau, histCov, pi, P, omega, Q):
        return tau*inv(histCov).dot(pi) + P.T.dot(inv(omega)).dot(Q)
    
    return np.dot(_blCov(tau, histCov, P, omega), _blView(tau, histCov, pi, P, omega, Q))
        
    
def main(argv):
    '''test config'''
    from datetime import datetime as dt
    start = dt(2014,1,1)
    end   = dt(2017,1,1)
    stocks = ['SPY', 'AMZN', 'GOOG', 'NFLX', 'GLD']
    stocks_data = market_data.api2df(stocks = stocks,start = start, end = end)
    minor = ['FB']
    minor_data = market_data.api2df(stocks = minor, start = start, end = end)
    #stocks_data = pdr.get_data_yahoo('FB')
    print(stocks_data)
    ret = market_data.price2ret(stocks_data,interval = 5, overlap = False, ret_type = 'log_ret')
    minorRet = market_data.price2ret(minor_data, interval = 5, overlap = False, ret_type = 'log_ret')
    print(ret)
    corr = mx.corrMatrix(ret)
    print(mx.corrArray(minorRet, ret))
    vol = mx.histVol(ret)
    histRet = mx.histRet(ret, weight = False)
    targetRet  = 0.25
    print(histRet)
    print(mvOpt(histRet, corr, vol, targetRet, rf = 0.0, longOnly = True))
    
    '''test Estimation'''
    mkts = ['SPY']
    inds = ['GOOG','NFLX','AMZN']
    mkts_data = market_data.api2df(stocks = mkts,start = start, end = end)
    inds_data = market_data.api2df(stocks = inds,start = start, end = end)
    mkts_rets = market_data.price2ret(mkts_data,interval = 5, overlap = False, ret_type = 'log_ret')
    inds_rets = market_data.price2ret(inds_data,interval = 5, overlap = False, ret_type = 'log_ret')
    mImp = mktImp(inds_rets['GOOG'], mkts_rets)
    

if __name__ == "__main__":
    main(sys.argv[1:])
    
    
    
    

    


    
    
    
        
    