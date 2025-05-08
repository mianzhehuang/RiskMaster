# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 18:51:38 2018

@author: huang
"""

from py_vollib.black.greeks.analytical import delta as delta_func
import optimization as opt
import scipy.stats as stats
import sys
import numpy as np
import matplotlib.pyplot as plt
import market_data
from scipy.stats import norm


def calMoneynessOnDelta(delta = None, flag = None, F = 100, t = None, r = None, sigma = None, calRange =[.5,.5]):
    '''
    Inputs:
        calRange: a pre-set range for delta searching based on Bisection method;
                  It is a relative percentage around the F price.
    '''
    tol = 10**(-6)
    def _deltaObjFunc(K):
        return delta_func(flag, F, K, t, r, sigma) - delta
    
    K = opt.bisect(_deltaObjFunc, F*(1-calRange[0]), F*(1+calRange[1]), tol)
    
    return np.log(F/K)/(sigma * np.sqrt(t))


def bs_path(S0 = None, mu = None, sigma = None, t = None, dt = 1.0/252, convergentTime = 2000):
    """Simulate geometric Brownian motion
    input:
        S0: positive float starting value
        mu: float drift
        sigma: float volatility
        t: positive float maturity
        dt: grid distance, default as per day
        convergentTime: Monte Carlo simulation times
    """
    nb_step = round(t/dt)
    brownian = np.empty((nb_step + 1, convergentTime))
    brownian[0] = 0.0
    brownian[1:] = np.cumsum(np.random.randn(nb_step, convergentTime), axis=0) *\
                   np.sqrt(dt)

    return S0 * np.exp(sigma * brownian + (mu - 0.5 * sigma ** 2) *\
                       np.linspace(0.0, t, nb_step + 1)[:, np.newaxis])    


#def path_boostrapping(S0 = None, ticker = None, period_start = None, period_end = None, t = None, dt = 1.0/252, overlap = True, ret_type = 'log_ret', convergentTime = 100000):
    #'''Step 1: get the price and return data in specific period'''
    #price_period = market_data.api2df(stocks = ticker,start = period_start, end = period_end)
    #rets_period  = market_data.price2ret(price_period, interval = int(252/dt),overlap = overlap, ret_type = ret_type)
    
    
    
def interpretDelta(delta = None, flag = None, F = 100, t = None, r = None, sigma = None, calRange = [.5,.5], convergentTime = 100000):
    '''Step 1: Calculate the corresponding Strike'''
    moneyness = calMoneynessOnDelta(delta = delta, flag = flag, F = F, t = t, r = r, sigma = sigma, calRange = calRange)
    K = F * np.exp(- moneyness * sigma * np.sqrt(t))
    
    '''Step 2: Calculate the percentange of changing between ITM and OTM based on simulation'''
    #paths = gbmSim(S0 = F, mu = 0, sigma = sigma, t = t, dt = 1.0/252, convergentTime = convergentTime)
    lossP_t, lossP_T = lossP(flag = flag, F = F, K = K, t = t, r = r, sigma = sigma, dt = 1.0/252, convergentTime = convergentTime)
    return lossP_t, lossP_T

def lossP(flag = None, F = None, K = None, t = None, r = None, sigma = None, dt = 1.0/252):
    def alpha(t):
        return (np.log(K/F) + 0.5*sigma ** 2 * t)/(sigma * np.sqrt(t))
    lossP_T = norm.cdf(alpha(t))
    
    
    lossP = 0 
    t_step = 0
    while(t_step <= t):
        t_step = t_step + dt
        lossP = lossP + norm.cdf(alpha(t_step))*(1-lossP)
        
    return lossP, lossP_T
    
def lossP_path(flag = None, F = None, K = None, t = None, r = None, sigma = None, dt = 1.0/252, convergentTime = 100000):
    paths = bs_path(S0 = F, mu = 0, sigma = sigma, t = t, dt = dt, convergentTime = convergentTime)
    F_T = paths[-1]
    if F>K:
        lossP =  (np.min(paths, axis = 0)<=K).sum()/convergentTime
        lossP_T = (F_T<=K).sum()/convergentTime
    else:
        lossP = (np.max(paths, axis = 0)>K).sum()/convergentTime
        lossP_T = (F_T>K).sum()/convergentTime
    
    return lossP, lossP_T    
    
def main(argv):
    '''Config'''
    flag = 'p'
    F = 266.58
    t = 1.0/12 - 7.0/252
    K = 243
    r = 0
    sigma = 0.20
    delta = -0.2
    price = 0.34
    '''Config End'''
    from py_vollib.black.implied_volatility import implied_volatility
    print(implied_volatility(price,F,K,r,t,'p'))
    print(delta_func(flag, F, K, t, r, implied_volatility(price,F,K,r,t,'p')))
    '''
    moneyness = calMoneynessOnDelta(delta = delta, flag = flag, F = F, t = t, r = r, sigma = sigma)
    print(moneyness)
    K  = F * np.exp( - moneyness * sigma * np.sqrt(t)) 
    print(K)
    print(delta_func('p', F, K, t, 0, sigma))
    #print(delta_func('p', 265, 244, 1.0/12, 0, 0.2310))
    loss1, loss2 = lossP(flag = flag, F = F, K = K, t = t, r = 0, sigma = sigma, dt = 1.0/252)
    print(loss1, loss2)
    
    paths = gbmSim(S0 = F, mu = 0, sigma = sigma, t = t, dt = 1.0/252, convergentTime = 200000)
    print((paths.T[-1]<=K).sum()) 
    paths2 = bs_path(S0 = F, mu = 0, sigma = sigma, t = t, dt = 1.0/252, convergentTime = 200000)
    print((paths2[-1]<=K).sum())
    '''
    #lossP, lossP_T = interpretDelta(delta = delta, flag = flag, F = F, t = t, r = r, sigma = sigma, calRange = [.5,.5], convergentTime = 200000)
    
    #print(lossP, lossP_T)
    #print(_delta(12.50))
    '''
    K = opt.bisect(lambda K: _delta(K) + 0.2, F*(1-0.5), F*(1+0.5), 0.00001)
    print(K)
    print(_delta(K))
    #print(stats.norm.ppf(0.2, , rets.std()))
    target = (np.log(K/F) +  sigma**2/2 * t)/sigma
    print(target)
    
    def _alpha(alpha):
        return stats.norm.ppf(alpha, 0, t)
    
    print(_alpha(0.05))

    alpha = opt.bisect(lambda alpha: _alpha(alpha) - target, 0.001, 0.5, 0.00001)
    
    print(alpha)
    '''
if __name__ == "__main__":
    main(sys.argv[1:])