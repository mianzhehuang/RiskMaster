# -*- coding: utf-8 -*-
"""
Created on Sat Apr 07 20:28:20 2018

@author: SUN
"""

from scipy import integrate
from scipy.stats import norm
import numpy as np
from py_vollib.black import undiscounted_black
from py_vollib.black.implied_volatility import implied_volatility 
import py_vollib.black.greeks.analytical as analytical
import market_data
import matrix as mx
import datetime
import option_player
import pandas as pd
from matrix import alpha_beta

def eloss_riskneutral(K1 = None, sigma1 = None, K2 = None, sigma2= None, F = None, t = None):
    ''' expected loss = expected win'''
    return - undiscounted_black(F, K1, sigma1, t, 'p') + undiscounted_black(F, K2, sigma2, t, 'p')
    
def d1(F = None, K = None, sigma = None, t = None):
    return (np.log(F/K) + sigma ** 2 * t)/(sigma * np.sqrt(t)) 
    
def eloss_sigma(sigma = None, F = None, K1 = None, K2 = None, t = None):
    d1_K1 = d1(F = K1, K= F, sigma = sigma, t = t)
    d1_K2 = d1(F = K2, K = F, sigma = sigma, t = t)
    def _payoff(x):
        F_x = F * np.exp(x * sigma * np.sqrt(t) - 0.5*sigma ** 2 * t)
        if x >= d1_K1:
            return 0 
        elif (x < d1_K1) and (x > d1_K2):
            return F_x - K1
        else:
            return K2 - K1
        
    pdf = lambda x: _payoff(x)*norm.pdf(x)
    
    eloss, err =  integrate.quad(pdf, -np.inf, np.inf)
    
    return eloss

def eloss_simulation(paths = None, F = None, K1 = None, K2 = None, t = None):
    F_Ts = paths[-1]
    cum_loss = 0 
    for F_T in F_Ts:
        if F_T>=K1:
            cum_loss = cum_loss + 0
        elif (F_T<K1) and (F_T>K2):
            cum_loss = cum_loss + F_T - K1
        else:
            cum_loss =  cum_loss + K2 - K1
            
    eloss = cum_loss / len(F_Ts)
    
    return eloss


#print integrate.quad(norm.pdf, -np.inf, np.inf)

class CreditPutSpread:
    K1 = None 
    K2 = None
    sigma1 = None
    sigma2 = None
    r = 0 
    #tradedPrice = None
    #mktPrice = None
    quantity = None
    ticker = None
    t = None
    asOf = None
    writer = None
    sp_start = datetime.date(2008,1,1)
    sp_end   = datetime.date(2008,12,31)
    def __init__(self, K1 = None, sigma1 = None, K2 = None, sigma2= None, F = None, 
                 ticker = None, asOf = None, quantity =100.0, Maturity = None, 
                 path = r'D:\\Carrick Huang\\supwin\\report output\\'): 
        '''Config'''
        self.K1 = K1
        self.sigma1 = sigma1
        self.K2 = K2
        self.sigma2 = sigma2
        self.F  = F
        self.t = self.calcBusinessInt(asOf, Maturity)
        #self.mktPrice = mktPrice
        #self.tradedPrice = tradedPrice
        self.quantity = quantity
        self.ticker = ticker
        self.asOf = asOf
        self.writer = pd.ExcelWriter(path + r'CreditPutSpread_'+str(K1)+'_'+str(K2)+'_'+Maturity.strftime('%Y%m%d')+r'.xlsx')
        '''Config End'''
        
    def calcBusinessInt(self,asOf, Maturity):
        return len(pd.date_range(asOf, Maturity))/260.0
     
    def lossP(self):
        lossP_dt, lossP_T = option_player.lossP(flag = 'p', F = self.F, K = self.K1, t = self.t, r = 0, \
                                                sigma = self.sigma1, dt = 1.0/252, convergentTime = 100000)
        
        return lossP_dt, lossP_T
    
    def vega(self):
        vega1 = analytical.vega('p',self.F, self.K1, self.t, self.r, self.sigma1)
        vega2 = analytical.vega('p',self.F, self.K2, self.t, self.r, self.sigma2)
        
        return (vega1, vega2)
    
    def sigma_ret_sp(self, interval = 1.0, overlap = True, ret_type = 'log_ret'):
        '''Calculate the historical stress period volatility and average return'''
        price_sp = market_data.api2df(stocks = [self.ticker],start = self.sp_start, end = self.sp_end)
        rets_sp  = market_data.price2ret(price_sp, interval = interval,overlap = overlap, ret_type = ret_type)       
        sigma_sp = mx.histVol(rets_sp, weight = False, interval = interval)
        ret_sp   = mx.histRet(rets_sp, weight = False, ret_type = ret_type, interval = interval)
        
        return sigma_sp.values[0][0], ret_sp.values[0]
           
    
    def eloss_sp(self, method = 'SpNormal', **kwargs):
        '''
        KWARGS:
            For method 'SpNormal', the inputs are interval, overlap, and ret_type
        Methods:
            1. SpNormal: return by stressed period historcal return, sigma by stressed period historical sigma
        '''
        if method == 'SpNormal':
            sigma_sp, ret_sp = self.sigma_ret_sp(interval = kwargs['interval'], overlap = kwargs['overlap'], ret_type = kwargs['ret_type'])
            F_sp = self.F * np.exp(ret_sp * self.t)
            eloss_sp = eloss_sigma(sigma = sigma_sp, F = F_sp, K1 = self.K1, K2 = self.K2, t = self.t)
            return eloss_sp
        else:
            return None
        
    def eloss_rn(self):
        '''Calculate the risk neutral expected loss'''
        return eloss_riskneutral(K1 = self.K1, sigma1 = self.sigma1, K2 = self.K2, sigma2= self.sigma2, F = self.F, t = self.t)
        
    def eloss_realized(self, dt = 1.0/252, sigma =None):
        t_step = 0
        eloss = 0 
        drop = (self.F - self.K1)/self.F
        if sigma == None:
            sigma = self.sigma1
        p = 0    
        while(t_step <= self.t):
            t_step = t_step + dt 
            '''Calculate the realized loss when doing the rolling'''
            loss = self.sensitive_underlying(vol_ticker = 'VIXY', start = datetime.date(2017,4,10), end = self.asOf, 
                                 drop = drop, interval = t_step * 252, overlap = True, ret_type = 'log_ret')
            
            '''Calculate the probability for the underlying touching strike in t_step'''
            lossP_dt, lossP_T = option_player.lossP(flag = 'p', F = self.F, K = self.K1, t = t_step, r = 0, \
                                                sigma = sigma, dt = 1.0/252, convergentTime = 100000)
            
            eloss = eloss + loss * lossP_T
            p = p + lossP_T
            
        return eloss
        
    def sensitive_underlying(self, vol_ticker = None, start = None, end = None, drop = 0.01, interval = 1.0, overlap = True, ret_type = 'log_ret'):
        dt = interval/252
        '''Sensitive drop underlying for the potential loss of CSP'''
        change_underlying = np.log(1 - drop)
        F_t1 = self.F * (1 - drop)
        '''Step 1: Caculate sensitive drop 1% underling for the increase of sigma '''
        price = market_data.api2df(stocks = [self.ticker, vol_ticker],start = start, end = end)
        rets  = market_data.price2ret(price, interval = interval, overlap = overlap, ret_type = ret_type)
        mdl = alpha_beta(pd.DataFrame(rets[vol_ticker]), pd.DataFrame(rets[self.ticker]), weight = False, decay = 1.0, info_type = None)
        change_sigma = mdl.predict(change_underlying)
        sigma1_t1 = self.sigma1 * np.exp(change_sigma)
        sigma2_t1 = self.sigma2 * np.exp(change_sigma)
        
        '''Step 2: Calculate the Credit Put Spread price based on the new underlying price and sigma'''
        return undiscounted_black(F_t1, self.K2, sigma2_t1, self.t - dt, 'p') - undiscounted_black(F_t1, self.K1, sigma1_t1, self.t - dt, 'p')
    
    def report(self, 
               index = ['Risk neutral probability of loss by holding the CPS to maturity',
                        'Risk neutral probability of loss by touching the strike in one day',
                        'Risk neutral expected loss',
                        'Expected loss in stressed period',
                        'Expected loss by 1% drop of underlying in one day',
                        'Expected loss by dropping to strike in one day']):
    
        '''Report the signal in excel report with contents:
            1. Risk neutral probability of loss by holding the CPS to maturity
            2. Risk neutral probability of loss by touching the strike in one day
            3. Risk neutral expected loss 
            4. Expected loss in stressed period
            5. Expected loss by 1% drop of underlying in one day
            6. Expected loss by dropping to strike in one day
        '''
        
        '''Step 1: Construct the report in excel'''
        index = index
        report = pd.DataFrame(index = index, columns = [['Signal']])
        
        '''Step 2: Compute the each signal'''
        if ('Risk neutral probability of loss by holding the CPS to maturity' in index):
            lossP_dt, lossP_T = self.lossP()
            report.loc['Risk neutral probability of loss by holding the CPS to maturity', 'Signal'] = lossP_T
        
        if ('Risk neutral probability of loss by touching the strike in one day' in index):
            lossP_dt, lossP_T = self.lossP()
            report.loc['Risk neutral probability of loss by touching the strike in one day', 'Signal'] = lossP_dt
        
        if ('Risk neutral expected loss' in index):
            report.loc['Risk neutral expected loss', 'Signal'] = self.eloss_rn() * self.quantity
        
        if ('Expected loss in stressed period' in index):
            report.loc['Expected loss in stressed period', 'Signal'] = \
            self.eloss_sp(method = 'SpNormal', interval = 5, overlap = False, ret_type = 'log_ret') * self.quantity
            
        if('Expected loss by 1% drop of underlying in one day' in index):
            report.loc['Expected loss by 1% drop of underlying in one day', 'Signal'] = \
            self.sensitive_underlying(vol_ticker = 'VIXY', start = datetime.date(2017,4,10), end = self.asOf, 
                                 drop = 0.01, interval = 1.0, overlap = True, ret_type = 'log_ret') * self.quantity
            
        if('Expected loss by dropping to strike in one day' in index):
            drop = (self.F - self.K1)/self.F
            report.loc['Expected loss by dropping to strike in one day', 'Signal'] = \
            self.sensitive_underlying(vol_ticker = 'VIXY', start = datetime.date(2017,4,10), end = self.asOf, 
                                 drop = drop, interval = 1.0, overlap = True, ret_type = 'log_ret') * self.quantity
            
        report.to_excel(self.writer,self.asOf.strftime('%Y%m%d')+'_F'+str(self.F)+'_Vol'+str(self.sigma1)+'_'+str(self.sigma2))
        self.writer.save()
        
        return report
                
        
class CreditPutSpreadStrats(CreditPutSpread):
    aum = None
    marginPercent = None
    tradedPrice = None
    size = None
    
    def __init__(self, K1 = None, sigma1 = None, K2 = None, sigma2= None, F = None, 
                 ticker = None, t = None, asOf = None, quantity =100.0, Maturity = None, 
                 path = r'D:\\Carrick Huang\\supwin\\report output\\', 
                 aum = None, marginPercent = None, tradedPrice = None, broker = 'IB'):
        
        super().__init__(K1 = K1, sigma1 = sigma1, K2 = K2, sigma2= sigma2, F = F, 
                         ticker = ticker, t = t, asOf = asOf, quantity = quantity, Maturity = Maturity, 
                         path = path)
        self.writer = pd.ExcelWriter(path + r'CreditPutSpread_Margin'+str(marginPercent) +'_' \
                                     +str(K1)+'_'+str(K2)+'_'+Maturity.strftime('%Y%m%d')+r'.xlsx')
        self.aum = aum
        self.marginPercent = marginPercent
        
        '''Update the tradedPrice:
            If the tradedPrice is None, the tradedPrice is the mark-to-market price.
        '''
        if self.tradedPrice == None:
            self.tradedPrice = -self.eloss_rn()
        else:    
            self.tradedPrice = tradedPrice
        
        '''Update the AUM according the contracts we sell'''
        self.size = round(self.aum * self.marginPercent/self.margin(broker = broker))
        self.aum  = aum + self.tradedPrice * self.quantity * self.size 
        
    def margin(self, broker = 'IB'):
        if broker == 'IB':
            return max((self.K1 - self.K2),0) * self.quantity
    
    def report(self, 
               index = ['Size of contracts',
                        'Risk neutral probability of loss by holding the CPS to maturity',
                        'Risk neutral probability of loss by touching the strike in one day',
                        'Risk neutral expected loss',
                        'Expected loss in stressed period',
                        'Expected loss by 1% drop of underlying in one day',
                        'Expected loss by dropping to strike in one day']):
    
        '''Report the signal in excel report with contents:
            1. Risk neutral probability of loss by holding the CPS to maturity
            2. Risk neutral probability of loss by touching the strike in one day
            3. Risk neutral expected loss 
            4. Expected loss in stressed period
            5. Expected loss by 1% drop of underlying in one day
            6. Expected loss by dropping to strike in one day
        '''
        
        '''Step 1: Construct the report in excel'''
        index = index
        report = pd.DataFrame(index = index, columns = [['Signal']])
        
        '''Step 2: Compute the each signal'''
        if ('Size of contracts' in index):
            report.loc['Size of contracts', 'Signal']= self.size
            
        if ('Risk neutral probability of loss by holding the CPS to maturity' in index):
            lossP_dt, lossP_T = self.lossP()
            report.loc['Risk neutral probability of loss by holding the CPS to maturity', 'Signal'] = lossP_T
        
        if ('Risk neutral probability of loss by touching the strike in one day' in index):
            lossP_dt, lossP_T = self.lossP()
            report.loc['Risk neutral probability of loss by touching the strike in one day', 'Signal'] = lossP_dt
        
        if ('Risk neutral expected loss' in index):
            report.loc['Risk neutral expected loss', 'Signal'] = \
            (self.tradedPrice + self.eloss_rn()) * self.quantity * self.size / self.aum
        
        if ('Expected loss in stressed period' in index):
            report.loc['Expected loss in stressed period', 'Signal'] = \
            (self.tradedPrice + self.eloss_sp(method = 'SpNormal', interval = 5, overlap = False, ret_type = 'log_ret'))\
            * self.quantity * self.size/self.aum
            
        if('Expected loss by 1% drop of underlying in one day' in index):
            report.loc['Expected loss by 1% drop of underlying in one day', 'Signal'] = \
            (self.tradedPrice +
            self.sensitive_underlying(vol_ticker = 'VIXY', start = datetime.date(2017,4,10), end = self.asOf, 
                                 drop = 0.01, interval = 1.0, overlap = True, ret_type = 'log_ret')) * self.quantity *self.size\
             /self.aum
            
        if('Expected loss by dropping to strike in one day' in index):
            drop = (self.F - self.K1)/self.F
            report.loc['Expected loss by dropping to strike in one day', 'Signal'] = \
            (self.tradedPrice +
            self.sensitive_underlying(vol_ticker = 'VIXY', start = datetime.date(2017,4,10), end = self.asOf, 
                                 drop = drop, interval = 1.0, overlap = True, ret_type = 'log_ret')) * self.quantity*self.size\
             /self.aum
            
        report.to_excel(self.writer,self.asOf.strftime('%Y%m%d')+'_F'+str(self.F)+'_Vol'+str(self.sigma1)+'_'+str(self.sigma2))
        self.writer.save()
        
        return report
        
        
    
        
#print(undiscounted_black(100, 92, 0.2, 1.0/12, 'p') - undiscounted_black(100, 95, 0.2, 1.0/12, 'p'))    
#print(eloss_sigma(sigma = 0.2, F = 100, K1 = 95, K2 = 92, t = 1.0/12))    
#from option_player import bs_path 
#paths = bs_path(S0 = 100, mu = 0, sigma = 0.2, t = 1.0/12, dt = 1.0/252, convergentTime = 1000000)
#print(eloss_simulation(paths = paths, F = 100, K1 = 95, K2 = 92, t = 1.0/12))     
CPS = CreditPutSpread(K1 = 262, sigma1 = 0.1519, K2 = 254, sigma2 = 0.1832, F = 266.66, ticker = 'SPY', asOf = datetime.date(2018,4,20), Maturity = datetime.date(2018,5,18))
print(CPS.vega())
#print(CPS.eloss_rn())
#print(CPS.eloss_sp(method = 'SpNormal', interval = 5, overlap = False, ret_type = 'log_ret'))
#print(CPS.sensitive_underlying(vol_ticker = 'VIXY', start = datetime.date(2017,4,1), end = datetime.date(2018,4,1), drop = 0.01, interval = 1.0, overlap = True, ret_type = 'log_ret'))
#print(CPS.lossP())
#print(CPS.report())
'''
print(undiscounted_black(270.39, 262, 0.1474, 1.0/12, 'p'))
print(implied_volatility(0.735,270.39,254,0,1.0/12,'p'))
print('gamma:',analytical.gamma('p',270.39,254,1.0/12,0,implied_volatility(0.735,270.39,254,0,1.0/12,'p'))*100)
'''
print('delta:',analytical.delta('p',2,254,1.0/12 - 2.0/252,0,implied_volatility(0.735,270.39,254,0,1.0/12,'p'))*100)
'''
print('theta:',analytical.theta('p',270.39,254,1.0/12,0,implied_volatility(0.735,270.39,254,0,1.0/12,'p'))*100)
CPS_strats = CreditPutSpreadStrats(K1 = 253, sigma1 = 0.2018, K2 = 244, sigma2= 0.2310, F = 265, 
                 ticker = 'SPY', t = 1.0/12, asOf = datetime.date(2018,4,11), quantity =100.0, Maturity = datetime.date(2018,5,11), 
                 path = r'D:\\Carrick Huang\\supwin\\report output\\', 
                 aum = 20000, marginPercent = 1, tradedPrice = None, broker = 'IB')
print(CPS_strats.eloss_realized())
'''
#price = market_data.api2df(stocks = ['SPY','VIXY'],start = datetime.date(2011,1,1), end = datetime.date(2011,12,31))
#rets  = market_data.price2ret(price, interval = 1,overlap = False, ret_type = 'log_ret')
#print(price)
