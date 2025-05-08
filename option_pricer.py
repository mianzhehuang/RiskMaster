# -*- coding: utf-8 -*-
"""
Created on Sun May  3 13:00:48 2020

@author: huang
"""

from py_vollib.black.greeks.analytical import delta, gamma,vega, theta
from py_vollib.black.implied_volatility import implied_volatility
import datetime
import pandas as pd



options_all = pd.read_excel(r"D:\personal doc\AF_20200508.xlsx", sheet_name='Open Positions')
underlyings_all = pd.read_excel(r"D:\personal doc\AF_20200508.xlsx", sheet_name='Dashboard')
def get_delta_gamma(today):
    #today = 43950
    options = options_all[options_all['Date'] == today]
    options = options[options['Underlying'] != 'VXX']
    
    deltas = 0
    r = 0
    gammas = 0
    vegas = 0
    thetas = 0
    mats_temp = 0.0
    quantities = 0
    underlyings = {'SPY':underlyings_all[underlyings_all['Date']==today]['SPY'].values[0],
                  'XSP':underlyings_all[underlyings_all['Date']==today]['SPY'].values[0],
                  'SPX':underlyings_all[underlyings_all['Date']==today]['SPY'].values[0]*10}
    for i in options.index:
        K = options['strike'][i]
        quantity = options['Quantity'][i]
        mv = options['Market Value'][i]
        callput = options['Type'][i]
        t = (options['Expiration'][i] - today)/365.25
        u = options['Underlying'][i]
        if(callput == 'Put'): flag = 'p'
        if(callput == 'Call'): flag = 'c'
        F = underlyings[u]
        price = mv/ quantity / 100
        iv = implied_volatility(price, F, K, r, t, flag)
        delta_res = delta(flag, F, K, t, r, iv)
        gamma_res = gamma(flag, F, K, t, r, iv)
        vega_res = vega(flag, F, K, t, r, iv)
        theta_res = theta(flag, F, K , t, r, iv)
        deltas = delta_res * quantity*100 + deltas
        gammas = gamma_res * quantity*100 + gammas
        vegas = vega_res * quantity * 100 + vegas
        thetas = theta_res * quantity * 100 + thetas
        mats_temp = t*abs(quantity) + mats_temp#t * quantity + mats
        #print([mats_temp, t*abs(quantity)])
        quantities =  quantities + abs(quantity)
        
    #print("Delta is: " + str(deltas) + "\n" + "Gamma is: " + str*(gammas))
    if (quantities == 0): 
        mats_res = 0
    else:
        mats_res = mats_temp * 1.0/quantities
        
    return deltas, gammas, vegas, thetas, mats_res * 365.25

days = underlyings_all['Date'][~pd.isna(underlyings_all['Date'])]

delta_list = []
gamma_list = []
vega_list = []
day_list = []
mat_list = []
theta_list = []
for today in days:
    day_list.append(datetime.datetime.fromordinal(int(datetime.datetime(1900, 1, 1).toordinal() + today - 2)))
    deltas, gammas, vegas, thetas, mats = get_delta_gamma(today)
    delta_list.append(deltas)
    gamma_list.append(gammas)
    vega_list.append(vegas)
    theta_list.append(thetas)
    mat_list.append(mats)
    print("Delta is: " + str(deltas) + ";" + "Gamma is: " + str(gammas))

data = {'Date': day_list,
        'Delta': delta_list,
        'Gamma': gamma_list,
        'Vega': vega_list,
        'Theta': theta_list,
        'Mats': mat_list}
res = pd.DataFrame.from_dict(data)

res = res.set_index('Date')
res['Delta'].plot(title='Delta of Portfolio')
res['Gamma'].plot(title='Gamma of Portfolio')
res['Vega'].plot(title = 'Vega of Portfolio')
res['Theta'].plot(title = 'Theta of Portfolio')
res.to_excel(r"D:\personal doc\delta_gamma_20200508.xlsx")