# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 21:46:19 2018

@author: huang
"""

import matrix as mx
import market_data
import seaborn as sns
import numpy as np
import datetime 
import matplotlib.pyplot as plt

def plot_heatMap(corr, paths, start = None, end = None, weight = None, decay = None, save = False):
    heatMap = sns.heatmap(corr, xticklabels = corr.columns, yticklabels = corr.columns, cmap='coolwarm')#,cmap="YlGnBu")
    if weight:
        pngname = "weight_" + str(decay) + start.strftime('%Y%m%d')+'_'+ end.strftime('%Y%m%d')+ ".png"
    else:
        pngname = "equalWeight" + start.strftime('%Y%m%d')+'_'+ end.strftime('%Y%m%d')+  ".png"
            
        #heatMap.savefig(self.outPath + "corrHeatMap_" + pngname)
    figure = heatMap.get_figure()
    if save: figure.savefig(paths + "corrHeatMap_" + pngname)

    
class EtfPortfolioSummary(object):
    '''input'''
    start = None
    end = None
    rets = None 
    kwargs = None
    outPath = None
    
    '''output'''
    corr  = None
    wCorr = None
    
    def __init__(self, outPath, etfs, start, end, **kwargs):
        #'''kwargs: interval = 5, overlap = False, ret_type = log_ret'''
        prcs = market_data.api2df(stocks = etfs, start = start, end = end)
        self.rets = market_data.price2ret(prcs, **kwargs)
        self.start = start
        self.end = end
        self.outPath = outPath
        self.kwargs = kwargs
    
    def subRets(self, start, end):
        sub_date = pd.bdate_range(start,end,freq = '1B')
        rets = self.rets[self.rets.index.isin(sub_date)]
        return rets
        
    def corrMatrix(self, weight = None, decay = None, save = False):
        corr =  mx.corrMatrix(self.rets, weight, decay)
        plot_heatMap(corr, self.outPath, start = self.start, end = self.end, weight = weight, decay = decay, save = save)
        return corr
    
    def tsHistVols(self, yearInterval = 3, weight = None, decay = None, save = False):
        end = datetime.date(self.start.year + yearInterval, self.start.month, self.start.day)
        tsVols = pd.DataFrame()
        while (end<=self.end):
            start = datetime.date(end.year - yearInterval, end.month, end.day)
            temp = self.histVol(start = start, end = end, weight = weight, decay = decay)
            tsVols = pd.concat([tsVols, temp])
            end = datetime.date(end.year + 1, end.month, end.day)
        
        if weight:
            pngname = "weight_" + str(decay) +self.start.strftime('%Y%m%d')+'_'+self.end.strftime('%Y%m%d')+ ".png"
        else:
            pngname = "equalWeight_" +self.start.strftime('%Y%m%d')+'_'+self.end.strftime('%Y%m%d')+  ".png"
        
        tsPlot  = tsVols.plot(title = 'Historical Volatility Change')
        figure  = tsPlot.get_figure()
        if save: figure.savefig(self.outPath + "tsHistVols_" + pngname)
        return tsVols
        
    def histVol(self, start = None, end = None, weight = None, decay = None):
        if not (start and end):
            rets = self.rets
        else:
            if not start: start = self.start 
            if not end: end = self.end 
            rets = self.subRets(start, end)
            
        histVol = mx.histVol(rets, weight = weight, decay = decay, interval = self.kwargs['interval'])
        histVol =  pd.DataFrame(data = np.diag(histVol), index = histVol.index, columns = [end]).T
        return histVol
    

outPath = 'C:\\Users\\huang\\Dropbox\\Carrick Huang\\supwin\\report output\\'
startDate = datetime.date(2010,1,1)
endDate   = datetime.date(2018,1,1)
supwinPortfolioReport = EtfPortfolioSummary(outPath, ['SPY', 'USO', 'UNG', 'IYR'], startDate, endDate, interval = 21, overlap = False, ret_type = 'log_ret')  
#print(supwinPortfolioReport.rets)
print(supwinPortfolioReport.corrMatrix())

data=[[1.00,0.42,0.32,0.86,0.01],
      [0.42,1.00,0.20,0.58,-0.01],
      [0.32,0.20,1.00,0.36,0.00],
      [0.86,0.58,0.36,1.00,0.00],
      [0.01,-0.01,0.00,0.00,1.00]]

df = pd.DataFrame(data=data, 
                  index=['Managed Risk Premia','Managed Volatility','All Weather','SPX','China 50'],
                  columns=['Managed Risk Premia','Managed Volatility','All Weather','SPX','China 50'])
#print(supwinPortfolioReport.tsHistVols(save = True))