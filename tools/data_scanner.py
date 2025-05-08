# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 22:28:52 2019

@author: huang
"""
#%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
import pandas 
import numpy
from scipy import stats
import seaborn


def dataQualityCheck(data, outputPath=None, outlierLevel=1.5):
    data = data.drop_duplicates()
    dataType = pandas.DataFrame(data=data.dtypes, columns=['Data_Type'])
    countNull = pandas.DataFrame(data=data.isnull().sum(), columns=['Count_Null'])
    countZero = pandas.DataFrame(data=(data==0).sum(), columns=['Count_Zero'])
    summary = pandas.concat([dataType, countNull, countZero], axis=1)
    summary['Null_Ratio'] = summary['Count_Null']/len(data)
    summary['Zero_Ratio'] = summary['Count_Zero']/len(data)
    summary = pandas.concat([summary, detectOutlier(data, outlierLevel=outlierLevel), 
                             data.quantile([0, .025, .25, .5, .75, .975, 1]).T, 
                             pandas.DataFrame(data=data.std(),columns=['Std'])], axis=1)
    if not outputPath:
        summary.to_csv(outputPath + 'dataQualityCheckSummary.csv')
    return data, summary

def detectOutlier(data, outlierLeverl=1.5):
    IQR = data.quantile(.75) - data.quantile(.25)
    outlier = (data[IQR.index]<(data.quantile(.25) - outlierLevel*IQR)) \
             |(data[IQR.index]>(data.quantile(.75) + outlierLevel*IQR))
    return pandas.DataFrame(data=outlier.sum(), columns=['Count_Outliers'])

def differentiateCategory(data, yName=None, categoryName=None, countRestrict=0, figsize=(20,5), outputPath=None):
    countCategory = pandas.DataFrame(data.groupby(categoryName)[yName].count())
    countCategory.columns = ['Counts']
    meanCategory  = pandas.DataFrame(data.groupby(categoryName)[yName].mean())
    meanCategory.columns = ['Mean']
    summary = pandas.concat([countCategory, meanCategory], axis=1).sort_values(['Counts'], ascending=False)
    plotFile = summary[summary['Counts']>countRestrict].plot.bar(rot=0, subplots=True,figsize=figsize)
    figure  = plotFile[0].get_figure()
    if outputPath:
        pngName = 'differentiate_' + categoryName + '_' + yName + '.png' 
        figure.savefig(outputPath + pngName)
        
    return summary

def zeroImputation(data, columnList=None, replaceWtih='Median'):
    for column in columnList:
        if replaceWith == 'Median':
            data[column] = data[column].replace({0:data[column][data[column]>0].median()})
        elif replaceWith == 'Mean':
            data[column] = data[column].replace({0:data[column][data[column]>0].mean()})
        else:
            data[column] = data[column].replace({0:replaceWith})
            
    return data

def numToBinColumn(data, targetColumn, bins=None, labels=None, outputPath=None, figsize=(20,5)):
    if labels:
        data[targetColumn+'_binned'] = data.cut(data[targetColumn], bins=bins, labels=labels)
    else:
        data[targetColumn+'_binned'] = numpy.searchsorted(bins, df[targetColumn].values)
        
    historgram = data.groupby(pandas.cut(data[targetColumn], bins=bins)).size()
    plotFile = histogram.plot.bar(rot=0, figsize=figsize)
    figure  = plotFile.get_figure()
    if outputPath:
        pngName = 'Histogram_' + targetColumn + '.png' 
        figure.savefig(outputPath + pngName)
        
    return data
    

