# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 02:37:31 2018

@author: huang
"""

import numpy as np

def reg_non_param(x, bdwidth, x_sample, y_sample, kernelName):
    """Values of the non-parametric regression of Y wrt X using a Gaussian kernel.

    Parameters
    ----------
    x: numpy array, one dimensional
        Values at which the regression is evaluated
    bdwidth: positive float, value of the bandwidth parameter
    x_sample: numpy array, one dimensional, non-empty
        x values of the sample
    y_sample: numpy array, one dimensional
        y values of the sample, must have the same length as x_sample.    
    """
    def kern(u, x, kernelName):
        """Gaussian kernel function"""
        if kernelName == 'gaussian':
            return np.exp(-(u[:, np.newaxis] - x) ** 2 / (2 * bdwidth ** 2))
        elif kernelName == 'triangular':
            return np.array([[1 - np.absolute(j) / bdwidth \
                              if np.absolute(j) < bdwidth else 0 for j in i] for i in (u[:, np.newaxis] - x) ])
    return np.sum(kern(x_sample, x, kernelName) * y_sample[:, np.newaxis], axis=0) \
        / np.sum(kern(x_sample, x, kernelName), axis=0)
        
def basis(knots, x):
    """Values of order-1 B-spline basis functions.
    
    For an increasingly sorted collection of knots and a collection of
    query points x, returns a 2-dimensional array of values, of dimension
    len(x) x len(knots).
    
    Parameters
    ----------
    knots: numpy array, one dimensional, increasingly sorted
        Knots of the B-spline function
    x: numpy array, one dimensional
        Query points where to evaluate the basis functions.
    """
    nb_knots = len(knots)
    diag = np.identity(nb_knots)
    res = np.empty((len(x), nb_knots))
    for i in xrange(nb_knots):
        res[:, i] = np.interp(x, knots, diag[i])
    return res
        
class VasicekModel:
    X = None
    Y = None
    method = 'OLS'
    rettype = None
    
    '''Calibration Result'''
    tsLongMean = None
    tsMeanRev = None
    tsVol = None
    def __init__(self, ts = None, method = 'OLS', rettype = 'Lognormal'):
        ts.sort_index(ascending = False, inplace = True)
        self.Y = ts.shift(-1).dropna()
        self.X = ts.shift(1).dropna()
        self.method = method
        self.rettype = rettype
        if rettype == 'Lognormal':
            self.Y = np.log(self.Y)
            self.Y = self.Y.fillna(self.Y.mean())
            self.X = np.log(self.X)
            self.X = self.X.fillna(self.X.mean())
            
        '''Get calibration result'''
        self.tsLongMean, self.tsMeanRev, self.tsVol = self.getCalibration()
           
    def getCalibration(self):
        if self.method == 'OLS':
            tsLongMean, tsMeanRev, tsVol = self.getCalibOLS()
            
        return tsLongMean, tsMeanRev, tsVol 
    
    def getCalibOLS(self):
        deltaT = 1.0/260.0
        n = len(self.Y)
        Sx = np.sum(self.X.values)
        Sy = np.sum(self.Y.values)
        Sxx = np.sum(self.X.values**2)
        Syy = np.sum(self.Y.values**2)
        Sxy = np.sum(self.X.values * self.Y.values)
        
        b1 = (n*Sxy - Sx*Sy)/(n*Sxx - Sx**2)
        if b1>1.0:
            b1 = 0.99
            b0 = 0.0
        else:
            b0 = (Sy - b1*Sx)/n
            
        tsSampleS = np.sqrt((n*Syy-Sy**2-b1*(n*Sxy-Sx*Sy))/(n*(n-2)))
        tsMeanRev = -np.log(b1)/deltaT
        tsLongMean = b0/(1-b1)
        tsVol = tsSampleS * np.sqrt((-2*np.log(b1))/(deltaT*(1-b1**2)))
        
        return tsLongMean, tsMeanRev, tsVol
    
    def project(self, path = 2, T = 21/260.0):
        paths = []
        for i in range(path):
            paths.append(self.onepathGenerate(T = T))
            
        return np.array(paths)
    
    def onepathGenerate(self, T = 21/260.0):
        import random
        i = 0
        tsProject = list(self.Y.ix[0].values)
        deltaT = 1/260.0
        n = int(T/deltaT)
        for i in range(n):
            brownian = random.normalvariate(0,1)
            ts_new = tsProject[i] + self.tsMeanRev * deltaT * (self.tsLongMean - tsProject[i]) \
                     + self.tsVol * brownian * deltaT
            tsProject.append(ts_new)
            
        if self.rettype == 'Lognormal':
            return list(np.exp(tsProject))
            
        return tsProject
            
        
        