# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 01:00:31 2018

@author: huang
"""
import matrix as mx
import pandas as pd
class StratsReport:
    strats_ret = None 
    universe_ret = None
    def __int__(self, strats_ret, universe_ret):
        self.strats_ret = strats_ret
        self.universe_ret = universe_ret
        
    def matrix(self,weight = False, decay = 0.97):
        rets = pd.concat([self.strats_ret,self.universe_ret], axis = 1)
        return 
        