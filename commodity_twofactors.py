# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 22:34:17 2018

@author: huang
"""

def _term_vol(T, kappa, h1, h2, hinf, a, d, T1, T2):
    _a = (h1**2 + h2**2)*np.exp(2*d(T))*(
        (np.exp(-2*kappa*(T-T2)) - np.exp(-2*kappa*(T-T1)))
        /(2*kappa*(T2-T1))
        )
    _b = 2*hinf*h1*np.exp(d(T))*(
            (np.exp(-kappa*(T-T2)) - np.exp(-kappa*(T-T1)))
            /(kappa *(T2-T1))
        )
    _c = hinf**2
    return np.exp(2*a(T)) * (_a + _b + _c)

