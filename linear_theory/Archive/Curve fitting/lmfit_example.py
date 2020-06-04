# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 14:24:19 2019

@author: Yoshi
"""
#!/usr/bin/env python

from numpy import exp, linspace, pi, random, sign, sin

from lmfit import Parameters, fit_report, minimize

import matplotlib.pyplot as plt

def residual(pars, x, data=None):
    vals  = pars.valuesdict()
    amp   = vals['amp']
    per   = vals['period']
    shift = vals['shift']
    decay = vals['decay']

    if abs(shift) > pi/2:
        shift = shift - sign(shift)*pi
        
    model = amp * sin(shift + x/per) * exp(-x*x*decay*decay)
    
    if data is None:
        return model
    else:
        return model - data

if __name__ == '__main__':
    n    = 1001
    xmin = 0.
    xmax = 250.0
    x    = linspace(xmin, xmax, n)
    random.seed(0)
    
    p_true = Parameters()
    p_true.add('amp', value=14.0)
    p_true.add('period', value=5.46)
    p_true.add('shift', value=0.123)
    p_true.add('decay', value=0.032)
    
    sample_function = residual(p_true, x) 
    noise           = random.normal(scale=0, size=n)
    data            = sample_function + noise
    
    fit_params = Parameters()
    fit_params.add('amp', value=13.0)
    fit_params.add('period', value=2)
    fit_params.add('shift', value=0.0)
    fit_params.add('decay', value=0.02)
    
    out = minimize(residual, fit_params, args=(x,), kws={'data': data})
    
    output_function = residual(out.params, x)
    
    plt.plot(x, sample_function, marker='o')
    plt.plot(x, output_function, marker='x')
    
    print(fit_report(out))
