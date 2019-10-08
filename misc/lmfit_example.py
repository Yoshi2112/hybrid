# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 17:24:21 2019

@author: Yoshi
"""
import numpy as np
import lmfit as lmf

def residual_exp(pars, t, data=None):
    vals   = pars.valuesdict()
    amp    = vals['amp']
    growth = vals['growth']

    model  = amp * np.exp(growth*t)
    
    if data is None:
        return model
    else:
        return model - data
    

def fit_magnetic_energy(by, bz):
    '''
    Calculates an exponential growth rate based on transverse magnetic field
    energy.
    '''
    bt       = np.sqrt(by ** 2 + bz ** 2) * 1e-9
  
    time_fit = np.arange(bt.shape[0])

    fit_params = lmf.Parameters()
    fit_params.add('amp'   , value=1.0            , min=None , max=None)
    fit_params.add('growth', value=0.001*cf.gyfreq, min=0.0  , max=None)
    
    fit_output      = lmf.minimize(residual_exp, fit_params, args=(time_fit,), kws={'data': U_B[:linear_cutoff]},
                               method='leastsq')
    fit_function    = residual_exp(fit_output.params, time_fit)

    fit_dict        = fit_output.params.valuesdict()
        
    return linear_cutoff, fit_dict['growth']