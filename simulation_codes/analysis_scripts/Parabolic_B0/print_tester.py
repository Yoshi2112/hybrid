# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 17:07:28 2020

@author: Yoshi
"""
import numpy as np

Nj          = 2
ccdens      = np.array([180.,  20.])
species_lbl = np.array(['$H^+$_cold', '$H^+$_warm'], dtype='<U10')
va_perp     = np.array([0.70655233, 2.23431466])
va_para     = np.array([0.70655233, 2.23431466])

cdens_str = va_perp_str = va_para_str = species_str = ''
for ii in range(Nj):
    cdens_str   += '{:>9.1f}'.format(ccdens[ii])
    species_str += '{:9}'.format(species_lbl[ii])
    va_perp_str += '{:9.1f}'.format(va_perp[ii])
    va_para_str += '{:>9.2f}'.format(va_para[ii])

print(cdens_str)