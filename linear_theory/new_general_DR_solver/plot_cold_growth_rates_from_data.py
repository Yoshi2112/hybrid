# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 13:30:20 2020

@author: Yoshi
"""
import numpy as np
import matplotlib.pyplot as plt

import extract_parameters_from_data   as data


if __name__ == '__main__':
    rbsp_path = 'G://DATA//RBSP//'
    save_drive= 'G://'
    
    time_start  = np.datetime64('2013-07-25T21:00:00')
    time_end    = np.datetime64('2013-07-25T22:00:00')
    probe       = 'a'
    pad         = 0
    
    date_string = time_start.astype(object).strftime('%Y%m%d')
    save_string = time_start.astype(object).strftime('%Y%m%d_%H%M_') + time_end.astype(object).strftime('%H%M')
    save_dir    = '{}NEW_LT//EVENT_{}//NEW_FIXED_DISPERSION_RESULTS//'.format(save_drive, date_string)

    Nf          = 1000
    f_max       = 5.0
    f_min       = 0.0
    freqs       = np.linspace(f_max, f_min, Nf) * 2 * np.pi
    pass
    # Create species array for each time
    # Set frequency range to look at (from 0-5Hz)
    # Get k (cold)
    # Calculate Dispersion relation
    # Calculate Temporal and Convective Growth Rates