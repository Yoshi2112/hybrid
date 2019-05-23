# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 22:29:00 2019

@author: Yoshi
"""

import os
import pickle
import time
import numpy as np
import pdb

def check_increment():
    h_name         = data_directory + 'Header.pckl'
    params         = pickle.load(open(h_name))
    print 'Current increment is {}'.format(params['data_dump_iter'])
    return


def check_max():
    h_name         = data_directory + 'Header.pckl'
    params         = pickle.load(open(h_name))
    print 'Current max_rev is {}'.format(params['max_rev'])
    return


def increase_increment(factor, delete=True, rename=True):
    check_increment()
    time.sleep(3)
    current_files    = [ii for ii in os.listdir(data_directory) if 'data0' in ii]

    if delete == True:
        for ii in range(len(current_files)):
            if ii%factor != 0:
                print 'Deleting {}'.format(data_directory + current_files[ii])
                os.remove(data_directory + current_files[ii])
        
        h_name                     = data_directory + 'Header.pckl'
        params                     = pickle.load(open(h_name))
        params['data_dump_iter']  *=  factor                            # Change header.pckl file to modify: data_iter *= 2
        
        with open(h_name, 'wb') as f:
            pickle.dump(params, f)
            f.close()
        check_increment()
    
    if rename == True:
        remaining_files = [ii for ii in os.listdir(data_directory) if 'data0' in ii]
        pdb.set_trace()
        for ii in range(len(remaining_files)):
            new_name = 'data{:05}.npz'.format(ii)
            print 'Renaming {} to {}'.format(remaining_files[ii], new_name)
            os.rename(data_directory + remaining_files[ii], data_directory + new_name)
    return


def delete_if_over(gyperiod_limit):
    h_name = data_directory + 'Header.pckl'
    params = pickle.load(open(h_name))
    
    B0  = params['B0']    
    dt  = params['dt'] * params['data_dump_iter']
    q   = 1.602177e-19                          # Elementary charge (C)
    m   = 1.672622e-27                          # Mass of proton (kg)
    
    gyperiod  = 2 * np.pi * m / (q * B0)
    
    file_list = [ii for ii in os.listdir(data_directory) if 'data0' in ii]
    N_slices  = len(file_list) - 2
    gy_times  = np.arange(N_slices) * dt / gyperiod

    check_max()
    time.sleep(3)
    for ii in range(N_slices):    
        if gy_times[ii] > gyperiod_limit:
            print 'Deleting {}'.format(data_directory + file_list[ii])
            os.remove(data_directory + file_list[ii])
    
    params['max_rev']  =  gyperiod_limit
    
    # Save again
    with open(h_name, 'wb') as f:
        pickle.dump(params, f)
        f.close()
        
    check_max()
    return


if __name__ == '__main__':
    series_directory = 'F://runs//Box_test_ev1_H_only//'
    
    num_runs = 1#len(os.listdir(series_directory))
    
    for xx in range(num_runs):
        print 'Examining run_{} directory...'.format(xx)
        time.sleep(3)
        
        data_directory   = series_directory + 'run_{}//data//'.format(xx)
        increase_increment(1, delete=False)
        #delete_if_over(30)
    

