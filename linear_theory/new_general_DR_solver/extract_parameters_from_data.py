# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 15:42:27 2019

@author: Yoshi
"""

import sys
sys.path.append('F://Google Drive//Uni//PhD 2017//Data//Scripts//')
import numpy as np
import rbsp_file_readers   as rfr
import rbsp_fields_loader  as rfl


def interpolate_to_edens(edens_time, data_time, data_array_dens, data_array_temp, data_array_anis):
    '''
    edens_time  :: WAVES electron density time to interpolate data_array to (length M)
    data_time   :: Current data sample times (HOPE/RBSPICE) of length N
    data_array  :: Data arrays consisting of ni, Ti, Ai in a 3xN ndarra
    '''
    new_data_dens = np.zeros((3, edens_time.shape[0]), dtype=np.float64)
    new_data_temp = np.zeros((3, edens_time.shape[0]), dtype=np.float64)
    new_data_anis = np.zeros((3, edens_time.shape[0]), dtype=np.float64)
    
    xi = edens_time.astype(np.int64)
    xp =  data_time.astype(np.int64)
    
    for ii in range(3):
        new_data_dens[ii, :] = np.interp(xi, xp, data_array_dens[ii, :])
        new_data_temp[ii, :] = np.interp(xi, xp, data_array_temp[ii, :])
        new_data_anis[ii, :] = np.interp(xi, xp, data_array_anis[ii, :])

    return new_data_dens, new_data_temp, new_data_anis


def interpolate_B(edens_time, b_time, b_array):
    
    xp = b_time.astype(np.int64)
    yp = np.sqrt(b_array[:, 0] ** 2 + b_array[:, 1] ** 2 + b_array[:, 2] ** 2)
    
    xi = edens_time.astype(np.int64)

    new_b = np.interp(xi, xp, yp)
    return new_b


def load_and_interpolate_plasma_params(time_start, time_end, probe, pad, rbsp_path='G://DATA//RBSP//', cold_composition=np.array([70, 20, 10])):
    '''
    Outputs as SI units: B0 in nT, densities in /m3, temperatures in eV (pseudo SI)
    '''
    
    # Cold (total?) electron plasma density
    den_times, edens, dens_err    = rfr.retrieve_RBSP_electron_density_data(rbsp_path, time_start, time_end, probe, pad=pad)

    # Magnetic magnitude
    mag_times, B_mag = rfl.load_magnetic_field(rbsp_path, time_start, time_end, probe, return_raw=True)

    # HOPE data
    itime, etime, pdict, perr = rfr.retrieve_RBSP_hope_moment_data(     rbsp_path, time_start, time_end, padding=pad, probe=probe)
    hope_dens = np.array([pdict['Dens_p_30'],       pdict['Dens_he_30'],       pdict['Dens_o_30']])
    hope_temp = np.array([pdict['Tperp_p_30'],      pdict['Tperp_he_30'],      pdict['Tperp_o_30']])
    hope_anis = np.array([pdict['Tperp_Tpar_p_30'], pdict['Tperp_Tpar_he_30'], pdict['Tperp_Tpar_o_30']]) - 1
    
    # SPICE data
    spice_dens = [];    spice_temp = [];    spice_anis = []
    for product, spec in zip(['TOFxEH', 'TOFxEHe', 'TOFxEO'], ['P', 'He', 'O']):
        spice_time , spice_dict  = rfr.retrieve_RBSPICE_data(rbsp_path, time_start, time_end, product , padding=pad, probe=probe)
        
        this_dens = spice_dict['F{}DU_Density'.format(spec)]
        this_anis = spice_dict['F{}DU_PerpPressure'.format(spec)] / spice_dict['F{}DU_ParaPressure'.format(spec)] - 1
        
        # Perp Temperature - Calculate as T = P/nk
        kB            = 1.381e-23; q = 1.602e-19
        t_perp_kelvin = 1e-9*spice_dict['F{}DU_PerpPressure'.format(spec)] / (kB*1e6*spice_dict['F{}DU_Density'.format(spec)])
        this_temp     = kB * t_perp_kelvin / q  # Convert from K to eV
        
        spice_dens.append(this_dens)
        spice_temp.append(this_temp)
        spice_anis.append(this_anis)
    
    ihope_dens , ihope_temp , ihope_anis  = interpolate_to_edens(den_times, itime, hope_dens, hope_temp, hope_anis)
    ispice_dens, ispice_temp, ispice_anis = interpolate_to_edens(den_times, spice_time, np.array(spice_dens),
                                                                 np.array(spice_temp), np.array(spice_anis))
    
    Bi = interpolate_B(den_times, mag_times, B_mag)
    
    # Subtract energetic components from total electron density (assuming each is singly charged)
    cold_dens = edens - ihope_dens.sum(axis=0) - ispice_dens.sum(axis=0)

    return den_times, Bi*1e-9, cold_dens*1e6, ihope_dens*1e6, ihope_temp, ihope_anis, ispice_dens*1e6, ispice_temp, ispice_anis


if __name__ == '__main__':
    _rbsp_path  = 'E://DATA//RBSP//'
    _time_start = np.datetime64('2013-07-25T21:00:00')
    _time_end   = np.datetime64('2013-07-25T22:00:00')
    _probe      = 'a'
    _pad        = 0
    
    _times, _B0, _cold_dens, _hope_dens, _hope_temp, _hope_anis, _spice_dens, _spice_temp, _spice_anis =\
        load_and_interpolate_plasma_params(_time_start, _time_end, _probe, _pad)
    
    #import matplotlib.pyplot as plt
    #plt.plot(_times, _cold_dens)