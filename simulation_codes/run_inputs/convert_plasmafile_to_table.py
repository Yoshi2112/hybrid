# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 15:22:26 2022

@author: Yoshi
"""
import os, pdb
import numpy as np

def read_plasmafile(fname):
    print('')
    with open(fname, 'r') as f:
        species_lbl = np.array(f.readline().split()[1:])
        
        temp_color = np.array(f.readline().split()[1:])
        temp_type  = np.array(f.readline().split()[1:], dtype=int)
        dist_type  = np.array(f.readline().split()[1:], dtype=int)
        nsp_ppc    = np.array(f.readline().split()[1:], dtype=int)
        
        mass       = np.array(f.readline().split()[1:], dtype=float)
        charge     = np.array(f.readline().split()[1:], dtype=float)
        drift_v    = np.array(f.readline().split()[1:], dtype=float)
        density    = np.array(f.readline().split()[1:], dtype=float)
        anisotropy = np.array(f.readline().split()[1:], dtype=float)
                                  
        E_perp     = np.array(f.readline().split()[1:], dtype=float)*1e-3
        E_e        = float(f.readline().split()[1])
        beta_flag  = int(f.readline().split()[1])
    
        L         = float(f.readline().split()[1])           # Field line L shell, used to calculate parabolic scale factor.
        B_eq      = f.readline().split()[1]                  # Magnetic field at equator in T: '-' for L-determined value :: 'Exact' value in node ND + NX//2
        B_xmax_ovr= f.readline().split()[1]                  # Magnetic field at simulation boundaries (excluding damping cells). Overrides 'a' calculation.

    return species_lbl, temp_color, temp_type, dist_type, nsp_ppc, \
        mass, charge, drift_v, density, anisotropy,\
            E_perp, E_e, beta_flag, L, float(B_eq)*1e9, B_xmax_ovr
            
            
def compile_run_series_into_runfile(run_folder):
    output_file = run_folder + '_table.txt'
    run_count = 0
    header = 'Run   & $B_0$ & $n_{H^+, c}$  & $n_{H^+, h}$ & $T_{H^+, h}$ & $A_{H^+, h}$ & $n_{He^+, c}$\
              & $n_{He^+, h}$ & $T_{He^+, h}$ & $A_{He^+, h}$ & $n_{O^+, c}$ & $n_{O^+, h}$ & $T_{O^+, h}$\
                  & $A_{O^+, h}$ \\\\'
    with open(output_file, 'w') as f:
        print(header, file=f)
        for file in os.listdir(run_folder):
            if file.endswith('.plasma'):
                print('Reading:', file)
                species_lbl, temp_color, temp_type, dist_type, nsp_ppc, \
                mass, charge, drift_v, density, anisotropy,\
                E_perp, E_e, beta_flag, L, B_eq, B_xmax_ovr = read_plasmafile(run_folder+file)
                
                H_str  = f'{density[0]:.3f} & {density[3]:.3f} & {E_perp[3]:.3f} & {anisotropy[3]:.3f} & '
                He_str = f'{density[1]:.3f} & {density[4]:.3f} & {E_perp[4]:.3f} & {anisotropy[4]:.3f} & '
                O_str  = f'{density[2]:.3f} & {density[5]:.3f} & {E_perp[5]:.3f} & {anisotropy[5]:.3f}'
                
                run_str = f'{run_count} & {B_eq:.2f} & ' + H_str + He_str + O_str + '\\\\'
                run_count += 1
                print(run_str, file=f)
    return


def compile_run_series_into_runfile_Honly(run_folder):
    output_file = run_folder + '_table.txt'
    run_count = 0
    header = 'Run & $B_0$ & $n_{H^+, c}$ & $n_{H^+, h}$ & $T_{H^+, h}$ & $A_{H^+, h}$ & $n_{He^+, c}$ & $n_{O^+, c}$\\\\'
    with open(output_file, 'w') as f:
        print(header, file=f)
        for file in os.listdir(run_folder):
            if file.endswith('.plasma'):
                print('Reading:', file)
                species_lbl, temp_color, temp_type, dist_type, nsp_ppc, \
                mass, charge, drift_v, density, anisotropy,\
                E_perp, E_e, beta_flag, L, B_eq, B_xmax_ovr = read_plasmafile(run_folder+file)

                H_str  = f'{density[0]:.3f} & {density[3]:.3f} & {E_perp[3]:.3f} & {anisotropy[3]:.3f} & '
                He_str = f'{density[1]:.3f} & '
                O_str  = f'{density[2]:.3f}'
                
                run_str = f'{run_count} & {B_eq:.2f} & ' + H_str + He_str + O_str + '\\\\'
                run_count += 1
                print(run_str, file=f)
    return


if __name__ == '__main__':
    main_folder = 'D://Google Drive//Uni//PhD 2017//Josh PhD Share Folder//Thesis//Hybrid Batch Scripts for parsing//'
    read_folder = 'JUL25_PROXYHONLY_30HE_PREDCORR//'
    
    compile_run_series_into_runfile_Honly(main_folder + read_folder)