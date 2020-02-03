# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 19:37:55 2019

@author: Yoshi
"""

import numpy as np

def compare_two_fields():
    runs       = [0, 1]
    lbx, lby, lbz, lex, ley, lez, lvex, lvey, lvez, lte, ljx, ljy, ljz, lqdens = [[] for _ in range(14)]
    
    for run in runs:
        cf.load_run(drive, series, run)
        
        tbx, tby, tbz, tex, tey, tez, tvex, tvey, tvez, tte, tjx, tjy, tjz,\
        tqdens = cf.get_array(get_all=True)
        
        lbx.append(tbx)
        lby.append(tby)
        lbz.append(tbz)
        lex.append(tex)
        ley.append(tey)
        lez.append(tez)
        lvex.append(tvex)
        lvey.append(tvey)
        lvez.append(tvez)
        lte.append(tte)
        ljx.append(tjx)
        ljy.append(tjy)
        ljz.append(tjz)
        lqdens.append(tqdens)
          
    return lbx, lby, lbz, lex, ley, lez, lvex, lvey, lvez, lte, ljx, ljy, ljz, lqdens


def compare_two_particles():
    
    main_folder = series_dir + 'particle_differences//'
        
    for ii in range(2312):
        print('Loading particle timestep {}'.format(ii))
        ts_folder = main_folder + 'ts{:04}//'.format(ii)
        os.makedirs(ts_folder)
        
        cf.load_run(drive, series, 0, extract_arrays=False)
        pos0, vel0 = cf.load_particles(ii)
        
        cf.load_run(drive, series, 1, extract_arrays=False)
        pos1, vel1 = cf.load_particles(ii)

        x_diff  = pos0 - pos1
        vx_diff = vel0[0, :] - vel1[0, :]
        vy_diff = vel0[1, :] - vel1[1, :]
        vz_diff = vel0[2, :] - vel1[2, :]
        
        np.savetxt(ts_folder + 'x_diff_{}.txt'.format( ii),  x_diff)
        np.savetxt(ts_folder + 'vx_diff_{}.txt'.format(ii), vx_diff)
        np.savetxt(ts_folder + 'vy_diff_{}.txt'.format(ii), vy_diff)
        np.savetxt(ts_folder + 'vz_diff_{}.txt'.format(ii), vz_diff)
    return