# -*- coding: utf-8 -*-
"""
Created on Mon May 11 15:14:23 2020

@author: Yoshi
"""

import numba as nb
import numpy as np

@nb.njit()
def get_qmi(qmi, idx, dt):
    qmi[:] = 0.5 * dt * qm_ratios[idx]
    return


if __name__ == '__main__':
    
    QMI = np.zeros(10)
    IDX = np.random.randint(0, 2, 10)
    DT  = 1.0
    
    qm_ratios = np.array([10.0, 14.0], dtype=np.float64)
    
    get_qmi(QMI, IDX, DT)
    print(QMI)
    print(IDX.astype(float))