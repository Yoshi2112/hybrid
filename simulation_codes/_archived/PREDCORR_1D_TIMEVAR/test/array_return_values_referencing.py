# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 12:29:37 2019

@author: Yoshi
"""

import numpy as np

def average_good(q1, q2):
    q1 *= 0.5; q1 += 0.5*q2
    return

def average_bad(q1, q2):
    q1 = 0.5*(q1 + q2)
    return

if __name__ == '__main__':
    good1 = np.arange(10, dtype=np.float64)
    good2 = np.arange(1, 100, 10, dtype=np.float64)
    average_good(good1, good2)
    
    bad1 = np.arange(10, dtype=np.float64)
    bad2 = np.arange(1, 100, 10, dtype=np.float64)
    average_bad(bad1, bad2)
    