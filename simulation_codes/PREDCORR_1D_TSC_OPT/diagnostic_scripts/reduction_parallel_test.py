#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 17:24:21 2019

@author: yoshi
"""

import numpy as np
import numba as nb

@nb.vectorize([nb.float64(nb.float64, nb.float64)])
def numba_sum(element1, element2):
    return element1 + element2

@nb.vectorize([nb.float64(nb.float64, nb.float64)], target='parallel')
def numba_sum_parallel(element1, element2):
    return element1 + element2

elements = 10000
array = np.ones(elements)

A = numba_sum.reduce(array)
B = numba_sum_parallel.reduce(array)
