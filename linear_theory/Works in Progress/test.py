# -*- coding: utf-8 -*-
"""
Created on Thu May 21 11:48:54 2020

@author: Yoshi
"""

import numpy as np

# Create structured array :: Either use class or numpy structured array? Can
# I create np structured array without knowing N_species ahead of time? Is that
# ever a problem?

Species = np.array([('Rex', 9, 81.0), ('Fido', 3, 27.0)],
       
                   dtype=[('name', 'U10'),
                          ('mass', 'i4'),
                          ('density', 'f4'),
                          ('tpar', 'f4')
                          ('anisotropy', 'f4')])



