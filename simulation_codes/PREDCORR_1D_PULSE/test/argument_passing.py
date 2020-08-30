# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 20:06:19 2020

@author: Yoshi
"""

import sys
import os

if os.path.exists(sys.argv[1]) == True:
    print('Path {} exists'.format(sys.argv[1]))
else:
    print('Path does not exist')