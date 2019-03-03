# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 11:04:11 2019

@author: Yoshi
"""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

legend_elements = [Line2D([0], [0], color='k', lw=1, label='Boris Algorithm'  , linestyle='-'),
                   Line2D([0], [0], color='k', lw=1, label='Two-step Leapfrog', linestyle=':')]

# Create the figure
fig, ax = plt.subplots()
ax.legend(handles=legend_elements, loc='center')

plt.show()