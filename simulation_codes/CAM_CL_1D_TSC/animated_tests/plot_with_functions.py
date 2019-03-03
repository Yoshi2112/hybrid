# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 14:46:45 2019

@author: Yoshi
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_something(y, ax_in):
    ax_in.plot(y)
    return


if __name__ == '__main__':
    plt.ioff()
    
    x  = np.arange(50)
    y1 = 3*x
    y2 = x**2

    fig  = plt.figure(figsize=(15, 7))
    ax   = plt.subplot2grid((7, 7), (0, 0), colspan=6, rowspan=7)

    for yi in [y1, y2]:
        plot_something(yi, ax)

    plt.show()