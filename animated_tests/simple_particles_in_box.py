# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 10:34:04 2020

@author: Yoshi
"""

import numpy as np
import matplotlib.pyplot as plt


plt.ion()
fig, ax = plt.subplots()

position = np.array([ 0.0, 0.0])
velocity = np.array([10.0, 0.0])
dt       = 0.1
blim     = 10

scat     = ax.scatter(position[0], position[1], c='k')
ax.set(xlim=(-blim, blim), ylim=(-blim, blim))

plt.draw()
for i in range(1000):
    position += velocity * dt
    
    if position[0] > blim or position[0] < -blim:
        velocity[0] *= -1.0
        position[0] += velocity[0] * dt
        
    if position[1] > blim or position[1] < -blim:
        velocity[1] *= -1.0
        position[1] += velocity[1] * dt
    
    scat.set_offsets(position)
    fig.canvas.draw_idle()
    plt.pause(0.3)

plt.waitforbuttonpress()
