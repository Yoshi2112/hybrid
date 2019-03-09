# -*- coding: utf-8 -*-
"""
Created on Sat Mar 09 11:17:41 2019

@author: iarey
"""

import psutil
import time

this_process = psutil.Process()

print 'This computer has {} CPUs'.format(psutil.cpu_count())
print 'This script with PID {} is running on CPUs {}'.format(this_process.pid, this_process.cpu_affinity())
print '\nChanging script CPU affinity...'

this_process.cpu_affinity([0])
print 'This script is now running on CPUs {}'.format(this_process.cpu_affinity())
print 'Sleeping for verification...'
time.sleep(600)