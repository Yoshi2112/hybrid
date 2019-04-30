# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 21:49:04 2019

@author: Yoshi
"""
import numpy as np
import matplotlib.pyplot as plt
import pdb

def basic_S(arr, k=5, h=1.5):
    N = arr.shape[0]
    S1 = np.zeros(N)
    S2 = np.zeros(N)
    S3 = np.zeros(N)
    
    for ii in range(N):
        if ii < k:
            left_vals = arr[:ii]
            right_vals = arr[ii + 1:ii + k + 1]
        elif ii >= N - k:
            left_vals  = arr[ii - k: ii]
            right_vals = arr[ii + 1:]
        else:
            left_vals  = arr[ii - k: ii]
            right_vals = arr[ii + 1:ii + k + 1]

        left_dist  = arr[ii] - left_vals
        right_dist = arr[ii] - right_vals
        
        if left_dist.shape[0] == 0:
            left_dist = np.append(left_dist, 0)
        elif right_dist.shape[0] == 0:
            right_dist = np.append(right_dist, 0)
        
        S1[ii] = 0.5 * (left_dist.max()     + right_dist.max()    )
        S2[ii] = 0.5 * (left_dist.sum() / k + right_dist.sum() / k)
        S3[ii] = 0.5 * ((arr[ii] - left_vals.sum() / k) + (arr[ii] - right_vals.sum() / k))
        
    S_ispeak = np.zeros((N, 3))
    
    for S, xx in zip([S1, S2, S3], np.arange(3)):
        for ii in range(N):
            if S[ii] > 0 and (S[ii] - S.mean()) > (h * S.std()):
                S_ispeak[ii, xx] = 1

    for xx in range(3):
        for ii in range(N):
            for jj in range(N):
                if ii != jj and S_ispeak[ii, xx] == 1 and S_ispeak[jj, xx] == 1:
                    if abs(jj - ii) <= k:
                        if arr[ii] < arr[jj]:
                            S_ispeak[ii, xx] = 0
                        else:
                            S_ispeak[jj, xx] = 0
                            
    S1_peaks = np.arange(N)[S_ispeak[:, 0] == 1]
    S2_peaks = np.arange(N)[S_ispeak[:, 1] == 1]
    S3_peaks = np.arange(N)[S_ispeak[:, 2] == 1]
    return S1_peaks, S2_peaks, S3_peaks


def S4(dat, k=5, h=1.5, w=None):
    
    if w is None:
        w = k
    
    def entropy(arr):
        M  = arr.shape[0]
        Hw = 0
        
        for ii in range(M):
            
            if ii + w < M:
                aiw = arr[ii + w]
            else:
                aiw = arr[-1]
                
            pw = 1. / (M * (arr[ii] - aiw))
            K_sum = 0
            for jj in range(M):
                K_arg = (arr[ii] - arr[jj]) / abs(arr[ii] - aiw)
                
                if abs(K_arg) < 1:
                    Kx = 0.75 * (1 - K_arg ** 2)
                else:
                    Kx = 0
                    
                K_sum += Kx
                
            pw *= K_sum
            Hw += -pw * np.log10(pw)
        return Hw
    
    peak_function = np.zeros(dat.shape[0])
    for ii in range(dat.shape[0]):
        N_full = dat
        N_miss = np.concatenate((dat[:ii]), dat[ii+1:])
        peak_function[ii] = entropy(N_miss) - entropy(N_full)
        
    return



if __name__ == '__main__':
    test_txt = 'F://Google Drive//Uni//PhD 2017//Data//SUNSPOT.txt'
    
    year = []; sunspot = []
    
    with open(test_txt) as fopen:
        for line in fopen:
            A = line.split()
            year.append(int(A[0]))
            sunspot.append(float(A[1]))
    year    = np.array(year)
    sunspot = np.array(sunspot)
    
    s1p, s2p, s3p = basic_S(sunspot)
    
    plt.plot(year, sunspot)
    plt.scatter(year[s2p], sunspot[s2p], c='r')