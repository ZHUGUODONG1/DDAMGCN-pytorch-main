import torch
import numpy as np
import pywt


def disentangle(x, w, j):
    x = x.transpose(2,1,0) # [S,N,T]
    coef = pywt.wavedec(x, w, level=j)
    coefl = [coef[0]]
    for i in range(len(coef)-1):
        coefl.append(None)
    coefh = [None]
    for i in range(len(coef)-1):
        coefh.append(coef[i+1])
    xl = pywt.waverec(coefl, w).transpose(2,1,0)
    xh = pywt.waverec(coefh, w).transpose(2,1,0)
    return xl, xh

data = np.load("dataset.npy")[:,:,:1]
xl, xh = disentangle(data, 'db1', 3)
np.savez('data_Trend.npz', data=xl)