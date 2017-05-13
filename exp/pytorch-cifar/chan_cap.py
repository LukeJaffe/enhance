#!/usr/bin/env python3

import noise
import progressbar
import numpy as np

fname = 'img1.npy'
#traindata, testdata = noise.get_data()
#with open(fname, 'wb') as fp:
#    np.save(fp, traindata[0])

with open(fname, 'rb') as fp:
    img1 = np.load(fp)

def chan_cap(img, N=1.0):
    X = np.ones_like(img)*255.0
    P = np.sum(X*X)/len(X.ravel())
    C = 0.5*np.log(1.0 + P/N)
    return C

for N in 10.0**(np.arange(-10, 11, 1)):
    n = 8.0
    C = chan_cap(img1[:, :, 0], N=N)
    S = 2**(n*C)
    print(N, C, S)

#with progressbar.ProgressBar(max_value=10) as bar:
#    for i in range(10):
#        time.sleep(0.1)
#        bar.update(i)
