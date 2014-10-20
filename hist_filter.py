import matplotlib.pyplot as plt
import numpy as np
from math import pi, exp, sqrt
import matplotlib.cm as cm

N = 31
numsteps = 5

x = np.linspace(-10, 10, N)
v = np.linspace(-10, 10, N)
sp = (x[1] - x[0])/2

X, V = np.meshgrid(x, v)

p0 = np.zeros(X.shape)
p0[N/2, N/2] = 1
plt.ion()
plt.imshow(p0, cmap = cm.Greys_r, interpolation='none')
plt.draw()

acc_var = 1

for ii in range(numsteps):
    print 'Step %d'%ii
    p1 = np.zeros(X.shape)
    for (i, j), p in np.ndenumerate(p0): #outer loop over config space
        for (k,l), x in np.ndenumerate(X):
            if X[i,j] > X[k,l] + V[k,l] - sp and X[i,j] <= X[k,l] + V[k,l] + sp:
                p1[i,j] = p1[i,j] + exp(-0.5 * (V[i,j] - V[k,l])**2 / acc_var) / sqrt(2*pi*acc_var) * p0[k,l]
    normalizer = np.sum(np.sum(p1))
    p1 = p1 / normalizer
    p0 = p1
    plt.imshow(p1, cmap = cm.Greys_r, interpolation='none')
    plt.draw()