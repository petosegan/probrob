import matplotlib.pyplot as plt
import numpy as np
from math import pi

EXP_LEN = 1.0
GAUSS_WIDTH = 0.01
GAUSS_VAR = GAUSS_WIDTH**2
RMAX = 10
NUM_THETA = 200
WALL_DIST = 2.0

theta = np.linspace(0, 2*pi, NUM_THETA)
r = np.linspace(0, RMAX, NUM_THETA) #radius
r_meas = []

w_exp = 0.1
w_gauss = 20
w_uni = 0.01
w_max = 2
w_min = 2

p_exp = np.exp(-r / EXP_LEN)
p_uni = np.ones(len(r))
p_max = np.zeros(len(r))
p_max[-1] = 1
p_min = np.zeros(len(r))
p_min[0] = 1

for th in theta:

    if np.cos(th) > 0 and WALL_DIST/np.cos(th) < RMAX: # An environment with a wall at X = WALL_DIST
        p_gauss = np.exp(-0.5*(r - WALL_DIST/np.cos(th))**2 / GAUSS_VAR)
        w_max = 2
    else:
        p_gauss = np.zeros(len(r))
        w_max = 20


    p_tot = w_exp * p_exp + w_gauss * p_gauss + w_uni * p_uni + w_max * p_max + w_min*p_min
    p_tot /= (np.sum(p_tot))
    r_meas.append(np.random.choice(r, p=p_tot))


fig=plt.figure()
plt.subplot(121)
plt.plot(r, p_tot)
plt.xlabel('Distance')
plt.ylabel('Probability Density')
plt.title('Probability Density at $\\theta$ = 0')

ax = plt.subplot(122, polar=True)
plt.plot(theta, r_meas, '.', color='r')
plt.title('Simulated Sonar Scan')
plt.show()