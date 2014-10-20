import matplotlib.pyplot as plt
import numpy as np
from math import pi, exp, sqrt
import matplotlib.cm as cm
from time import sleep
from scipy.stats import norm as norm_dist

numsteps = 400
N = 1000
win_size = numsteps
win_size = 30
meas_rate = 5
goal = -20
p_gain = 0.1
# p_gain = 0

x_ens = np.zeros(N)
v_ens = np.zeros(N)

acc_var = 0.01
acc_std = np.sqrt(acc_var)

meas_var = 1
meas_std = np.sqrt(meas_var)
    
def blind_particle_filter(ensemble, control):
    x_last, v_last = ensemble
    acc = np.random.normal(0,acc_std,N)
    x_update = x_last + v_last + control[0]
    v_update = v_last + acc + control[1]
    return (x_update, v_update)
    
def particle_filter(ensemble, control, measure):
    x_update, v_update = blind_particle_filter(ensemble, control)
    weight = norm_dist.pdf(x_update, measure[0], meas_std)
    weight = weight / np.sum(weight) # normalize
    resample = np.random.choice(range(N), N, p=weight)
    x_meas = np.array([x_update[i] for i in resample])
    v_meas = np.array([v_update[i] for i in resample])
    return (x_meas, v_meas)
    
def show_ensemble(x_ens, v_ens, col = 'b'):
    plt.cla()
    plt.scatter(x_ens, v_ens, color = col)
    plt.xlim(-win_size, win_size)
    plt.ylim(-win_size/10, win_size/10)
    plt.xlabel('Position')
    plt.ylabel('Velocity')
    plt.title('One Dimension Brownian Motion Particle Filter')
    plt.draw()
    # sleep(0.1)
    
def show_history(x_ens, v_ens, col = 'b'):
    plt.scatter(x_ens, v_ens, color = col)
    plt.xlim(-win_size, win_size)
    plt.ylim(-win_size/10, win_size/10)
    plt.xlabel('Position')
    plt.ylabel('Velocity')
    plt.title('One Dimension Brownian Motion Particle Filter')
    plt.draw()
    # sleep(0.1)
    
if __name__ == "__main__": 
    control = np.array([0,0]) #stay put
    # control = np.array([1,1]) #move right
    # measurement = np.array([0,0]) # you never really left
    plt.ion()
    fig = plt.figure()
    show_ensemble(x_ens, v_ens)
    for i in range(numsteps):
        measurement = (np.mean(x_ens), np.mean(v_ens)) # good eye
        # measurement = (np.random.choice(x_ens), np.random.choice(v_ens))
        control = (0, -1*p_gain*np.sign(measurement[0] + goal))
        # x_ens, v_ens = particle_filter((x_ens, v_ens), control, measurement)
        # show_ensemble(x_ens, v_ens)
        x_ens, v_ens = blind_particle_filter((x_ens, v_ens), control)
        show_ensemble(x_ens, v_ens)
        if i % meas_rate == 0:
            measurement = (np.random.choice(x_ens), np.random.choice(v_ens))
            x_ens, v_ens = particle_filter((x_ens, v_ens), np.array([0,0]), measurement)
            show_ensemble(x_ens, v_ens, col = 'r')