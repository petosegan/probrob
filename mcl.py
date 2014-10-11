from mapdef import mapdef
import ogmap
import locate
import matplotlib.pyplot as plt
import numpy as np
from math import pi, exp, sqrt
import matplotlib.cm as cm
from time import sleep
from scipy.stats import norm as norm_dist
import os

numsteps = 500
N_PART = 500
win_size = 100

class Ensemble():
    ''' Container for particles used in monte carlo localization '''

    def __init__(self,pose = (0,0, 0), N = N_PART, acc_var = np.array([[0.0001], [0.0001]]), meas_var = np.array((0.0001**2, 0.0001**2, 0.0001**2))):
        self.N = N
        self.pose = pose
        self.x_ens = np.tile(np.reshape(np.array(pose), (3,1)), (1, self.N))
        self.v_ens = np.zeros((2,N))
        self.dx = np.zeros((3, N))
        self.weight = np.ones(N) / N
        self.acc_std = np.sqrt(acc_var)
        self.meas_std = np.sqrt(meas_var)

    def pf_update(self, control_x, control_v):
        ''' Carry out update step of a particle filter algorithm'''
        acc = np.random.normal(0,self.acc_std,(2,self.N))
        self.x_ens = self.x_ens + self.dx + np.tile(control_x, (1, self.N))
        self.v_ens = self.v_ens + acc + control_v
        vr = self.v_ens[0,:]
        omega = self.v_ens[1,:]
        phi = self.x_ens[2,:]
        self.dx = np.array([np.abs(vr)*np.cos(phi), np.abs(vr)*np.sin(phi), omega])
        
    def pf_measure(self, measure_x):
        ''' Carry out measurement step of a particle filter algorithm'''
        weight_x = norm_dist.pdf(self.x_ens[0][:], measure_x[0], self.meas_std[0])
        weight_y = norm_dist.pdf(self.x_ens[1][:], measure_x[1], self.meas_std[1])
        weight_th = norm_dist.pdf(self.x_ens[2][:], measure_x[2], self.meas_std[2])
        weight = weight_x * weight_y * weight_th
        self.weight = weight / np.sum(weight) # normalize
        resample = np.random.choice(range(self.N), self.N, p=self.weight)
        self.x_ens = np.transpose(np.array([self.x_ens[:,i] for i in resample]))
        self.v_ens = np.transpose(np.array([self.v_ens[:,i] for i in resample]))
        
    def pf_sonar(self, scan, this_sonar, this_map):
        ''' Carry out measurement step of a particle filter algorithm, using sonar data '''
        weight = np.zeros(self.N)
        for i in range(self.N):
            weight[i] = np.exp(locate.scan_loglikelihood(self.x_ens[:, i], scan, this_map, this_sonar))
        weight = weight / np.sum(weight) # normalize
        resample = np.random.choice(range(self.N), self.N, p=weight)
        self.x_ens = np.transpose(np.array([self.x_ens[:,i] for i in resample]))
        self.v_ens = np.transpose(np.array([self.v_ens[:,i] for i in resample]))
        
    def show(self, col = 'b', win_size = win_size):
        plt.subplot(121)
        plt.cla()
        plt.scatter(self.x_ens[0][:], self.x_ens[1][:], color = col)
        plt.xlim(0, win_size)
        plt.ylim(0, win_size)
        plt.subplot(122)
        plt.cla()
        vel = plt.scatter(self.v_ens[0][:],self.v_ens[1][:], color = col)
        plt.xlim(-win_size/10, win_size/10)
        plt.ylim(-win_size/10, win_size/10)
        plt.draw()
    
    def show_map_scan(self, this_map, scan, pose, col = 'b', win_size = win_size):
        true_x, true_y, true_phi = pose
        plt.subplot(111)
        plt.cla()
        plt.plot(true_x, true_y, '*', color = 'y', markersize = 10)
        plt.quiver(self.x_ens[0][:], self.x_ens[1][:], np.cos(self.x_ens[2][:]), np.sin(self.x_ens[2][:]))
        plt.imshow(this_map.grid,cmap=cm.Greens_r,interpolation = 'none', origin='lower')
        plt.plot(true_x + scan.rs*np.cos(scan.thetas+true_phi), true_y + scan.rs*np.sin(scan.thetas+true_phi), '.',color = 'y', markersize = 10)
        plt.xlim(0, win_size)
        plt.ylim(0, win_size)
        plt.draw()    

if __name__ == "__main__": 
    from mapdef import mapdef, NTHETA
    control_x = np.array([[0],[0], [0]]) #stay put
    control_v = np.array([[0],[0]])
    measure_x = np.array([[0],[0], [0]])
    
    meas_rate = 10
    
    true_pose = (50,50, 0)
    
    this_ens = Ensemble(pose = true_pose)
    this_sonar = ogmap.Sonar(NUM_THETA = 10, GAUSS_VAR = 1)
    this_map = mapdef()
    
      
    plt.ion()
    fig = plt.figure()
    scan = this_sonar.simulate_scan(true_pose, this_map)
    for i in range(numsteps):
        this_ens.pf_update(control_x, control_v)
        this_ens.show_map_scan(col = 'b', scan = scan, this_map = this_map, pose = true_pose)
        if i % meas_rate == 0:
            scan = this_sonar.simulate_scan(true_pose, this_map)
            this_ens.pf_sonar(scan, this_sonar, this_map)
            this_ens.show_map_scan(col = 'r', scan = scan, this_map = this_map, pose=true_pose)
            plt.draw()