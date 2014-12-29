#!/usr/bin/env python

# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
'''Localization Helper Function

This module provides functions for calculating the likelihood of sonar scan
data in a given map with a given pose.
'''

import os
import ogmap
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from math import pi

def find_nearest(array,value):
    '''Return the entry in array that is closest to value'''
    idx = (np.abs(array-value)).argmin()
    return array[idx]

def ping_likelihood(pose, ping, this_map, this_sonar):
    ''' Calculate the probability of a sonar measurement (ping) at a location,
        given a pose and map'''
    (theta, range) = ping
    range_pdf = this_sonar.ping_pdf(pose, theta, this_map)
    nearest_range_idx = (np.abs(this_sonar.rs - range)).argmin()
    return range_pdf[nearest_range_idx]
    
def scan_loglikelihood(pose, scan, this_map, this_sonar):
    '''Return the log-likelihood of a full sonar scan at a location, given a pose and map'''
    L = 0
    if this_map.collision(pose[0], pose[1]):
        return float('NaN')
    for ping in scan.pings:
        L += np.log(ping_likelihood(pose, ping, this_map, this_sonar))
    return L
    
def loglike_map(pose, scan, this_map, this_sonar,ll_N = 100
                , PLOT_ON = False):
    '''Return likelihood of scan at all points in grid'''
    x0, y0, phi = pose
    phi_guess = phi
    
    xs = np.linspace(1, this_map.N-1, ll_N)
    ys = np.linspace(1, this_map.N-1, ll_N)
    ll = np.zeros((ll_N, ll_N))
    for i, xpos in np.ndenumerate(xs):
        for j, ypos in np.ndenumerate(ys):
            this_pose = np.array((xpos, ypos, phi_guess))
            ll[j][i] = scan_loglikelihood(this_pose
                                        , scan
                                        , this_map
                                        , this_sonar
                                        )
            
    if PLOT_ON:
        ll_masked = np.ma.masked_where(np.isnan(ll),ll)
        y0, x0 =  np.unravel_index(ll_masked.argmax(), ll_masked.shape)
        ll_obstacles = np.ma.masked_where(np.isfinite(ll), ll)
        plt.ion()
        fig = plt.figure()
        fig.clf()
        plt.imshow(ll_masked, cmap=cm.Greys_r,interpolation = 'none'
                    , origin='lower')
        plt.colorbar()
        plt.imshow(np.isnan(ll_obstacles)
                , cmap=cm.Greens_r
                ,interpolation = 'none'
                , origin='lower'
                )
        plt.plot(true_pose[0]
                , true_pose[1]
                , '*'
                , color='y'
                , markersize=30)
        plt.plot(x0, y0, '.', color = 'b', markersize = 20)
        plt.plot(x0 + scan.rs*np.cos(scan.thetas+phi_guess)
                , y0 + scan.rs*np.sin(scan.thetas+phi_guess)
                , '.'
                ,color = 'r'
                , markersize = 10
                )
        # plt.xlim(0, ll_N)
        # plt.ylim(0, ll_N)
        plt.draw()
    return (ll, (xs, ys))


if __name__ == "__main__":
    from mapdef import mapdef
    NTHETA = 2
    
    true_pose = np.array((50,50, 0))
    x0, y0, phi = true_pose
    phi_guess = phi
    
    this_sonar = ogmap.Sonar(NUM_THETA = NTHETA, GAUSS_VAR = 1)
    this_map = mapdef()
       
    scan = this_sonar.simulate_scan(true_pose, this_map)
    plt.ion() 
    loglike_map(true_pose, scan, this_map, this_sonar, PLOT_ON = True)
    plt.show(block=True)
