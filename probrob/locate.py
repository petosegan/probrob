#!/usr/bin/env python

# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
"""Localization Helper Function

This module provides functions for calculating the likelihood of sonar scan
data in a given map with a given pose.
"""

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import ogmap


def find_nearest(array, value):
    """Return the entry in array that is closest to value"""
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def ping_likelihood(pose, ping, some_map, some_sonar):
    return some_sonar.ping_likelihood(pose, ping, some_map)


def scan_loglikelihood(pose, some_scan, some_map, some_sonar):
    """Return the log-likelihood of a full sonar scan at a location, given a pose and map"""
    ll = 0
    if some_map.collision(pose[0], pose[1]):
        return float('NaN')
    for ping in some_scan.pings:
        ll += np.log(ping_likelihood(pose, ping, some_map, some_sonar))
    return ll


def loglike_map(pose, some_scan, some_map, some_sonar, ll_n=100
                , plot_on=False):
    """Return likelihood of scan at all points in grid"""
    this_x0, this_y0, this_phi = pose
    this_phi_guess = this_phi

    xs = np.linspace(1, some_map.gridsize - 1, ll_n)
    ys = np.linspace(1, some_map.gridsize - 1, ll_n)
    ll = np.zeros((ll_n, ll_n))
    for i, xpos in np.ndenumerate(xs):
        for j, ypos in np.ndenumerate(ys):
            this_pose = np.array((xpos, ypos, this_phi_guess))
            ll[j][i] = scan_loglikelihood(this_pose
                                          , some_scan
                                          , some_map
                                          , some_sonar
            )

    if plot_on:
        ll_masked = np.ma.masked_where(np.isnan(ll), ll)
        this_y0, this_x0 = np.unravel_index(ll_masked.argmax(), ll_masked.shape)
        ll_obstacles = np.ma.masked_where(np.isfinite(ll), ll)
        plt.ion()
        fig = plt.figure()
        fig.clf()
        plt.imshow(ll_masked, cmap=cm.Greys_r, interpolation='none'
                   , origin='lower')
        plt.colorbar()
        plt.imshow(np.isnan(ll_obstacles)
                   , cmap=cm.Greens_r
                   , interpolation='none'
                   , origin='lower'
        )
        plt.plot(true_pose[0]
                 , true_pose[1]
                 , '*'
                 , color='y'
                 , markersize=30)
        plt.plot(this_x0, this_y0, '.', color='b', markersize=20)
        plt.plot(this_x0 + some_scan.rs * np.cos(some_scan.thetas + this_phi_guess)
                 , this_y0 + some_scan.rs * np.sin(some_scan.thetas + this_phi_guess)
                 , '.'
                 , color='r'
                 , markersize=10
        )
        plt.xlim(0, some_map.gridsize)
        plt.ylim(0, some_map.gridsize)
        plt.draw()
    return ll, (xs, ys)


if __name__ == "__main__":
    from mapdef import mapdef

    N_THETA = 10

    true_pose = np.array((50, 50, 0))
    x0, y0, phi = true_pose
    phi_guess = phi

    this_sonar = ogmap.Sonar(num_theta=N_THETA, gauss_var=1)
    this_map = mapdef()

    scan = this_sonar.simulate_scan(true_pose, this_map)
    plt.ion()
    loglike_map(true_pose, scan, this_map, this_sonar, plot_on=True)
    plt.show(block=True)
