#!/usr/bin/env python

# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4

'''Occupancy Grid Map Representation and Operations

This module implements a binary occupancy grid representation of a map of the
environment, and related operations useful for localization and mapping.
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from math import floor, pi
import ray_trace
from sonar import Sonar, Scan, BadScanError

class OGMap():
    '''Representation of a square binary occupancy grid map'''
    def __init__(self, N, cache_file = 'trace_cache.npy'):
        '''Initialize an instance

        Args:
            N (int): side length of map
            cache_file (file): saved numpy array of ray traces
        '''
        self.N = N
        self.xs = np.array(range(self.N))
        self.ys = np.array(range(self.N))
        self.grid = np.ones((N,N), dtype=bool)
        self.edges = []
        self.rects = []

    def show(self):
        '''Plot a top down view of the map'''
        plt.imshow(self.grid
                    , interpolation = 'none'
                    , cmap = cm.Greys_r
                    , origin='lower'
                    )
        
    def rect(self, x0, y0, width, height):
        '''Place a rectangle with lower left corner at (x0, y0) 
        and dimensions (width x height)'''
        self.rects.append(Rect(x0, y0, width, height))
        self.edges.append((x0, y0, x0 + width, y0))
        self.edges.append((x0, y0, x0, y0 + height))
        self.edges.append((x0+width, y0, x0+width, y0+height))
        self.edges.append((x0, y0+height, x0+width, y0+height))
        self.grid[y0:y0+height, x0:x0+width] = 0
        
    def ray_trace(self, pose, theta, rmax):
        ''' Test for intersection of a ray with edges in the map
        
        Args:
          pose (1x3 array): robot pose, as (x,y) position and heading (rad)
          theta (radian): heading of ray_trace, in the robot frame
          rmax (int): maximum range of ray tracing
        '''
        assert pose.shape==(3,)
        #moved implementation to ray_trace.pyx for cython
        return ray_trace.ray_trace(self.edges, pose, theta, rmax)
    
    def ray_plot(self, pose, theta, rmax):
        '''Plot the map with a ray cast from (x0, y0) with heading theta'''
        x0, y0,phi = pose
        theta = theta+phi
        ray_len = self.ray_trace(pose, theta, rmax)
        self.show()
        plt.plot(x0, y0, '.', color='b', markersize = 20)
        plt.plot([x0
                , x0+ray_len*np.cos(theta)]
                ,[y0, y0+ray_len*np.sin(theta)]
                ,  color='r'
                , linestyle='-'
                , linewidth=2
                )
        plt.xlim((0,self.N))
        plt.ylim((0,self.N))
        plt.draw()
        
    def collision(self, x, y):
        '''Check if point (x, y) lies in an obstacle'''
        for rect in self.rects:
            if rect.collision(x, y):
                return True
        return False

class Rect():
    '''Representation of a rectangular obstacle'''
    def __init__(self, x0, y0, width, height):
        self.x0 = x0
        self.y0 = y0
        self.width = width
        self.height = height
        
    def collision(self, x, y):
        '''Check for overlap of point (x, y) with self'''
        return ((self.x0 <= x <= self.x0 + self.width) and (self.y0 <= y <=
            self.y0+self.height))


if __name__ == "__main__":
    from mapdef import mapdef, NTHETA
    this_map = mapdef()
    this_sonar = Sonar(NUM_THETA = 200, GAUSS_VAR = 1)
    pose = (50,50,0)

    plt.ion()
    this_sonar.simulate_scan(pose, this_map, PLOT_ON = True)
    plt.show(block=True)
