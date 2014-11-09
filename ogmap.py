'''Occupancy Grid Map Representation and Operations

This module implements a binary occupancy grid representation of a map of the
environment, and related operations useful for localization and mapping.
'''
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from math import floor, pi

os.chdir(os.path.dirname(os.path.realpath(__file__)))

class BadScanError(Exception):
    pass

def cross(a, b):
    '''return two-dimensional cross product'''
    return a[0]*b[1] - a[1]*b[0]

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
        self.cache_file = cache_file
        if os.path.isfile(self.cache_file):
            self.cache = np.load(self.cache_file)
            self.TRACES_CACHED = True
            self.cache_thetas = np.linspace(0, 2*pi, self.cache.shape[2])
            print 'found cache file'
        else:
            self.cache = None
            self.TRACES_CACHED = False
            self.cache_thetas = []
        
    def show(self):
        '''Plot a top down view of the map'''
        plt.imshow(self.grid
                    , interpolation = 'none'
                    , cmap = cm.Greys_r
                    , origin='lower'
                    )
        plt.draw()
        
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
          pose (tuple): robot pose, as (x,y) position and heading (rad)
          theta (radian): heading of ray_trace, in the robot frame
          rmax (int): maximum range of ray tracing
        '''
        dists = []
        x0, y0, phi = pose
        p = (x0, y0)
        s = (np.cos(theta + phi), np.sin(theta + phi))
        for edge in self.edges:
            r = (edge[0], edge[1])
            q = (edge[2] - edge[0], edge[3] - edge[1])
            den = cross(q, s)
            pr = (p[0] - r[0], p[1] - r[1])
            if den == 0:
                if cross(pr, s) == 0:
                    if np.dot(pr, s) < 0:
                        dists.append(np.linalg.norm(pr))
                        # print('parallel, intersecting')
                        continue
                    dists.append(rmax)
                    # print('parallel, non-intersecting')
                    continue
            u = cross(pr, s) / den
            if u > 1 or u < 0:
                dists.append(rmax)
                # print ('non-intersecting')
                continue
            t = cross(pr, q) / den
            if t < 0:
                dists.append(rmax)
                # print('wrong side')
                continue
            dists.append(t)
            # print('intersection')
            # print t
        return min(dists)
    
    def ray_plot(self, pose, theta, rmax):
        '''Plot the map with a ray cast from (x0, y0) with heading theta'''
        x0, y0,phi = pose
        theta = theta+phi
        ray_len = self.ray_trace(pose, theta, rmax)
        plt.imshow(self.grid
                    , interpolation = 'none'
                    , cmap = cm.Greys_r
                    , origin='lower'
                    )
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
        
    def cache_traces(self, filename, NUM_THETA = 8, RMAX = 100):
        f = open(filename, 'w')
        f.write('')
        
        traces = np.zeros((self.N, self.N, NUM_THETA))
        self.cache_thetas = np.linspace(0, 2*pi, NUM_THETA)
        
        xs = np.array(range(self.N))
        ys = np.array(range(self.N))
        
        for x_idx, x in np.ndenumerate(xs):
            print '%d out of %d'%(x, self.N)
            for y_idx, y in np.ndenumerate(ys):
                for th_idx, th in np.ndenumerate(self.cache_thetas):
                    traces[x_idx][y_idx][th_idx] = self.ray_trace(
                                                    (x, y, th)
                                                    ,  0
                                                    , RMAX
                                                    )
        
        np.save(filename, traces)
        f.close()
        self.CACHED = True

    def collision(self, x, y):
        '''Check if point (x, y) lies in an obstacle'''
        for rect in self.rects:
            if rect.collision(x, y):
                return True
        return False
        

class Sonar():
    def __init__(self, NUM_THETA = 200, GAUSS_VAR = (0.1)**2):
        ''' Example sonar parameters '''
        self.RMAX = 100             # max range
        self.EXP_LEN = 0.1            # length scale for under-ranges
        self.NUM_THETA = NUM_THETA        # number of headings
        self.GAUSS_VAR = GAUSS_VAR    # variance of accurate readings
        self.w_exp = 0.001            # weight for under-ranges
        self.w_gauss = 20           # weight for accurate ranges
        self.w_uni = 0.001           # weight for uniform glitches
        # weight for max glitches, obstacle present
        self.w_max_hit = 2          
        # weight for max glitches, no obstacle present
        self.w_max_miss = 20        
        self.w_min = .001              # weight for min glitches
        self.r_rez = 0.5              # resolution of range sensor
        
        self.thetas = np.linspace(0, 2*pi, self.NUM_THETA) #headings
        self.rs = np.arange(0, self.RMAX, self.r_rez)
        
        self.p_exp = np.exp(-self.rs / self.EXP_LEN)
        self.p_exp /= np.sum(self.p_exp)
        self.p_uni = np.ones(len(self.rs))
        self.p_uni /= np.sum(self.p_uni)
        self.p_max = np.zeros(len(self.rs))
        self.p_max[-1] = 1
        self.p_min = np.zeros(len(self.rs))
        self.p_min[0] = 1
        
        self.p_tot_partial = self.w_exp * self.p_exp \
                            + self.w_uni * self.p_uni \
                            + self.w_min * self.p_min

        
    def maxmin_filter(self, scan):
        '''Discard readings of 0 or RMAX, assumed to be spurious'''
        filtered = [(th, r) for (th, r) in scan.pings 
                            if r > 0 and r < self.RMAX]
        if not filtered:
            raise BadScanError
        else:
            return Scan(scan.pose, *zip(*filtered))
        
    def simulate_scan(self, pose, this_map, PLOT_ON = False):
        '''Return a simulation of a sonar reading from point (x0, y0) '''
        x0, y0, phi = pose
        theta = self.thetas#headings
        r = self.rs #radius
        r_meas = []
        
        for th in theta:
            r_meas.append(self.simulate_ping(pose, th, this_map))
        
        if PLOT_ON:
            ax = plt.subplot(111)
            plt.imshow(this_map.grid
                        , interpolation = 'none'
                        , cmap = cm.Greys_r
                        , origin='lower'
                        )
            plt.plot(x0+r_meas*np.cos(theta + phi)
                    , y0+r_meas*np.sin(theta+phi)
                    , '.'
                    , color='r'
                    )
            plt.plot(x0, y0, '.', color='b', markersize = 20) 
            # plt.xlim(0, this_map.N)
            # plt.ylim(0, this_map.N)
            plt.title('Simulated Sonar Scan')
            plt.draw()
        
        return Scan(pose, theta, r_meas)
        
    def simulate_ping(self, pose, th, this_map):
        '''Return a sample from the sonar probability density

        Args:
          pose (tuple): Robot pose, as (x, y) position and heading (rad)
          th (rad): Sensor heading, in robot frame
          this_map (OGMap): occupancy grid map'''
        p_tot = self.ping_pdf(pose, th, this_map)
        # sample
        r_meas = np.random.choice(self.rs, p=p_tot)
        return r_meas
        
    def ping_pdf(self, pose, th, this_map):
        '''Return a sonar probability density of a specified ray'''
        x0, y0, phi = pose
        if this_map.TRACES_CACHED:
            x_idx = np.argmin(abs(this_map.xs - x0))
            y_idx = np.argmin(abs(this_map.ys - y0))
            th_idx = np.argmin(abs((this_map.cache_thetas -\
                                    (th + phi))%(2*pi)))
            true_r = this_map.cache[x_idx][y_idx][th_idx]
        else:
            true_r = this_map.ray_trace(pose, th, self.RMAX)
        if true_r < self.RMAX:
            p_gauss = np.exp(-0.5*(self.rs - true_r)**2 / self.GAUSS_VAR)
            p_gauss /= np.sum(p_gauss)
            w_gauss = self.w_gauss
            w_max = self.w_max_hit
        else:
            p_gauss = 0
            w_gauss = 0
            w_max = self.w_max_miss

        # # weighted total probability
        p_tot = w_gauss * p_gauss + w_max * self.p_max + self.p_tot_partial
        # # normalize
        p_tot /= (np.sum(p_tot))
        # # sample
        return p_tot
        
class Scan():
    '''Representation of a sonar scan result'''
    def __init__(self, pose, thetas, rs):
        self.pose = pose
        x0, y0, phi = pose
        self.x0 = x0
        self.y0 = y0
        self.phi = phi
        self.thetas = thetas
        self.rs = rs
        self.pings = zip(self.thetas, self.rs)
        
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
    #this_map.show()
    this_sonar.simulate_scan(pose, this_map, PLOT_ON = True)
    plt.show(block=True)
