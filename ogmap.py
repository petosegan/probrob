import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm
from math import floor, pi

def cross(a, b):
    return a[0]*b[1] - a[1]*b[0]

class OGMap():
    def __init__(self, N):
        self.N = N
        self.grid = np.ones((N,N), dtype=bool)
        self.edges = []
        self.rects = []
        
    # def show(self, fig):
        # ax = fig.add_subplot(111)
        # ax.imshow(self.grid, interpolation = 'none', cmap = cm.Greys_r, origin='lower')
        # plt.draw()
        
    def rect(self, x0, y0, width, height):
        '''Place a rectangle with lower left corner at (x0, y0) and dimensions width x height'''
        self.rects.append((x0, y0, width, height))
        self.edges.append((x0, y0, x0 + width, y0))
        self.edges.append((x0, y0, x0, y0 + height))
        self.edges.append((x0+width, y0, x0+width, y0+height))
        self.edges.append((x0, y0+height, x0+width, y0+height))
        self.grid[y0:y0+height, x0:x0+width] = 0
        
    def ray_cast(self, x0, y0, theta, rmax):
        ''' Returns the distance to the nearest occupied point at heading theta.
        If no points are occupied along the ray, return rmax.'''
        
        theta = theta % (2*pi)
        
        # take care of special cases and get the ystep sign right
        if theta == pi/2:
            ystep = self.N
        elif theta == -pi/2:
            ystep = -self.N
        else:
            ystep = np.abs(np.tan(theta))*np.sign(np.sin(theta))            
            
        # get the xstep sign right
        if theta < pi/2 or theta > 3*pi/2:
            xend = self.N
            xstep = 1
        else:
            xend = 0
            xstep = -1
            
        y_this = y0
        
        # follow a ray until you hit an occupied square or go out of the map
        for x in np.arange(x0, xend, xstep):
            for y in np.arange(y_this, y_this+ystep, np.sign(ystep)):
                if y >= self.N or y < 0:
                    return rmax
                elif self.grid[y, int(x)] == 0:
                    return min(np.sqrt((x-x0)**2 + (y-y0)**2), rmax)
            y_this = y_this + ystep
        return rmax
    
    def ray_trace(self, x0, y0, theta, rmax):
        dists = []
        p = (x0, y0)
        s = (np.cos(theta), np.sin(theta))
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
    
    # def ray_plot(self, x0, y0, theta, rmax):
        # '''Plot the map with a ray cast from (x0, y0) with heading theta'''
        # ray_len = self.ray_trace(x0, y0, theta, rmax)
        # plt.imshow(self.grid, interpolation = 'none', cmap = cm.Greys_r, origin='lower')
        # plt.plot(x0, y0, '.', color='b', markersize = 20)
        # plt.plot([x0, x0+ray_len*np.cos(theta)],[y0, y0+ray_len*np.sin(theta)],  color='r', linestyle='-', linewidth=2)
        # plt.xlim((0,self.N))
        # plt.ylim((0,self.N))
        # plt.draw()
        
    def sonar_simulate(self, x0, y0, sonar):
        ''' Produce a simulation of a sonar reading from point (x0, y0) '''
        theta = np.linspace(0, 2*pi, sonar.NUM_THETA) #headings
        r = np.linspace(0, sonar.RMAX, sonar.NUM_THETA) #radius
        r_meas = []
        
        # probability densities
        p_exp = np.exp(-r / sonar.EXP_LEN)
        p_uni = np.ones(len(r))
        p_max = np.zeros(len(r))
        p_max[-1] = 1
        p_min = np.zeros(len(r))
        p_min[0] = 1

        for th in theta:
            true_r = self.ray_trace(x0, y0, th, sonar.RMAX)
            if true_r < sonar.RMAX:
                p_gauss = np.exp(-0.5*(r - true_r)**2 / sonar.GAUSS_VAR)
                w_max = sonar.w_max_hit
            else:
                p_gauss = np.zeros(len(r))
                w_max = sonar.w_max_miss

            # weighted total probability
            p_tot = sonar.w_exp * p_exp + sonar.w_gauss * p_gauss + sonar.w_uni * p_uni + w_max * p_max + sonar.w_min*p_min
            # normalize
            p_tot /= (np.sum(p_tot))
            # sample
            r_meas.append(np.random.choice(r, p=p_tot))

        # ax = plt.subplot(111)
        # plt.imshow(self.grid, interpolation = 'none', cmap = cm.Greys_r, origin='lower')
        # plt.plot(x0+r_meas*np.cos(theta), y0+r_meas*np.sin(theta), '.', color='r')
        # plt.plot(x0, y0, '.', color='b', markersize = 20) 
        # plt.xlim(0, self.N)
        # plt.ylim(0, self.N)
        # plt.title('Simulated Sonar Scan')
        # plt.draw()
        
        return (theta, r_meas)

class Sonar():
    def __init__(self):
        ''' Example sonar parameters '''
        self.RMAX = 100             # max range
        self.EXP_LEN = 0.1            # length scale for under-ranges
        self.NUM_THETA = 200        # number of headings
        self.GAUSS_VAR = (.1)**2    # variance of accurate readings
        self.w_exp = 0.1            # weight for under-ranges
        self.w_gauss = 20           # weight for accurate ranges
        self.w_uni = 0.01           # weight for uniform glitches
        self.w_max_hit = 2          # weight for max glitches, obstacle present
        self.w_max_miss = 20        # weight for max glitches, no obstacle present
        self.w_min = 2              # weight for min glitches
        self.r_rez = 0.5              # resolution of range sensor
        
        self.thetas = np.linspace(0, 2*pi, self.NUM_THETA) #headings
        self.rs = np.arange(0, self.RMAX, self.r_rez)
        
        self.p_exp = np.exp(-self.rs / self.EXP_LEN)
        self.p_uni = np.ones(len(self.rs))
        self.p_max = np.zeros(len(self.rs))
        self.p_max[-1] = 1
        self.p_min = np.zeros(len(self.rs))
        self.p_min[0] = 1
        
        self.p_tot_partial = self.w_exp * self.p_exp + self.w_uni * self.p_uni + self.w_min * self.p_min

        
    def maxmin_filter(self, scan):
        filtered = [(th, r) for (th, r) in scan.pings if r > 0 and r < self.RMAX]
        return Scan(scan.x0, scan.y0, *zip(*filtered))
        
    def simulate_scan(self, x0, y0, this_map, PLOT_ON = False):
        ''' Produce a simulation of a sonar reading from point (x0, y0) '''
        theta = self.thetas #headings
        r = self.rs #radius
        r_meas = []
        
        for th in theta:
            r_meas.append(self.simulate_ping(x0, y0, th, this_map))
        
        # if PLOT_ON:
            # ax = plt.subplot(111)
            # plt.imshow(this_map.grid, interpolation = 'none', cmap = cm.Greys_r, origin='lower')
            # plt.plot(x0+r_meas*np.cos(theta), y0+r_meas*np.sin(theta), '.', color='r')
            # plt.plot(x0, y0, '.', color='b', markersize = 20) 
            # plt.xlim(0, this_map.N)
            # plt.ylim(0, this_map.N)
            # plt.title('Simulated Sonar Scan')
            # plt.draw()
        
        return Scan(x0, y0, theta, r_meas)
        
    def simulate_ping(self, x0, y0, th, this_map):
        p_tot = self.ping_pdf(x0, y0, th, this_map)
        # sample
        r_meas = np.random.choice(self.rs, p=p_tot)
        return r_meas
        
    def ping_pdf(self, x0, y0, th, this_map):
        if this_map.grid[y0, x0] == 0:
            return self.p_min
        true_r = this_map.ray_trace(x0, y0, th, self.RMAX)
        if true_r < self.RMAX:
            p_gauss = np.exp(-0.5*(self.rs - true_r)**2 / self.GAUSS_VAR)
            w_max = self.w_max_hit
        else:
            p_gauss = np.zeros(len(self.rs))
            w_max = self.w_max_miss

        # weighted total probability
        p_tot = self.w_gauss * p_gauss + w_max * self.p_max + self.p_tot_partial
        # normalize
        p_tot /= (np.sum(p_tot))
        # sample
        return p_tot
        
class Scan():
    def __init__(self, x0, y0, thetas, rs):
        self.x0 = x0
        self.y0 = y0
        self.thetas = thetas
        self.rs = rs
        self.pings = zip(self.thetas, self.rs)
        
if __name__ == "__main__":
    # plt.ion()
    
    test = OGMap(100)
    test.rect(95, 0, 10, 50)
    test.rect(0, 0, 10, 50)
    # test.ray_plot(50, 50, 5*pi/4, 100)
    
    # test.show()
    for xpos in np.linspace(0, 80, 10):
        test.sonar_simulate(xpos, 60, Sonar())