import numpy as np        
import matplotlib.pyplot as plt
from math import pi

class BadScanError(Exception):
    pass

class Sonar():
    def __init__(self
            , NUM_THETA = 200
            , GAUSS_VAR = (0.1)**2
            , weights={'w_exp':0.001
                , 'w_gauss':20
                , 'w_uni':0.001
                , 'w_max_hit':2
                , 'w_max_miss':20
                , 'w_min':0.001
                }
            , params={'RMAX':100
                , 'EXP_LEN':0.1
                , 'r_rez':0.5
                }
            ):
        ''' Example sonar parameters '''
        self.NUM_THETA = NUM_THETA        # number of headings
        self.GAUSS_VAR = GAUSS_VAR    # variance of accurate readings
        self.weights = weights
        self.params = params
        
        self.thetas = np.linspace(0, 2*pi, self.NUM_THETA) #headings
        self.rs = np.arange(0, self.params['RMAX'], self.params['r_rez'])
        
        self.p_exp = np.exp(-self.rs / self.params['EXP_LEN']) / self.params['EXP_LEN']
        self.p_uni = np.ones(len(self.rs)) / len(self.rs)
        self.p_max = np.zeros(len(self.rs))
        self.p_max[-1] = 1
        self.p_min = np.zeros(len(self.rs))
        self.p_min[0] = 1
        
        self.p_tot_partial = self.weights['w_exp'] * self.p_exp \
                            + self.weights['w_uni'] * self.p_uni \
                            + self.weights['w_min'] * self.p_min

    def maxmin_filter(self, scan):
        '''Discard readings of 0 or RMAX, assumed to be spurious'''
        filtered = [(th, r) for (th, r) in scan.pings 
                            if r > 0 and r < self.params['RMAX']]
        if not filtered:
            raise BadScanError
        else:
            return Scan(scan.pose, *zip(*filtered))
        
    def simulate_scan(self, pose, this_map, PLOT_ON = False):
        '''Return a simulation of a sonar reading from point (x0, y0) '''
        pose = np.array(pose)
        x0, y0, phi = pose
        theta = self.thetas#headings
        r = self.rs #radius
        r_meas = []

        for th in theta:
            r_meas.append(self.simulate_ping(pose, th, this_map))

        simscan = Scan(pose, theta, r_meas)

        if PLOT_ON:
            this_map.show()
            simscan.show(markersize=5)
            plt.plot(x0, y0, '.', color='b', markersize = 20) 
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
        r_meas = np.random.choice(self.rs, p=p_tot)
        return r_meas

    def ping_likelihood(self, pose, ping, this_map):
        theta, distance = ping
        x0, y0, phi = pose
        exp_term = np.exp(-distance / self.params['EXP_LEN']) / self.params['EXP_LEN']
        uni_term = 1.0 / len(self.rs)
        min_term = 1 if distance <= self.rs[1] else 0
        max_term = 1 if distance >= self.rs[-1] else 0
        true_r = this_map.ray_trace(pose, theta, self.params['RMAX'])
        gauss_term = (2*pi*self.GAUSS_VAR)**0.5*np.exp(-0.5*(distance-true_r)**2 / self.GAUSS_VAR)

        total_weight = self.weights['w_exp'] + self.weights['w_uni'] + self.weights['w_min'] + self.weights['w_max_hit'] + self.weights['w_gauss']

        return ((self.weights['w_exp']*exp_term
            +   self.weights['w_uni']*uni_term
            +   self.weights['w_min']*min_term
            +   self.weights['w_max_hit']*max_term
            +   self.weights['w_gauss']*gauss_term)/total_weight)


    #@profile    
    def ping_pdf(self, pose, th, this_map):
        '''Return a sonar probability density of a specified ray'''

        x0, y0, phi = pose
        true_r = this_map.ray_trace(pose, th, self.params['RMAX'])
        if true_r < self.params['RMAX']:
            p_gauss = np.exp(-0.5*(self.rs - true_r)**2 / self.GAUSS_VAR) / np.sqrt(2*pi*self.GAUSS_VAR)
            #p_gauss /= np.sum(p_gauss)
            w_gauss = self.weights['w_gauss']
            w_max = self.weights['w_max_hit']
        else:
            p_gauss = 0
            w_gauss = 0
            w_max = self.weights['w_max_miss']

        # weighted total probability
        p_tot = w_gauss * p_gauss + w_max * self.p_max + self.p_tot_partial
        # normalize
        p_tot /= (np.sum(p_tot))
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
        self.obst_distance = min(self.rs)

    def show(self, markersize=10, color='y', **kwargs):
        plt.plot(self.x0 + self.rs*np.cos(self.thetas + self.phi)
                ,self.y0 + self.rs*np.sin(self.thetas + self.phi)
                , '.'
                , color=color
                , markersize=markersize
                , **kwargs
                )


