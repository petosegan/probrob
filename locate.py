import ogmap
import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return array[idx]

def ping_likelihood(pt, ping, this_map, this_sonar):
    ''' Calculate the probability of a sonar measurement at a location '''
    (x0, y0) = pt
    (theta, range) = ping
    range_pdf = this_sonar.ping_pdf(x0, y0, theta, this_map)
    nearest_range_idx = (np.abs(this_sonar.rs - range)).argmin()
    return range_pdf[nearest_range_idx]
    
def scan_loglikelihood(pt, scan, this_map, this_sonar):
    L = 0
    for ping in scan.pings:
        L += np.log(ping_likelihood(pt, ping, this_map, this_sonar))
    return L
    
if __name__ == "__main__":
    # plt.ion()
    # fig = plt.figure()
    
    test = ogmap.OGMap(100)
    test.rect(95, 0, 10, 50)
    test.rect(0, 0, 10, 50)
    test.rect(10, 70, 50, 10)

    
    this_sonar = ogmap.Sonar()
    
    scan = this_sonar.simulate_scan(50, 50, test)
    # print scan
    
    ll_N = 10
    xs = np.linspace(10, 90, ll_N)
    ys = np.linspace(10, 90, ll_N)
    ll = np.zeros((ll_N, ll_N))
    for i, xpos in np.ndenumerate(xs):
        for j, ypos in np.ndenumerate(ys):
            # print i,j
            ll[i][j] = scan_loglikelihood((xpos,ypos), scan, test, this_sonar)
    # plt.imshow(ll, cmap=cm.Greys_r,interpolation = 'none', origin='lower')
    # plt.colorbar()
    # plt.draw()