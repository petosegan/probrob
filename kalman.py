import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import math
from time import sleep

accel_var = 1 # Variance of random acceleration
meas_var = 10 # Variance of position measurement

A = np.matrix('1, 1; 0, 1') # Update matrix
A_tr = A.H

B = np.matrix('1, 0; 0, 1') # Control matrix

C = np.matrix('1, 0; 0, 1') # Measurement matrix
C_tr = C.H

R = np.matrix('0, 0; 0, %f'%accel_var) # Update covariance matrix

Q = np.matrix('%f, 0; 0, 1e6'%meas_var) # Measurement covariance matrix; 1e6 ~= infinity => no information

I = np.matrix(np.identity(2))

def kalman_filter(mu_last, cov_last, control, measure):
    ''' Carry out one iteration of a kalman filter '''
    mu_guess_now = A*mu_last + B*control
    cov_guess_now = A*cov_last*A_tr + R
    
    K = cov_guess_now*C_tr*np.linalg.inv(C*cov_guess_now*C_tr + Q)
    mu_now = mu_guess_now + K*(measure - C*mu_guess_now)
    cov_now = (I - K*C)*cov_guess_now
    return (mu_now, cov_now)
    
def blind_kalman_filter(mu_last, cov_last, control):
    ''' Carry out one iteration of a kalman filter with no measurements '''
    mu_now = A*mu_last + B*control
    cov_now = A*cov_last*A_tr + R

    return (mu_now, cov_now)
   
def plotEllipse(pos,P,edge,face):
    U, s , Vh = np.linalg.svd(P)
    orient = math.atan2(U[1,0],U[0,0])
    ellipsePlot = Ellipse(xy=pos, width=math.sqrt(s[0]), height=math.sqrt(s[1]), angle=orient,facecolor=face, edgecolor=edge)
    return ellipsePlot
   
if __name__ == "__main__":
    num_steps = 21
    meas_rate = 6
    win_size = 25 # plot window size

    mu = np.matrix('0; 0') # Initial mean
    cov = np.matrix('0, 0; 0, 0') # Initial covariance
    
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    ax.set_xlim(0, win_size)
    ax.set_ylim(0, win_size)
    plt.ion()
    plt.show()
    
    for i in range(num_steps):
        control = np.matrix([[0],[0]]) # Don't go anywhere
        mu, cov = blind_kalman_filter(mu, cov, control)
        print cov
        ellipsePlot = plotEllipse(mu, cov, 'k', 'none')
        ax.add_artist(ellipsePlot)
        sleep(0.1)
        plt.draw()
        
        if i % meas_rate == 0:
            measurement = mu # Suppose the best guess is always right
            mu, cov = kalman_filter(mu, cov, np.matrix('0;0'), measurement)
            print cov
            ellipsePlot = plotEllipse(mu, cov, 'red', 'none')
            ax.add_artist(ellipsePlot)
            sleep(0.1)
            plt.draw()
    
    mu, cov = kalman_filter(mu, cov, np.matrix('0;0'), np.matrix('0;0'))
    print cov
    ellipsePlot = plotEllipse(mu, cov, 'red', 'none')
    ax.add_artist(ellipsePlot)