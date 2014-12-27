import numpy as np
import ogmap
from mapdef import mapdef, NTHETA
import matplotlib.pyplot as plt
from math import pi, exp, sin, cos, sqrt
import matplotlib.cm as cm

class Robot():
    def __init__(self, pose, goal, this_map, sonar):
	''' Create an instance of robot
	Args:
	    pose - 1x3 array, initial x, y, phi pose
	    this_map - Ogmap
	    sonar - Sonar
	    '''
        self.pose = np.array(pose)
        self.vel = np.array([0,0,0])
        self.this_map = this_map
        self.sonar = sonar
        
        self.goal = goal
        self.goal_radius = 3
        self.goal_attained = False
        self.crashed = False

        self.vel_max = 1
        self.omega_max = 0.1
        self.displacement_slowdown = 25
        self.avoid_threshold = 5
        self.flee_vec = np.array((0,0))
        
    def command(self, control_x, control_v):
        x0, y0, phi = self.pose
        vx, vy, omega = self.vel
        vr = sqrt(vx**2 + vy**2)
        vr = min(vr, self.this_map.ray_trace(self.pose, 0, self.vel_max))
        if self.this_map.ray_trace(self.pose, 0, self.vel_max) < vr:
            self.crashed = True
        else:
            self.dx = self.vel
            self.pose = self.pose + self.dx + control_x
            self.vel = self.vel + control_v
    
    def measure(self):
        scan = self.sonar.simulate_scan(self.pose, self.this_map)
        try:
            self.last_scan = self.sonar.maxmin_filter(scan)
        except ValueError, BadScanError:
            pass
        
    def estimate_state(self):
        """return best guess of robot state"""
        return (self.pose, self.vel)

    def flee_vector(self):
        """return unit vector for avoiding obstacles"""
        eps = 1
        x0, y0, phi = self.pose
        pings = self.last_scan.pings
        xs = [cos(ping[0]+phi) / (ping[1]+eps) for ping in pings]
        ys = [sin(ping[0]+phi) / (ping[1]+eps) for ping in pings]
        avoid_vec = (-1*np.sum(xs), -1*np.sum(ys))
        return (avoid_vec / np.linalg.norm(avoid_vec))

    def control_policy(self):
        '''return appropriate control vectors'''
        control_x = np.array([0,0,0])
        pos_guess, vel_guess = self.estimate_state()
        displacement = (self.goal-pos_guess)[0:2]
        displacement_norm = np.linalg.norm(displacement)
        self.flee_vec = self.flee_vector()

        if min(self.last_scan.rs) < self.avoid_threshold:
            print min(self.last_scan.rs)
            print 'AVOID!'
            vel_des_rect = self.flee_vec
        else:
            vel_des_rect = displacement / displacement_norm
        vx_des = vel_des_rect[0]
        vy_des = vel_des_rect[1]
        phi_des = np.arctan2(vy_des, vx_des)
        phi_guess = pos_guess[2]
        slowdown_factor = (1 -
            exp(-displacement_norm/self.displacement_slowdown))
        vel_des_rect *= self.vel_max * slowdown_factor
        vel_des_phi = self.omega_max*(phi_des%(2*pi) - phi_guess%(2*pi))
        vel_des = np.array((vel_des_rect[0], vel_des_rect[1], vel_des_phi))
        if displacement_norm <= self.goal_radius:
            vel_des = np.array((0,0))
            self.goal_attained = True
        control_v = vel_des - vel_guess

        return (control_x, control_v)
    
    def show_state(self):
        x0, y0, phi = self.pose
        plt.cla()
        plt.imshow(self.this_map.grid
                , interpolation='none'
                , cmap=cm.Greys_r
                , origin='lower'
                )
        plt.plot(self.goal[0]
                , self.goal[1]
                , '*', color='r'
                , markersize = 20)
        plt.plot(x0
                , y0
                , 'o', color='g'
                , markersize=10
                )
        plt.quiver(x0
                , y0
                , np.cos(phi)
                , np.sin(phi)
                )
        plt.quiver(x0
                , y0
                , self.flee_vec[0]
                , self.flee_vec[1]
                , color='r'
                )
        plt.plot(x0 + self.last_scan.rs*np.cos(self.last_scan.thetas + phi)
                , y0 + self.last_scan.rs*np.sin(self.last_scan.thetas + phi)
                , '.', color = 'y'
                , markersize = 10 
                )
        plt.xlim(0, 100) 
        plt.ylim(0, 100)
        plt.draw()
    
    def automate(self, numsteps = 100):
        for step in range(numsteps):
            if self.goal_attained:
                print 'GOAL REACHED'
                break
            if self.crashed:
                print 'CRASH!'
                break
            self.measure()
            self.show_state()
            control_x, control_v = self.control_policy()
            self.command(control_x, control_v)
           
if __name__ == "__main__":
    print """Legend:
        Yellow star\t -\t True position of robot
        Blue arrows\t -\t Particle cloud
        Yellow dots\t -\t Sonar pings
        Green boxes\t -\t Obstacles
        Red star\t -\t Goal"""
    true_pose = (20, 90, pi)
    this_goal = (50,50,0)
    this_map = mapdef()
    this_sonar = ogmap.Sonar(NUM_THETA = 10, GAUSS_VAR = .01)
    this_robot = Robot(true_pose, this_goal, this_map, this_sonar)
    plt.ion()
    this_robot.automate()
