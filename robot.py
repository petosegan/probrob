import numpy as np
import ogmap
import locate
from mapdef import mapdef, NTHETA
import mcl
import matplotlib.pyplot as plt
from math import pi, exp, sin, cos
from random import randint



class Robot():
    def __init__(self, pose, this_map, sonar, ensemble):
        self.pose = np.reshape(np.array(pose), (3, 1))
        self.vel = np.array([[0],[0]])
        self.this_map = this_map
        self.sonar = sonar
        self.ensemble = ensemble
        
        self.control_std = 0.01
        
        self.goal = (50, 50, pi)
        self.goal_radius = 3
        self.goal_attained = False
        self.crashed = False

        self.vel_max = 3
        self.omega_max = 0.3
        self.displacement_slowdown = 25
        self.avoid_threshold = 5
        
    def command(self, control_x, control_v):
        x0, y0, phi = self.pose
        vr, omega = self.vel
        vr = min(vr, self.this_map.ray_trace(self.pose, 0, self.vel_max))
        random_move =np.random.normal(0, self.control_std, (3, 1))
        random_dist = np.linalg.norm(random_move[0:2])
        if self.this_map.ray_trace(self.pose, 0, self.vel_max) < vr+random_dist:
            self.crashed = True
        else:
            self.dx = (vr*np.cos(phi), vr*np.sin(phi), omega)
            self.pose = self.pose + self.dx + control_x \
                + random_move 
            self.vel = self.vel + control_v
            try:
                self.ensemble.pf_update(control_x, control_v)
            except:
                pass
    
    def measure(self):
        scan = self.sonar.simulate_scan(self.pose, self.this_map)
        try:
            self.last_scan = self.sonar.maxmin_filter(scan)
            self.ensemble.pf_sonar(scan, self.sonar, self.this_map)
            pose_guess, _ = self.estimate_state()
            self.ensemble.inject_random(pose_guess, scan, self.sonar,
                self.this_map)
        except ValueError, BadScanError:
            pass
        
    def estimate_state(self):
        """return best guess of robot state"""
        idx_guess = np.argmax(self.ensemble.weight)
        pos_guess = self.ensemble.x_ens[:, idx_guess]
        vel_guess = self.ensemble.v_ens[:, idx_guess]
        return (pos_guess, vel_guess)

    def flee_vector(self):
        """return unit vector for avoiding obstacles"""
        pings = self.last_scan.pings
        xs = [cos(ping[0]) / (ping[1]+1) for ping in pings]
        ys = [sin(ping[0]) / (ping[1]+1) for ping in pings]
        avoid_vec = (np.sum(xs), np.sum(ys))
        return (avoid_vec / np.linalg.norm(avoid_vec))

    def control_policy(self):
        '''return appropriate control vectors'''
        control_x = np.array([[0],[0],[0]])
        pos_guess, vel_guess = self.estimate_state()
        displacement = (self.goal-pos_guess)[0:2]
        displacement_norm = np.linalg.norm(displacement)

        if min(self.last_scan.rs) < self.avoid_threshold:
            print min(self.last_scan.rs)
            print 'AVOID!'
            vel_des_rect = self.flee_vector()
        else:
            vel_des_rect = displacement / displacement_norm
        vx_des = vel_des_rect[0]
        vy_des = vel_des_rect[1]
        phi_des = np.arctan2(vy_des, vx_des)
        phi_guess = pos_guess[2]
        slowdown_factor = (1 -
            exp(-displacement_norm/self.displacement_slowdown))
        vel_des_r = self.vel_max * slowdown_factor
        vel_des_phi = self.omega_max*(phi_des%(2*pi) - phi_guess%(2*pi))
        vel_des_pol = (vel_des_r, vel_des_phi)
        if displacement_norm <= self.goal_radius:
            vel_des_pol = (0,0)
            self.goal_attained = True
        control_v = np.reshape(vel_des_pol - vel_guess, (2, 1))

        return (control_x, control_v)
    
    def show_state(self):
        this_ens.show_map_scan(col = 'b'
                            , scan = self.last_scan
                            , this_map = self.this_map
                            , pose = self.pose
                            )
        plt.plot(self.goal[0]
                , self.goal[1]
                , '*', color='r'
                , markersize = 20)
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
    true_pose = (randint(15, 90), randint(5, 65), pi)
    true_pose = (50,90,-90) # fails without obstacle avoidance
    this_map = mapdef()
    this_sonar = ogmap.Sonar(NUM_THETA = 10, GAUSS_VAR = .01)
    this_ens = mcl.Ensemble(pose = true_pose
                        , acc_var = np.array([[.001],[.001]]))
    this_robot = Robot(true_pose, this_map, this_sonar, this_ens)
    plt.ion()
    this_robot.automate()
