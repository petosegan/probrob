import numpy as np
import ogmap
import locate
from mapdef import mapdef, NTHETA
import mcl
import matplotlib.pyplot as plt
from math import pi



class Robot():
    def __init__(self, pose, this_map, sonar, ensemble):
        self.pose = np.reshape(np.array(pose), (3, 1))
        self.vel = np.array([[0],[0]])
        self.this_map = this_map
        self.sonar = sonar
        self.ensemble = ensemble
        
        self.control_std = 0.01
        
        self.goal = (25, 25, pi)
        
    def command(self, control_x, control_v):
        self.ensemble.pf_update(control_x, control_v)
    
    def measure(self):
        scan = self.sonar.simulate_scan(self.pose, self.this_map)
        self.last_scan = scan
        self.ensemble.pf_sonar(scan, self.sonar, self.this_map)
        
    def control_policy(self):
        idx_guess = np.argmax(self.ensemble.weight)
        pos_guess = self.ensemble.x_ens[:, idx_guess]
        vel_guess = self.ensemble.v_ens[:, idx_guess]
        displacement = (self.goal-pos_guess)[0:2]
        vel_des_rect = displacement / np.linalg.norm(displacement)
        vx_des = vel_des_rect[0]
        vy_des = vel_des_rect[1]
        phi_des = np.arctan2(vy_des, vx_des)
        phi_guess = pos_guess[2]
        vel_des_pol = (3, .3*(phi_des%(2*pi) - phi_guess%(2*pi)))
        control_x = np.array([[0],[0],[0]])
        control_v = np.reshape(vel_des_pol - vel_guess, (2, 1))
        x0, y0, phi = self.pose
        vr, omega = self.vel
        self.dx = (vr*np.cos(phi), vr*np.sin(phi), omega)
        self.pose = self.pose + self.dx + control_x + np.random.normal(0,self.control_std, (3, 1))
        self.vel = self.vel + control_v
        return (control_x, control_v)
    
    def show_state(self):
        this_ens.show_map_scan(col = 'b', scan = self.last_scan, this_map = self.this_map, pose = self.pose)
        plt.plot(self.goal[0], self.goal[1], '*', color='r', markersize = 20)
        plt.draw()
    
    def automate(self, numsteps = 100):
        for step in range(numsteps):
            control_x, control_v = self.control_policy()
            self.command(control_x, control_v)
            self.measure()
            self.show_state()
            
if __name__ == "__main__":
    true_pose = (75, 60, pi)
    this_map = mapdef()
    this_sonar = ogmap.Sonar(NUM_THETA = 10, GAUSS_VAR = 1)
    this_ens = mcl.Ensemble(pose = true_pose, acc_var = np.array([[.001],[.001]]))
    this_robot = Robot(true_pose, this_map, this_sonar, this_ens)
    plt.ion()
    this_robot.automate()