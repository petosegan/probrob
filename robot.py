import numpy as np
import ogmap
import locate
from mapdef import mapdef, NTHETA
import mcl
import matplotlib.pyplot as plt



class Robot():
    def __init__(self, pose, this_map, sonar, ensemble):
        self.pose = np.reshape(np.array(pose), (2, 1))
        self.this_map = this_map
        self.sonar = sonar
        self.ensemble = ensemble
        
        self.control_std = 0.5
        
        self.goal = (25, 25)
        
    def command(self, control_x, control_v):
        self.ensemble.blind_particle_filter(control_x, control_v)
    
    def measure(self):
        scan = self.sonar.simulate_scan(self.pose, self.this_map)
        self.last_scan = scan
        self.ensemble.particle_filter_sonar(scan, self.sonar, self.this_map)
        
    def control_policy(self):
        pos_guess = self.ensemble.x_ens[:,np.random.choice(range(self.ensemble.N), p=self.ensemble.weight)]
        control_x = np.reshape(np.sign(self.goal - pos_guess), (2, 1))
        control_v = np.array([[0],[0]])
        self.pose = self.pose + control_x + np.random.normal(0,self.control_std, (2, 1))
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
    true_pose = (75, 60)
    this_map = mapdef()
    this_sonar = ogmap.Sonar(NUM_THETA = NTHETA, GAUSS_VAR = 1)
    this_ens = mcl.Ensemble(pose = true_pose, acc_var = np.array([[1],[1]]))
    this_robot = Robot(true_pose, this_map, this_sonar, this_ens)
    plt.ion()
    this_robot.automate()