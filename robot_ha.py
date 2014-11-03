import numpy as np
import ogmap
import locate
from mapdef_pocket import mapdef, NTHETA
import mcl
import matplotlib.pyplot as plt
from math import pi, exp, sin, cos
from random import randint
from navigator import navigator


class Robot():
    def __init__(self, pose, this_map, sonar, ensemble):
        self.pose = np.reshape(np.array(pose), (3, 1))
        self.vel = np.array([[0],[0]])
        self.this_map = this_map
        self.sonar = sonar
        self.ensemble = ensemble
        self.navigator = navigator
        
        self.control_std = 0.01
        
        self.goal = (40, 50, pi)
        self.goal_radius = 3
        self.goal_attained = False

        self.vel_max = 3
        self.omega_max = 0.3
        self.displacement_slowdown = 25
        self.avoid_threshold = 10 
        self.guard_fatness = 3

        self.fixed_params = {'omega_max': self.omega_max
                            ,'vel_max'  : self.vel_max
                            ,'slowdown_radius': self.displacement_slowdown
                            ,'guard_fatness': self.guard_fatness
                            ,'flee_threshold': self.avoid_threshold
                            ,'goal_radius': self.goal_radius
                            }
        
    def command(self, control_x, control_v):
        x0, y0, phi = self.pose
        vr, omega = self.vel
        self.dx = (vr*np.cos(phi), vr*np.sin(phi), omega)
        self.pose = self.pose + self.dx + control_x \
            + np.random.normal(0,self.control_std, (3, 1))
        self.vel = self.vel + control_v
        self.ensemble.pf_update(control_x, control_v)
    
    def measure(self):
        scan = self.sonar.simulate_scan(self.pose, self.this_map)
        self.last_scan = self.sonar.maxmin_filter(scan)
        self.ensemble.pf_sonar(scan, self.sonar, self.this_map)
        pose_guess, _ = self.estimate_state()
        self.ensemble.inject_random(pose_guess, scan, self.sonar,
                self.this_map)
        
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
        phi_guess = pos_guess[2]
        flee_vector = self.flee_vector()
        obst_distance = min(self.last_scan.rs)

        estimated_state = {'phi_guess':phi_guess
                          ,'vel_guess':vel_guess
                          ,'goal_vector':displacement
                          ,'flee_vector':flee_vector
                          ,'obst_distance':obst_distance
                          }

        robot_state = self.fixed_params.copy()
        robot_state.update(estimated_state)
        robot_state.update(self.navigator.state)
        print self.navigator.current_behavior.name
        if self.navigator.current_behavior.name == "Goal-reached":
            self.goal_attained = True

        policy = self.navigator.update(robot_state)
        control_v = policy(**robot_state)

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
        plt.ylim(0, 150)
        plt.xlim(-50, 150)
        plt.draw()
    
    def automate(self, numsteps = 100):
        for step in range(numsteps):
            if self.goal_attained:
                print 'GOAL REACHED'
                break
            self.measure()
            control_x, control_v = self.control_policy()
            self.command(control_x, control_v)
            self.show_state()
           
if __name__ == "__main__":
    print """Legend:
        Yellow star\t -\t True position of robot
        Blue arrows\t -\t Particle cloud
        Yellow dots\t -\t Sonar pings
        Green boxes\t -\t Obstacles
        Red star\t -\t Goal"""
    true_pose = (randint(15, 90), randint(5, 65), pi)
    true_pose = (90,50,0) # fails without obstacle avoidance
    this_map = mapdef()
    this_sonar = ogmap.Sonar(NUM_THETA = 10, GAUSS_VAR = 0.01)
    this_ens = mcl.Ensemble(pose = true_pose
                        , acc_var = np.array([[.001],[.001]]))
    this_robot = Robot(true_pose, this_map, this_sonar, this_ens)
    plt.ion()
    this_robot.automate()
