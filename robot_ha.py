import numpy as np
import ogmap
import locate
from mapdef_pocket import mapdef, NTHETA
import mcl
import matplotlib.pyplot as plt
from math import pi, exp, sin, cos
from random import randint
from navigator import navigator
from robot import Robot

class Robot_HA(Robot):
    def __init__(self, pose, this_map, sonar, ensemble):
        Robot.__init__(self, pose, this_map, sonar, ensemble) 

        self.navigator = navigator

        self.control_std = 0.01
        
        self.goal = (40, 50, pi)
        self.goal_radius = 3
        self.goal_attained = False

        self.vel_max = 3
        self.omega_max = 0.3
        self.displacement_slowdown = 25
        self.avoid_threshold = 5 
        self.guard_fatness = 3

        self.fixed_params = {'omega_max': self.omega_max
                            ,'vel_max'  : self.vel_max
                            ,'slowdown_radius': self.displacement_slowdown
                            ,'guard_fatness': self.guard_fatness
                            ,'flee_threshold': self.avoid_threshold
                            ,'goal_radius': self.goal_radius
                            }
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
    this_robot = Robot_HA(true_pose, this_map, this_sonar, this_ens)
    plt.ion()
    this_robot.automate()
