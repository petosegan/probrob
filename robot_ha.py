import numpy as np
import ogmap
import locate
from mapdef import mapdef, NTHETA
import mcl
import matplotlib.pyplot as plt
from math import pi, exp, sin, cos
from random import randint
from navigator import navigator
from robot import Robot

class Robot_HA(Robot):
    def __init__(self, pose, goal, this_map, sonar):
        Robot.__init__(self, pose, goal, this_map, sonar) 

        self.navigator = navigator
	self.avoid_threshold = 6
        self.guard_fatness = 5

        self.fixed_params = {'omega_max': self.omega_max
                            ,'vel_max'  : self.vel_max
                            ,'slowdown_radius': self.displacement_slowdown
                            ,'guard_fatness': self.guard_fatness
                            ,'flee_threshold': self.avoid_threshold
                            ,'goal_radius': self.goal_radius
                            }
    def control_policy(self):
        '''return appropriate control vectors'''
        control_x = np.array([0,0,0])
        pos_guess, vel_guess = self.estimate_state()
        displacement = (self.goal-pos_guess)[0:2]
        phi_guess = pos_guess[2]
        self.flee_vec = self.flee_vector()
        obst_distance = min(self.last_scan.rs)

        estimated_state = {'phi_guess':phi_guess
                          ,'vel_guess':vel_guess
                          ,'goal_vector':displacement
                          ,'flee_vector':self.flee_vec
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
    true_pose = (20,90,pi) 
    this_goal = (50,50,0)
    this_map = mapdef()
    this_sonar = ogmap.Sonar(NUM_THETA = 16, GAUSS_VAR = 0.01)
    this_robot = Robot_HA(true_pose, this_goal, this_map, this_sonar)
    plt.ion()
    this_robot.automate(500)
