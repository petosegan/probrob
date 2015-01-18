#!/usr/bin/env python

"""
robot_ha - class for robot simulator with hybrid automaton architecture
"""

# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4

import numpy as np
import matplotlib.pyplot as plt
import ogmap
from mapdef import mapdef, NTHETA
from math import pi, exp, sin, cos
from navigator import Navigator
from robot import Robot, Goal, Parameters


class RobotHA(Robot):
    def __init__(self, parameters, sonar):
        Robot.__init__(self, parameters, sonar) 

        self.navigator = Navigator()
        self.guard_fatness = 5

        self.fixed_params = {'omega_max': self.parameters.omega_max
                            ,'vel_max'  : self.parameters.vel_max
                            ,'slowdown_radius': self.parameters.displacement_slowdown
                            ,'guard_fatness': self.guard_fatness
                            ,'flee_threshold': self.parameters.avoid_threshold
                            }

    def situate(self, this_map, pose, goal):
        Robot.situate(self, this_map, pose, goal)
        self.fixed_params['goal_radius'] = self.goal.radius

    def control_policy(self):
        '''return appropriate control vectors'''
        control_x = np.array([0,0,0])
        pos_guess, vel_guess = self.estimate_state()
        displacement = (self.goal.location-pos_guess)[0:2]
        phi_guess = pos_guess[2]

        estimated_state = {'phi_guess':phi_guess
                          ,'vel_guess':vel_guess
                          ,'goal_vector':displacement
                          ,'flee_vector':self.flee_vector()
                          ,'obst_distance':self.last_scan.obst_distance
                          }

        robot_state = self.fixed_params.copy()
        robot_state.update(estimated_state)
        robot_state.update(self.navigator.state)

        policy = self.navigator.update(robot_state)
        #print self.navigator.current_behavior.name
        if self.navigator.current_behavior.name == "Goal-reached":
            self.goal_attained = True

        control_v = self.vcontroller(policy(**robot_state))

        return (control_x, control_v)


if __name__ == "__main__":
    print """Legend:
        Yellow star\t -\t True position of robot
        Blue arrows\t -\t Particle cloud
        Yellow dots\t -\t Sonar pings
        Green boxes\t -\t Obstacles
        Red star\t -\t Goal"""

    parameters = Parameters(vel_max=1
        , omega_max=0.1
        , displacement_slowdown=25
        , avoid_threshold=6
        )
    
    true_pose = (20,90,pi) 
    this_goal = Goal(location=(50,50,0)
            , radius=3)
    this_map = mapdef()
    this_sonar = ogmap.Sonar(NUM_THETA = 16
            , GAUSS_VAR = 0.01
            )
    this_robot = RobotHA(parameters, this_sonar)
    this_robot.situate(this_map
            , true_pose
            , this_goal
            )

    plt.ion()
    this_robot.automate(500)
