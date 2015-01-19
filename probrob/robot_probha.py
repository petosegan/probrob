#!/usr/bin/env python

"""
robot_probha - class for robot simulator with hybrid automaton architecture and probabilistic localization
"""

# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4

import robot_ha
import robot_prob
import robot
import math
import mapdef
import ogmap
import mcl
import numpy as np
import random
import matplotlib.pyplot as plt


# noinspection PyMethodOverriding
class RobotProbHA(robot_prob.RobotProb, robot_ha.RobotHA):
    def __init__(self, this_parameters, sonar):
        robot_prob.RobotProb.__init__(self, this_parameters, sonar)
        robot_ha.RobotHA.__init__(self, this_parameters, sonar)

    def situate(self
                , some_map
                , this_pose
                , some_goal
                , some_ens):
        robot_prob.RobotProb.situate(self, some_map, this_pose, some_goal, some_ens)
        self.fixed_params['goal_radius'] = self.goal.radius

    def control_policy(self):
        return robot_ha.RobotHA.control_policy(self)

    # def show_state(self):
    #     """
    #     Override show_state to disable visualization
    #     """
    #     pass


def main():
    """


    :return:
    """
    parameters = robot_prob.ParametersProb()
    this_goal = robot_prob.Goal(location=(random.randrange(20, 80), random.randrange(10, 60), math.pi)
                                , radius=3)
    true_pose = (random.randrange(10, 50), 90, 0.1)
    this_map = mapdef.mapdef()
    sonar_params = {'RMAX': 100
        , 'EXP_LEN': 0.1
        , 'r_rez': 2
    }
    this_sonar = ogmap.Sonar(num_theta=16
                             , gauss_var=0.1
                             , params=sonar_params
    )
    this_ens = mcl.Ensemble(pose=true_pose
                            , nn=100
                            , acc_var=np.array((0.001, 0.001, 0.001))
                            , meas_var=np.array((0.1, 0.1, 0.1)))
    this_robot = RobotProbHA(parameters, this_sonar)
    this_robot.situate(this_map, true_pose, this_goal, this_ens)

    plt.ion()

    # print "Robot Running"
    this_robot.automate(num_steps=100)
    # plt.close()
    if robot.check_success(this_goal, this_robot):
        print "SUCCESS"
        return True
    else:
        print "FAILURE"
        return False


if __name__ == "__main__":
    main()
