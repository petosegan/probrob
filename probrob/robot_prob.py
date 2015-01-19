#!/usr/bin/env python

"""
robot_prob - robot simulation using probabilistic localization
"""

# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4

import numpy as np
from math import pi

import matplotlib.pyplot as plt

import ogmap
from mapdef import mapdef
import mcl
from robot import Robot, Goal, Parameters


class ParametersProb(Parameters):
    def __init__(self
                 , vel_max=3
                 , omega_max=0.1
                 , displacement_slowdown=10
                 , avoid_threshold=5
                 , control_std=0.01
    ):
        Parameters.__init__(self, vel_max, omega_max, displacement_slowdown, avoid_threshold)
        self.control_std = control_std


# noinspection PyAttributeOutsideInit
class RobotProb(Robot):
    def __init__(self, parameters, sonar):
        Robot.__init__(self, parameters, sonar)
        self.control_std = self.parameters.control_std

    # noinspection PyMethodOverriding
    def situate(self
                , some_map
                , this_pose
                , some_goal
                , some_ens):
        Robot.situate(self, some_map, this_pose, some_goal)
        self.ensemble = some_ens

    def command(self, control_x, control_v):
        Robot.command(self, control_x, control_v)

        forward_obstacle_distance = self.this_map.ray_trace(self.pose, 0, self.vel_max)
        random_move = np.random.normal(0, self.control_std, 3)
        random_dist = np.linalg.norm(random_move[0:2])
        if forward_obstacle_distance < random_dist:
            self.crashed = True
        elif self.this_map.collision(*self.pose[0:2]):
            self.crashed = True
        else:
            self.pose += random_move
            self.ensemble.pf_update(control_x, control_v)

    def measure(self):
        scan = self.sonar.simulate_scan(self.pose, self.this_map)
        try:
            self.last_scan = self.sonar.maxmin_filter(scan)
            self.ensemble.pf_sonar(scan, self.sonar, self.this_map)
            pose_guess, _ = self.estimate_state()
            # self.ensemble.inject_random(pose_guess, scan, self.sonar,
            # self.this_map)
        except ValueError, BadScanError:
            pass

    def estimate_state(self):
        """return best guess of robot state"""
        idx_guess = np.argmax(self.ensemble.weight)
        pos_guess = self.ensemble.x_ens[idx_guess, :]
        vel_guess = self.ensemble.v_ens[idx_guess, :]
        return pos_guess, vel_guess

    def show_state(self):
        x0, y0, phi = self.pose
        plt.cla()
        self.this_map.show()
        plt.plot(x0
                 , y0
                 , 'o'
                 , color='b'
                 , markersize=10
        )
        plt.quiver(x0
                   , y0
                   , np.cos(phi)
                   , np.sin(phi)
                   , color='g'
                   , headwidth=1.5
                   , headlength=10
        )
        self.last_scan.show()
        self.ensemble.show()
        self.goal.show()
        plt.xlim(0, self.this_map.gridsize)
        plt.ylim(0, self.this_map.gridsize)
        plt.draw()


if __name__ == "__main__":
    print """Legend:
        Yellow star\t -\t True position of robot
        Blue arrows\t -\t Particle cloud
        Yellow dots\t -\t Sonar pings
        Green boxes\t -\t Obstacles
        Red star\t -\t Goal"""

    parameters = ParametersProb()
    this_goal = Goal(location=(50, 50, pi)
                     , radius=3)
    true_pose = (50, 90, -90)
    this_map = mapdef()
    sonar_params = {'RMAX': 100
        , 'EXP_LEN': 0.1
        , 'r_rez': 2
    }
    this_sonar = ogmap.Sonar(num_theta=16
                             , gauss_var=.1
                             , params=sonar_params
    )
    this_ens = mcl.Ensemble(pose=true_pose
                            , nn=50
                            , acc_var=np.array((.0001, .0001, .0001))
                            , meas_var=np.array((.01, .01, .01)))
    this_robot = RobotProb(parameters, this_sonar)
    this_robot.situate(this_map, true_pose, this_goal, this_ens)

    plt.ion()
    fig = plt.figure()
    fig.set_size_inches(20, 20)
    plt.get_current_fig_manager().resize(1000, 1000)

    this_robot.automate(num_steps=100)
