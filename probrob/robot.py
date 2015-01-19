#!/usr/bin/env python

"""
robot - base class for robot simulator
"""

# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
from math import pi, exp, sin, cos

import matplotlib.pyplot as plt

import ogmap
from mapdef import mapdef
from utils import *


def check_success(goal, robot):
    displacement = (goal.location - robot.pose)[0:2]
    distance_to_goal = np.linalg.norm(displacement)
    return distance_to_goal < 2 * goal.radius


# noinspection PyAttributeOutsideInit
class Robot():
    def __init__(self, parameters, sonar):

        """

        :param parameters:
        :param sonar:
        """
        self.parameters = parameters
        self.sonar = sonar

        self.vel_max = self.parameters.vel_max
        self.omega_max = self.parameters.omega_max
        self.displacement_slowdown = self.parameters.displacement_slowdown
        self.avoid_threshold = self.parameters.avoid_threshold
        self.fig, self.ax = plt.subplots()

    def situate(self, some_map, pose, goal):
        """

        :param some_map:
        :param pose:
        :param goal:
        """
        self.this_map = some_map
        self.pose = np.array(pose)
        self.goal = goal

        self.vel = np.array((0, 0, 0))

        self.goal_attained = False
        self.crashed = False
        self.this_map.show()
        self.goal.show()
        x0, y0, phi = self.pose
        self.flee_vector_artist = self.ax.quiver(x0, y0, 1, 0, color='r')
        self.pose_dot_artist, = self.ax.plot(x0, y0, 'o', color='g', markersize=10)
        self.pose_arrow_artist = self.ax.quiver(x0, y0, x0 * cos(phi), y0 * sin(phi))
        self.scan_artist, = self.ax.plot([], [], 'o', color='r', markersize=5)


    def automate(self, num_steps=100):
        """

        :param num_steps:
        """
        for step in range(num_steps):
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

    def measure(self):
        """


        """
        scan = self.sonar.simulate_scan(self.pose, self.this_map)
        try:
            self.last_scan = self.sonar.maxmin_filter(scan)
        except ValueError, BadScanError:
            pass

    def show_state(self):
        self.show_pose()
        self.show_flee_vector()
        self.show_last_scan()
        plt.xlim(0, self.this_map.gridsize)
        plt.ylim(0, self.this_map.gridsize)

        self.fig.canvas.flush_events()

    def show_pose(self):
        x0, y0, phi = self.pose
        self.pose_dot_artist.set_xdata(x0)
        self.pose_dot_artist.set_ydata(y0)
        self.pose_arrow_artist.set_offsets([x0, y0])
        self.pose_arrow_artist.set_UVC(cos(phi), sin(phi))

    def show_flee_vector(self):
        x0, y0, _ = self.pose
        fvx, fvy = self.flee_vector()
        self.flee_vector_artist.set_offsets([x0, y0])
        self.flee_vector_artist.set_UVC(fvx, fvy)

    def show_last_scan(self):
        self.scan_artist.set_xdata(self.last_scan.lab_exes)
        self.scan_artist.set_ydata(self.last_scan.lab_wyes)


    def control_policy(self):
        """return appropriate control vectors"""
        pos_guess, _ = self.estimate_state()
        displacement = (self.goal.location - pos_guess)[0:2]
        distance_to_goal = np.linalg.norm(displacement)

        if distance_to_goal <= self.goal.radius:
            vel_des_rect = np.array((0, 0))
            self.goal_attained = True
        elif self.last_scan.obst_distance < self.avoid_threshold:
            vel_des_rect = self.flee_vector()
        else:
            vel_des_rect = displacement / distance_to_goal
            vel_des_rect *= (1 - exp(-distance_to_goal / self.displacement_slowdown))

        return np.array((0, 0, 0)), self.vcontroller(vel_des_rect)

    def estimate_state(self):
        """return best guess of robot state"""
        return self.pose, self.vel

    def flee_vector(self):
        """return unit vector for avoiding obstacles"""
        eps = 0.25
        x0, y0, phi = self.pose
        pings = self.last_scan.pings
        xs = [-cos(ping[0] + phi) / (ping[1] + eps) ** 2 for ping in pings]
        ys = [-sin(ping[0] + phi) / (ping[1] + eps) ** 2 for ping in pings]
        avoid_vec = np.array((np.sum(xs), np.sum(ys)))
        return avoid_vec / np.linalg.norm(avoid_vec)

    def vcontroller(self, vel_des_rect):
        """


        :type vel_des_rect: 2-vector
        :param vel_des_rect:
        :return:
        """
        vel_des_phi = self.omega_des(vel_des_rect)
        vel_des = np.append(vel_des_rect, vel_des_phi)
        _, vel_guess = self.estimate_state()
        return vel_des - vel_guess

    def omega_des(self, vel_des_rect):
        """

        :rtype : float
        """
        return self.omega_max * (rect2phi(vel_des_rect) % (2 * pi) - self.estimate_state()[0][2] % (2 * pi))

    def command(self, control_x, control_v):
        """

        :param control_x:
        :param control_v:
        """
        self.dx = self.vel
        self.pose = self.pose + self.dx + control_x
        self.pose[2] %= 2 * pi
        self.vel = self.vel + control_v


class Parameters():
    def __init__(self
                 , vel_max=1
                 , omega_max=0.1
                 , displacement_slowdown=25
                 , avoid_threshold=5
    ):
        """



        :type omega_max: float
        :type vel_max: float
        :param vel_max:
        :param omega_max:
        :param displacement_slowdown:
        :param avoid_threshold:
        """
        self.vel_max = vel_max
        self.omega_max = omega_max
        self.displacement_slowdown = displacement_slowdown
        self.avoid_threshold = avoid_threshold


class Goal():
    def __init__(self, location, radius):
        self.location = np.array(location)
        self.radius = radius

    def show(self):
        plt.plot(self.location[0]
                 , self.location[1]
                 , '*'
                 , color='red'
                 , markersize=20
        )


def main():
    print """Legend:
        Yellow star\t -\t True position of robot
        Blue arrows\t -\t Particle cloud
        Yellow dots\t -\t Sonar pings
        Green boxes\t -\t Obstacles
        Red star\t -\t Goal"""

    these_parameters = Parameters(vel_max=1
                                  , omega_max=0.1
                                  , displacement_slowdown=5
                                  , avoid_threshold=5
    )
    true_pose = (20, 90, pi)
    this_goal = Goal(location=(50, 50, 0)
                     , radius=3)
    this_map = mapdef()
    this_sonar = ogmap.Sonar(num_theta=10
                             , gauss_var=.01
    )

    this_robot = Robot(these_parameters
                       , this_sonar
    )
    this_robot.situate(this_map
                       , true_pose
                       , this_goal
    )

    plt.ion()
    this_robot.automate()
    if check_success(this_goal, this_robot):
        print "SUCCESS"
    else:
        print "FAILURE"

if __name__ == "__main__":
    main()
