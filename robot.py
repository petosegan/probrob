#!/usr/bin/env python

"""
robot - base class for robot simulator
"""

# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
import numpy as np
import ogmap
from mapdef import mapdef, NTHETA
import matplotlib.pyplot as plt
from math import pi, exp, sin, cos, sqrt
import matplotlib.cm as cm
from utils import *


class Robot():
    def __init__(self, parameters, sonar):

        self.parameters = parameters
        self.sonar = sonar

        self.vel_max = self.parameters.vel_max
        self.omega_max = self.parameters.omega_max
        self.displacement_slowdown = self.parameters.displacement_slowdown
        self.avoid_threshold = self.parameters.avoid_threshold

    def situate(self, this_map, pose, goal):
        self.this_map = this_map
        self.pose = np.array(pose)
        self.goal = goal

        self.vel = np.array((0,0,0))

        self.goal_attained = False
        self.crashed = False

    def automate(self, numsteps=100):
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

    def measure(self):
        scan = self.sonar.simulate_scan(self.pose, self.this_map)
        try:
            self.last_scan = self.sonar.maxmin_filter(scan)
        except ValueError, BadScanError:
            pass

    def show_state(self):
        plt.cla()
        self.this_map.show()
        self.goal.show()
        self.show_pose()
        self.show_flee_vector()
        self.last_scan.show_scan()
        plt.xlim(0, 100) 
        plt.ylim(0, 100)
        plt.draw()

    def show_pose(self):
        x0, y0, phi = self.pose
        plt.plot(x0
                , y0
                , 'o', color='g'
                , markersize=10
                )
        plt.quiver(x0
                , y0
                , np.cos(phi)
                , np.sin(phi)
                )

    def show_flee_vector(self):
        x0, y0, _ = self.pose
        plt.quiver(x0
                , y0
                , self.flee_vector()[0]
                , self.flee_vector()[1]
                , color='r'
                )

    def control_policy(self):
        '''return appropriate control vectors'''
        pos_guess, _  = self.estimate_state()
        displacement = (self.goal.location-pos_guess)[0:2]
        distance_to_goal = np.linalg.norm(displacement)

        if distance_to_goal <= self.goal.radius:
            vel_des_rect = np.array((0,0))
            self.goal_attained = True
        elif self.last_scan.obst_distance < self.avoid_threshold:
            vel_des_rect = self.flee_vector()
        else:
            vel_des_rect = displacement / distance_to_goal
            vel_des_rect *= (1 - exp(-distance_to_goal / self.displacement_slowdown))

        return (np.array((0,0,0)), self.vcontroller(vel_des_rect))
        
    def estimate_state(self):
        """return best guess of robot state"""
        return (self.pose, self.vel)
 
    def flee_vector(self):
        """return unit vector for avoiding obstacles"""
        eps = 0.25
        x0, y0, phi = self.pose
        pings = self.last_scan.pings
        xs = [-cos(ping[0]+phi) / (ping[1]+eps)**2 for ping in pings]
        ys = [-sin(ping[0]+phi) / (ping[1]+eps)**2 for ping in pings]
        avoid_vec = np.array((np.sum(xs), np.sum(ys)))
        return (avoid_vec / np.linalg.norm(avoid_vec))

    def vcontroller(self, vel_des_rect):
        vel_des_phi = self.omega_des(vel_des_rect)
        vel_des = np.append(vel_des_rect, vel_des_phi)
        _, vel_guess = self.estimate_state()
        return vel_des - vel_guess

    def omega_des(self, vel_des_rect):
        return self.omega_max * (rect2phi(vel_des_rect) % (2*pi) - self.estimate_state()[0][2] % (2*pi))

    def command(self, control_x, control_v):
        x0, y0, phi = self.pose
        vx, vy, omega = self.vel
        vr = np.linalg.norm((vx, vy))
        forward_obstacle_distance = self.this_map.ray_trace(self.pose, 0, self.vel_max)
        vr = min(vr, forward_obstacle_distance)
        if forward_obstacle_distance < vr:
            self.crashed = True
        else:
            self.dx = self.vel
            self.pose = self.pose + self.dx + control_x
            self.vel = self.vel + control_v


class Parameters():
    def __init__(self, vel_max, omega_max, displacement_slowdown, avoid_threshold):
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


if __name__ == "__main__":
    print """Legend:
        Yellow star\t -\t True position of robot
        Blue arrows\t -\t Particle cloud
        Yellow dots\t -\t Sonar pings
        Green boxes\t -\t Obstacles
        Red star\t -\t Goal"""

    these_parameters = Parameters(vel_max=1
            , omega_max=0.1
            , displacement_slowdown=25
            , avoid_threshold=5
            )
    true_pose = (20, 90, pi)
    this_goal = Goal(location=(50,50,0)
            , radius=3)
    this_map = mapdef()
    this_sonar = ogmap.Sonar(NUM_THETA = 10
            , GAUSS_VAR = .01
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
