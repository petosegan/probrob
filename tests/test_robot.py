#!/usr/bin/env python

"""
test_robot - tests for the robot module
"""

# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4

import pytest
import robot
import ogmap

import numpy as np
from math import pi

def setup_module(module):
    """setup state for tests"""
    global test_robot

    parameters = robot.Parameters()
    this_sonar = ogmap.Sonar(NUM_THETA=1
                           , GAUSS_VAR = 0.1**2)
    test_robot = robot.Robot(parameters
                           , this_sonar
                           )

    this_pose = (0,0,0)
    this_goal = robot.Goal(location=(10,10,0)
            , radius=3)

    this_map = ogmap.OGMap(N=1)

    test_robot.situate(this_map, this_pose, this_goal)


class TestRobot():
    def test_command(self, monkeypatch):
        '''test response to control vectors.
        known_values: control_x, control_v, result_pose, result_vel'''

        known_values = ( ( (0,0,0), (0,0,0), (0,0,0), (0,0,0))
                , ( (1,0,0), (0,0,0), (1,0,0), (0,0,0))
                , ( (0,0,0), (0,0,pi), (0,0,0), (0,0,pi))
                , ( (0,0,pi), (0,0,0), (0,0,pi), (0,0,0))
                , ( (0,0,4*pi), (0,0,0), (0,0,0), (0,0,0))
                )
        def mock_raytrace(pose, theta, rmax):
            return 5
        monkeypatch.setattr(test_robot.this_map, 'ray_trace', mock_raytrace)
        for control_x, control_v, result_pose, result_vel in known_values:
            test_robot.pose = np.array((0,0,0))
            test_robot.vel = np.array((0,0,0))
            test_robot.command(np.array(control_x)
                    , np.array(control_v))
            np.testing.assert_array_equal(test_robot.pose, np.array(result_pose))
            np.testing.assert_array_equal(test_robot.vel, np.array(result_vel))

    def test_evolve(self, monkeypatch):
        '''test evolution in absence of control.
        known_values: initial_pose, initial_velocity, result_pose, result_vel'''

        known_values = ( ( (0,0,0), (0,0,0), (0,0,0), (0,0,0))
                , ( (1,0,0), (0,0,0), (1,0,0), (0,0,0))
                , ( (0,0,0), (0,0,pi), (0,0,pi), (0,0,pi))
                , ( (0,0,pi), (0,0,0), (0,0,pi), (0,0,0))
                , ( (0,0,4*pi), (0,0,0), (0,0,0), (0,0,0))
                , ( (0,0,pi), (0,0,2*pi), (0,0,pi), (0,0,2*pi))
                , ( (0,0,0), (0,0,-pi), (0,0,pi), (0,0,-pi))
                , ( (0,0,0), (0,-1,0), (0,-1,0), (0,-1,0))
                )
        def mock_raytrace(pose, theta, rmax):
            return 5
        monkeypatch.setattr(test_robot.this_map, 'ray_trace', mock_raytrace)
        for initial_pose, initial_vel, result_pose, result_vel in known_values:
            test_robot.pose = np.array(initial_pose)
            test_robot.vel = np.array(initial_vel)
            test_robot.command(np.array((0,0,0))
                    , np.array((0,0,0)))
            np.testing.assert_array_equal(test_robot.pose, np.array(result_pose))
            np.testing.assert_array_equal(test_robot.vel, np.array(result_vel))
