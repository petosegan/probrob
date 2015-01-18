#!/usr/bin/env python

"""
test_robot_probha - tests for the robot_probha module
"""

# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4

import pytest
import robot_probha
import robot
import ogmap
import mcl
import mapdef

import numpy as np
from math import pi

class TestRobot():
    def test_success(self):
        print '\n'
        NRUNS = 5
        successes = 0
        for ii in range(NRUNS):
            this_success = robot_probha.main()
            if this_success:
                successes += 1
        print "Success Rate: %d of %d"%(successes, NRUNS)
        assert(successes > 0)
