#!/usr/bin/env python

"""
utils - utility functions for my robot simulator
"""

# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4


import numpy as np

def rect2phi(rect_vec):
    return np.arctan2(rect_vec[1], rect_vec[0])
