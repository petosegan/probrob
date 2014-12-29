#!/usr/bin/env python

# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
""" Navigator

This module defines a hybrid automaton that acts as a navigator for my robot
simulator.

This hybrid automaton design was inspired by the Coursera course "Control of
Mobile Robots" taught by Dr. Magnus Egerstedt of Georgia Tech.
"""

import hybrid_automaton as ha
import numpy as np
from math import exp, pi

MAX_GOAL_DISTANCE = 1000 # I assume goal distance is never larger than this

## Policies

def gtg_policy(slowdown_radius, vel_max, vel_guess, goal_vector, **kwargs):
    """control policy for go to goal behavior
    move directly towards goal, slowing down upon approach"""

    displacement_norm = np.linalg.norm(goal_vector)
    slowdown_factor = (1 - exp(-displacement_norm / slowdown_radius))
    vel_des_r = vel_max * slowdown_factor
    return vel_des_r * goal_vector / displacement_norm

def ao_policy(vel_max, vel_guess, flee_vector, **kwargs):
    """control_policy for avoid obstacle behavior
    flee from obstacles"""

    return vel_max * flee_vector

def fw_cc_policy(vel_guess, flee_vector, vel_max, **kwargs):
    """control policy for follow wall counter clockwise behavior
    Move normal to obstacle in ccw sense"""

    flee_x, flee_y = flee_vector
    fw_cc_vector = (-flee_y, flee_x)
    return vel_max * fw_cc_vector

def fw_c_policy(vel_guess, flee_vector, vel_max, **kwargs):
    """control policy for follow wall clockwise behavior
    Move normal to obstacle in cw sense"""

    flee_x, flee_y = flee_vector
    fw_c_vector = (flee_y, -flee_x)
    return vel_max * fw_c_vector

def goal_policy(vel_guess,**kwargs):
    """control policy for goal reached behavior
    Stop"""

    return np.array((0,0))

## Guard Conditions

def condition_fw_gtg(goal_vector, flee_vector, last_goal_distance, **kwargs):
    """condition for transition from follow wall to go-to-goal
    Stop wall following and seek the goal when the obstacle is on the opposite side of you from
    the goal, and you are closer to the goal than when wall following began."""

    distance = np.linalg.norm(goal_vector)
    direction = np.dot(goal_vector, flee_vector)
    return (distance < last_goal_distance and direction> 0)

def condition_fw_ao(obst_distance, flee_threshold, guard_fatness, **kwargs):
    """condition for transision from follow wall to avoid obstacle
    Stop wall following and avoid the obstacle when the distance to the
    obstacle is below the threshold"""

    return (obst_distance < (flee_threshold - 0.5*guard_fatness))

def condition_gtg_fw_cc(obst_distance, goal_vector, flee_vector, flee_threshold,
        guard_fatness, **kwargs):
    """condition for transition from go-to-goal to follow wall
    counter-clockwise
    Stop seeking the goal and start following the wall in ccw sense when the
    obstacle distance is below threshold and moving ccw will approach the goal"""

    flee_x, flee_y = flee_vector
    fw_cc_vector = (-flee_y, flee_x)
    direction = np.dot(goal_vector, fw_cc_vector)
    return (obst_distance < flee_threshold + 0.5*guard_fatness and direction >
            0)

def condition_gtg_fw_c(obst_distance, goal_vector, flee_vector, flee_threshold, guard_fatness,
        **kwargs):
    """condition for transition from go-to-goal to follow wall
    clockwise"""

    flee_x, flee_y = flee_vector
    fw_c_vector = (flee_y, -flee_x)
    direction = np.dot(goal_vector, fw_c_vector)
    return (obst_distance < flee_threshold + 0.5*guard_fatness and direction >
            0)

def condition_ao_fw_cc(obst_distance, goal_vector, flee_vector, flee_threshold, guard_fatness,
        **kwargs):
    """condition for transition from avoid obstacle to follow wall
    counter-clockwise
    Stop avoiding the obstacle and follow the wall when the obstacle distane is
    above threshold and moving ccw will approach the goal"""

    flee_x, flee_y = flee_vector
    fw_cc_vector = (-flee_y, flee_x)
    direction = np.dot(goal_vector, fw_cc_vector)
    return (obst_distance > flee_threshold - 0.5*guard_fatness and direction >
            0)


def condition_ao_fw_c(obst_distance, goal_vector, flee_vector, flee_threshold, guard_fatness,
        **kwargs):
    """condition for transition from avoid obstacle to follow wall
    clockwise"""

    flee_x, flee_y = flee_vector
    fw_c_vector = (flee_y, -flee_x)
    direction = np.dot(goal_vector, fw_c_vector)
    return (obst_distance > flee_threshold - 0.5*guard_fatness and direction >
            0)

def condition_gtg_goal(goal_vector, goal_radius, **kwargs):
    """condition for transition from go-to-goal to goal
    Stop seeking the goal when within the goal radius"""

    distance = np.linalg.norm(goal_vector)
    return (distance < goal_radius)


## Resets

def record_goal_distance(state, goal_vector, **kwargs):
    """record the goal distance upon entering a follow wall behavior"""

    goal_distance = np.linalg.norm(goal_vector)
    state['last_goal_distance'] = goal_distance
    return state

def no_reset(state, **kwargs):
    return state


## Behaviors

behavior_gtg = ha.Behavior('Go-to-goal',gtg_policy)
behavior_ao = ha.Behavior('Avoid-obstacle',ao_policy)
behavior_fw_cc = ha.Behavior('Follow-wall-ccw',fw_cc_policy)
behavior_fw_c = ha.Behavior('Follow-wall-cw',fw_c_policy)
behavior_goal = ha.Behavior('Goal-reached',goal_policy)


## Guards
# Defines the network structure of the navigator

guard_gtg_fw_cc = ha.Guard(condition_gtg_fw_cc, behavior_fw_cc,
        record_goal_distance)
guard_gtg_fw_c  = ha.Guard(condition_gtg_fw_c, behavior_fw_c, record_goal_distance)
guard_gtg_goal  = ha.Guard(condition_gtg_goal, behavior_goal, no_reset)
behavior_gtg.guards=[guard_gtg_fw_cc, guard_gtg_fw_c, guard_gtg_goal]

guard_fw_cc_gtg = ha.Guard(condition_fw_gtg, behavior_gtg, no_reset)
guard_fw_cc_ao  = ha.Guard(condition_fw_ao, behavior_ao, no_reset)
behavior_fw_cc.guards=[guard_fw_cc_gtg, guard_fw_cc_ao]

guard_fw_c_gtg  = ha.Guard(condition_fw_gtg, behavior_gtg,no_reset)
guard_fw_c_ao   = ha.Guard(condition_fw_ao, behavior_ao, no_reset)
behavior_fw_c.guards=[guard_fw_c_gtg, guard_fw_c_ao]

guard_ao_fw_cc  = ha.Guard(condition_ao_fw_cc, behavior_fw_cc,
        record_goal_distance)
guard_ao_fw_c   = ha.Guard(condition_ao_fw_c, behavior_fw_c, record_goal_distance)
behavior_ao.guards=[guard_ao_fw_cc, guard_ao_fw_c]

## Navigator
navigator = ha.HybridAutomaton([behavior_gtg, behavior_ao, behavior_fw_cc,
    behavior_fw_c], behavior_gtg, {'last_goal_distance':MAX_GOAL_DISTANCE})


if __name__ == "__main__":
    # A carelessly chosen set of test parameters
    robot_state = {'omega_max': 3,
                   'phi_guess': 0,
                   'slowdown_radius':5,
                   'vel_max': 5,
                   'vel_guess': (1, 1),
                   'displacement': (1, 1),
                   'flee_vector': (1, 1),
                   'goal_vector': (1, 1),
                   'flee_threshold': 3,
                   'guard_fatness': 1,
                   'goal_radius': 10,
                   'obst_distance': 1}
    print navigator.current_behavior.name
    navigator.update(robot_state)
    print navigator.current_behavior.name
