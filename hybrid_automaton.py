'''Hybrid Automaton

This module implements a hybrid automaton architecture for control of a mobile
robot. The automaton is a finite state machine where each node is associated
with a control policy. Edges, corresponding to transitions between policies,
are activated when their GUARD condition becomes true, and alter the state via
their RESET action.'''

class HybridAutomaton():
    '''Represents a hybrid automaton'''
    def __init__(self, behaviors, initial_state):
        """create a hybrid automaton instance
        
        Args:
            behaviors (dict) - a dictionary of behavior_name (str) : Behavior
            pairs
            initial_state (Behavior) - the starting Behavior"""
        self.behaviors = behaviors
        self.status = initial_state

    def automate(self):
        """execute automation update
        
        Return:
            control_policy (fn) - a function for determining control outputs
            given the robot state"""
        for guard in self.status.guards:
            if guard.condition(robot_state):
                self.status = guard.new_behavior
                break
        control_policy = self.status.policy
        return control_policy

class Behavior():
    ''''Represents a node in a hybrid automaton'''
    def __init__(self, guards):
        """create a node instance"""
        self.guards = guards

    def policy(self, robot_state):
        abstract

class Guard():
    def __init__(self, condition, new_behavior):
        """create a guard edge instance"""
        self.condition = condition
        self.new_behavior = new_behavior
