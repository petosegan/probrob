'''Hybrid Automaton

This module implements a hybrid automaton architecture for control of a mobile
robot. The automaton is a finite state machine where each node is associated
with a control policy. Edges, corresponding to transitions between policies,
are activated when their GUARD condition becomes true, and alter the state via
their RESET action.'''

class HybridAutomaton():
    '''Represents a hybrid automaton'''
    def __init__(self, behaviors, initial_behavior, state):
        """create a hybrid automaton instance
        
        Args:
            behaviors (dict) - a dictionary of behavior_name (str) : Behavior
            pairs
            initial_state (Behavior) - the starting Behavior
            state (dict) - a dictionary of automaton state parameters"""
        self.behaviors = behaviors
        self.current_behavior = initial_behavior
        self.state = state

    def update(self, robot_state):
        """execute automation update

        Args:
            robot_state (dict) - a dictionary of robot state parameters
        
        Return:
            control_policy (fn) - a function for determining control outputs
            given the robot state"""
        for guard in self.current_behavior.guards:
            if guard.condition(**robot_state):
                self.current_behavior = guard.new_behavior
                self.state = guard.reset(self.state, **robot_state)
                break
        control_policy = self.current_behavior.policy
        return control_policy

class Behavior():
    ''''Represents a node in a hybrid automaton'''
    def __init__(self, name, policy, guards = []):
        """create a node instance
        
        Args:
            name (str) - descriptive label for behavior
            guards (list of Guards) - outgoing edges from behavior
            policy (fn) - a function for determining control outputs given the
            robot state"""
        self.name = name
        self.guards = guards
        self.policy = policy

    def add_guards(self, guard_list):
        self.guards.extend(guard_list)

class Guard():
    def __init__(self, condition, new_behavior, reset):
        """create a guard edge instance
        
        Args:
            condition (fn) - a function of robot state that triggers a behavior
            transition if true
            new_behavior (Behavior) - the successor behavior if condition is
            met
            reset (fn) - a function of automaton state that changes the
            automaton parameters upon a transition"""
        self.condition = condition
        self.new_behavior = new_behavior
        self.reset = reset
