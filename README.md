Probabilistic Robotics

What Is It?
-----------
This is a collection of modules written to demonstrate ideas from the book
'Probabilistic Robotics' by Thrun, Burgard, and Fox. The aim is to implement
Simultaneous Localization and Mapping (SLAM) for a simulated robot in a simple
environment. At this time the simulated robot is capable of Monte Carlo
Localization based on rangefinder data and autonomous goal-finding based on a hybrid automaton

Usage
----------
Run robot_probha.py to watch the robot navigate to a goal using Monte Carlo 
Localization.

File List
-----------
locate.py
mapdef.py
mcl.py
ogmap.py
robot.py
robot_prob.py
robot_ha.py
robot_probha.py
hybrid_automaton.py
navigator.py
ray_trace.c
ray_trace.pyx
ray_trace_setup.py
ray_trace.so
sonar.py
utils.py

TO DO
----------
Implement SLAM.
Improve packaging
