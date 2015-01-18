from robot_prob import RobotProb, ParametersProb, Goal
from robot_ha import Robot_HA
import matplotlib.pyplot as plt
from math import pi
from mapdef import mapdef, NTHETA
import ogmap
import mcl
import numpy as np

class RobotProbHa(RobotProb, Robot_HA):
    def __init__(self, parameters, sonar):
	RobotProb.__init__(self, parameters, sonar)
	Robot_HA.__init__(self, parameters, sonar)

    def situate(self
            , this_map
	    , this_pose
	    , this_goal
	    , this_ens):
        RobotProb.situate(self, this_map, this_pose, this_goal, this_ens)
	Robot_HA.situate(self, this_map, this_pose, this_goal)

    def control_policy(self):
        return Robot_HA.control_policy(self)	


if __name__ == "__main__":
    print """Legend:
        Yellow star \t -\t True Position of Robot
	Blue arrows \t -\t Particle Cloud
	Yellow dots \t -\t Sonar pings
	Green boxes \t -\t Obstacles
	Red star \t -\t Goal"""

    parameters = ParametersProb()
    this_goal = Goal(location=(50, 50, pi)
		    , radius = 3)
    true_pose = (50, 90, -90)
    this_map = mapdef()
    sonar_params = {'RMAX':100
		    , 'EXP_LEN':0.1
		    , 'r_rez':2
		    }
    this_sonar = ogmap.Sonar(NUM_THETA=16
		    , GAUSS_VAR=0.1
		    , params=sonar_params
		    )
    this_ens = mcl.Ensemble(pose=true_pose
		    , N=50
		    , acc_var = np.array((0.0001, 0.0001, 0.0001))
		    , meas_var = np.array((0.01, 0.01, 0.01)))
    this_robot = RobotProbHa(parameters, this_sonar)
    this_robot.situate(this_map, true_pose, this_goal, this_ens)

    plt.ion()
    fig = plt.figure()
    fig.set_size_inches(20,20)
    plt.get_current_fig_manager().resize(1000, 1000)

    this_robot.automate(numsteps=300)
