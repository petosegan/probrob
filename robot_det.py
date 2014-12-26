''' Deterministic Robot
Robot with perfect knowledge of location'''

import numpy as np
import ogmap
import locate
from mapdef import mapdef, NTHETA
import mcl
import matplotlib.pyplot as plt
from math import pi, exp, sin, cos
from random import randint
from robot import Robot

class RobotDet(Robot):
    def __init__(self, pose, this_map, sonar, ensemble):
        Robot.__init__(self, pose, this_map, sonar, ensemble)

    def command(self, control_x, control_v):
        x0, y0, phi = self.pose
        vr, omega = self.vel
        vr = min(vr, self.this_map.ray_trace(self.pose, 0, self.vel_max))
        if self.this_map.ray_trace(self.pose, 0, self.vel_max) < vr:
            self.crashed = True
        else:
            self.dx = (vr*np.cos(phi), vr*np.sin(phi), omega)
            self.pose = self.pose + self.dx + control_x
            self.vel = self.vel + control_v
        
    def estimate_state(self):
        """return best guess of robot state"""
        return (self.pose, self.vel)

if __name__ == "__main__":
    print """Legend:
        Yellow star\t -\t True position of robot
        Blue arrows\t -\t Particle cloud
        Yellow dots\t -\t Sonar pings
        Green boxes\t -\t Obstacles
        Red star\t -\t Goal"""
    true_pose = (50,30,-90) # fails without obstacle avoidance
    this_map = mapdef()
    this_sonar = ogmap.Sonar(NUM_THETA = 10, GAUSS_VAR = .0001)
    this_ens = mcl.Ensemble(pose = true_pose
                        , acc_var = np.array([[.00001],[.00001]]))
    this_robot = RobotDet(true_pose, this_map, this_sonar, this_ens)
    plt.ion()
    this_robot.automate()
