from math import pi
from probrob import ogmap
from probrob.mapdef import mapdef
import matplotlib.pyplot as plt

__author__ = 'Richard W. Turner'

import robot
import numpy as np
import sonar


class RobotMapper(robot.Robot):
    def __init__(self, parameters, this_sonar):
        robot.Robot.__init__(self, parameters, this_sonar)
        self.scan_map = ScanMap()

    def measure(self):
        robot.Robot.measure(self)
        # need to make sure the estimated pose is used here
        pose_guess, _ = self.estimate_state()
        relative_scan = sonar.Scan(pose_guess, self.last_scan.thetas, self.last_scan.rs)
        self.scan_map.add_scan(relative_scan)


class ScanMap():
    def __init__(self):
        self.ping_points = []  # will be a list of 2-tuple coordinates
        # I think this should own a figure?

    def add_scan(self, scan):
        # need to extract points from pose, rs, thetas
        lab_angles = scan.phi + scan.thetas
        lab_exes = scan.x0 + scan.rs * np.cos(lab_angles)
        lab_wyes = scan.y0 + scan.rs * np.sin(lab_angles)
        ping_coords = zip(lab_exes, lab_wyes)
        self.ping_points.extend(ping_coords)

    def show_map(self):
        pass


def main():
    print """Legend:
        Yellow star\t -\t True position of robot
        Blue arrows\t -\t Particle cloud
        Yellow dots\t -\t Sonar pings
        Green boxes\t -\t Obstacles
        Red star\t -\t Goal"""

    these_parameters = robot.Parameters(vel_max=1
                                  , omega_max=0.1
                                  , displacement_slowdown=5
                                  , avoid_threshold=5
    )
    true_pose = (80, 90, pi)
    this_goal = robot.Goal(location=(20, 20, 0)
                     , radius=3)
    this_map = mapdef()
    this_sonar = ogmap.Sonar(num_theta=20
                             , gauss_var=2
    )

    this_robot = RobotMapper(these_parameters
                       , this_sonar
    )
    this_robot.situate(this_map
                       , true_pose
                       , this_goal
    )

    plt.ion()
    this_robot.automate(num_steps=300)
    if robot.check_success(this_goal, this_robot):
        print "SUCCESS"
    else:
        print "FAILURE"
    plt.figure()
    ping_coords = this_robot.scan_map.ping_points
    plt.plot(*zip(*ping_coords), marker='o', linestyle='')
    plt.xlim(0, this_map.gridsize)
    plt.ylim(0, this_map.gridsize)
    plt.show(block=True)


if __name__ == '__main__':
    main()