__author__ = 'Richard W. Turner'

from robot import Robot
import numpy as np
import sonar


class RobotMapper(Robot):
    def __init__(self, parameters, this_sonar):
        Robot.__init__(self, parameters, this_sonar)
        self.scan_map = ScanMap()

    def measure(self):
        Robot.measure(self)
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
