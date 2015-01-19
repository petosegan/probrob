from math import pi, sin, cos
import ogmap
from mapdef import mapdef

__author__ = 'will'

import robot
import matplotlib.pyplot as plt


class RobotPlotter(robot.Robot):
    # noinspection PyAttributeOutsideInit
    def situate(self, some_map, pose, goal):
        robot.Robot.situate(self, some_map, pose, goal)
        self.this_map.show()
        self.goal.show()
        x0, y0, phi = self.pose
        self.flee_vector_artist = self.ax.quiver(x0, y0, 1, 0, color='r')
        self.pose_dot_artist, = self.ax.plot(x0, y0, 'o', color='g', markersize=10)
        self.pose_arrow_artist = self.ax.quiver(x0, y0, x0 * cos(phi), y0 * sin(phi))
        self.scan_artist, = self.ax.plot([], [], 'o', color='r', markersize=5)

    def __init__(self, parameters, sonar):
        robot.Robot.__init__(self, parameters, sonar)
        self.fig, self.ax = plt.subplots()

    def show_state(self):
        self.show_pose()
        self.show_flee_vector()
        self.show_last_scan()
        plt.xlim(0, self.this_map.gridsize)
        plt.ylim(0, self.this_map.gridsize)

        #self.ax.draw_artist(self.pose_dot_artist)

        #self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def show_pose(self):
        x0, y0, phi = self.pose
        self.pose_dot_artist.set_xdata(x0)
        self.pose_dot_artist.set_ydata(y0)
        self.pose_arrow_artist.set_offsets([x0, y0])
        self.pose_arrow_artist.set_UVC(cos(phi), sin(phi))

    def show_flee_vector(self):
        x0, y0, _ = self.pose
        fvx, fvy = self.flee_vector()
        self.flee_vector_artist.set_offsets([x0, y0])
        self.flee_vector_artist.set_UVC(fvx, fvy)

    def show_last_scan(self):
        self.scan_artist.set_xdata(self.last_scan.lab_exes)
        self.scan_artist.set_ydata(self.last_scan.lab_wyes)


def main():
    print """Legend:
        Yellow star\t -\t True position of robot
        Blue arrows\t -\t Particle cloud
        Yellow dots\t -\t Sonar pings
        Green boxes\t -\t Obstacles
        Red star\t -\t Goal"""

    these_parameters = robot.Parameters(vel_max=1
                                        , omega_max=0.1
                                        , displacement_slowdown=25
                                        , avoid_threshold=5
    )
    true_pose = (20, 90, pi)
    this_goal = robot.Goal(location=(50, 50, 0)
                           , radius=3)
    this_map = mapdef()
    this_sonar = ogmap.Sonar(num_theta=10
                             , gauss_var=.01
    )

    this_robot = RobotPlotter(these_parameters
                              , this_sonar
    )
    this_robot.situate(this_map
                       , true_pose
                       , this_goal
    )

    plt.ion()
    this_robot.automate(num_steps=100)
    if robot.check_success(this_goal, this_robot):
        print "SUCCESS"
    else:
        print "FAILURE"


if __name__ == "__main__":
    main()
