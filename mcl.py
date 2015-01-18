import numpy as np
from math import pi
from scipy.stats import norm as norm_dist

import matplotlib.pyplot as plt

import ogmap

import locate


num_steps = 500
N_PART = 100
win_size = 100


class Ensemble():
    """ Container for particles used in monte carlo localization """

    def __init__(self
                 , pose=(0, 0, 0)
                 , nn=N_PART
                 , acc_var=np.array((0.0001
                                     , 0.0001
                                     , 0.0001))
                 , meas_var=np.array((0.0001 ** 2
                                      , 0.0001 ** 2
                                      , 0.0001 ** 2
            ))
                 , diff_std=np.array((.5
                                      , .5
                                      , .1
            ))
    ):
        self.N = nn
        self.pose = np.array(pose)
        self.x_ens = np.tile(pose, (nn, 1))
        self.v_ens = np.zeros((nn, 3))
        self.dx = np.zeros((nn, 3))
        self.weight = np.ones(nn) / nn
        self.acc_std = np.sqrt(acc_var)
        self.meas_std = np.sqrt(meas_var)
        self.diff_std = diff_std

    def pf_update(self, this_control_x, this_control_v):
        """ Carry out update step of a particle filter algorithm"""
        num_part = self.x_ens.shape[0]
        acc = np.random.normal(0, self.acc_std, (num_part, 3))
        diffusion = np.random.normal(0, self.diff_std, (num_part, 3))
        self.x_ens = self.x_ens + self.dx + np.tile(this_control_x, (num_part, 1)) + diffusion
        self.v_ens = self.v_ens + acc + this_control_v
        vx = self.v_ens[:, 0]
        vy = self.v_ens[:, 1]
        omega = self.v_ens[:, 2]
        self.dx = np.array([vx, vy, omega])
        self.dx = np.transpose(self.dx)

    def pf_measure(self, this_measure_x):
        """ Carry out measurement step of a particle filter algorithm"""
        weight_x = norm_dist.pdf(self.x_ens[:][0]
                                 , this_measure_x[0]
                                 , self.meas_std[0]
        )
        weight_y = norm_dist.pdf(self.x_ens[:][1]
                                 , this_measure_x[1]
                                 , self.meas_std[1]
        )
        weight_th = norm_dist.pdf(self.x_ens[:][2]
                                  , this_measure_x[2]
                                  , self.meas_std[2]
        )
        weight = weight_x * weight_y * weight_th
        self.weight = weight / np.sum(weight)  # normalize
        resample = np.random.choice(range(self.N)
                                    , self.N
                                    , p=self.weight
        )
        self.x_ens = np.array([self.x_ens[i, :] for i in resample])
        self.v_ens = np.array([self.v_ens[i, :] for i in resample])

    def pf_sonar(self, some_scan, some_sonar, some_map):
        """ Carry out measurement step of a particle filter algorithm
            , using sonar data """
        num_part = self.x_ens.shape[0]
        get_ll = lambda i: locate.scan_loglikelihood(self.x_ens[i, :], some_scan, some_map, some_sonar)
        lls = np.array(map(get_ll, xrange(num_part)))
        weight = 1.0 / np.abs(lls) ** 2
        bad_weights = np.isnan(weight)
        weight[bad_weights] = 0
        weight = weight / np.sum(weight)  # normalize
        resample = np.random.choice(range(num_part), self.N, p=weight)
        self.x_ens = np.array([self.x_ens[ii, :] for ii in resample])
        self.v_ens = np.array([self.v_ens[ii, :] for ii in resample])

    def inject_random(self, pose, some_scan, some_sonar, some_map, num_inject=10):
        """add particles at high likelihood locations"""
        ll_n = some_map.N / 4

        # calculate a coarse likelihood map, and process to remove points in
        # obstacles
        coarse_ll_map, coords = locate.loglike_map(pose, some_scan, some_map,
                                                   some_sonar, ll_n)
        coarse_ll_map = np.where(np.isnan(coarse_ll_map),
                                 np.zeros(coarse_ll_map.shape), coarse_ll_map)
        min_ll = np.min(coarse_ll_map)
        coarse_ll_map = np.where(coarse_ll_map == 0,
                                 min_ll * np.ones(coarse_ll_map.shape), coarse_ll_map)

        xs, ys = coords
        Xs, Ys = np.meshgrid(xs, ys)

        weight = np.exp(np.ravel(coarse_ll_map))
        weight = weight / np.sum(weight)
        sample = np.random.choice(range(len(weight)), num_inject, p=weight)
        best_xs = np.ravel(Xs)[sample]
        best_ys = np.ravel(Ys)[sample]
        # best_phis = [pose[2]]*num_inject
        best_phis = np.random.normal(pose[2], pi, num_inject)

        new_x_ens_x = np.append(self.x_ens[:, 0], best_xs)
        new_x_ens_y = np.append(self.x_ens[:, 1], best_ys)
        new_x_ens_phi = np.append(self.x_ens[:, 2], best_phis)
        self.x_ens = np.array([new_x_ens_x, new_x_ens_y, new_x_ens_phi])
        self.x_ens = np.transpose(self.x_ens)
        mean_v_ens_x = np.mean(self.v_ens[:, 0]) * np.ones(num_inject)
        mean_v_ens_y = np.mean(self.v_ens[:, 1]) * np.ones(num_inject)
        mean_v_ens_phi = np.mean(self.v_ens[:, 2]) * np.ones(num_inject)
        self.v_ens = np.array([np.append(self.v_ens[:, 0], mean_v_ens_x),
                               np.append(self.v_ens[:, 1], mean_v_ens_y),
                               np.append(self.v_ens[:, 2], mean_v_ens_phi)
        ])
        self.v_ens = np.transpose(self.v_ens)

        vx = self.v_ens[:, 0]
        vy = self.v_ens[:, 1]
        omega = self.v_ens[:, 2]
        self.dx = np.array([vx
            , vy
            , omega
        ])
        self.dx = np.transpose(self.dx)

    def show_scatter(self, col='b', win_size=win_size):
        plt.subplot(121)
        plt.cla()
        plt.scatter(self.x_ens[:][0], self.x_ens[:][0], color=col)
        plt.xlim(0, win_size)
        plt.ylim(0, win_size)
        plt.subplot(122)
        plt.cla()
        plt.xlim(-win_size / 10, win_size / 10)
        plt.ylim(-win_size / 10, win_size / 10)
        plt.draw()

    def show(self):
        phis = self.x_ens[:, 2]
        plt.quiver(self.x_ens[:, 0]
                   , self.x_ens[:, 1]
                   , np.cos(phis)
                   , np.sin(phis)
                   , color='b'
                   , alpha=0.1
        )

    def show_map_scan(self
                      , this_map
                      , scan
                      , pose
    ):
        true_x, true_y, true_phi = pose
        plt.cla()
        plt.plot(true_x
                 , true_y
                 , '*'
                 , color='y'
                 , markersize=10
        )
        self.show()
        this_map.show()
        scan.show()
        plt.xlim(0, this_map.N)
        plt.ylim(0, this_map.N)
        plt.draw()


if __name__ == "__main__":
    from mapdef import mapdef

    control_x = np.array([0, 0, 0])  # stay put
    control_v = np.array([0, 0, 0])
    measure_x = np.array([0, 0, 0])

    meas_rate = 10

    true_pose = (50, 50, 0)

    this_ens = Ensemble(pose=true_pose)
    this_sonar = ogmap.Sonar(num_theta=10, gauss_var=1)
    this_map = mapdef()

    plt.ion()
    fig = plt.figure()
    scan = this_sonar.simulate_scan(true_pose, this_map)
    for i in range(num_steps):
        print this_ens.x_ens.shape
        this_ens.pf_update(control_x, control_v)
        this_ens.show_map_scan(scan=scan
                               , this_map=this_map
                               , pose=true_pose
        )
        if i % meas_rate == 0:
            print this_ens.x_ens.shape
            scan = this_sonar.simulate_scan(true_pose, this_map)
            this_ens.pf_sonar(scan, this_sonar, this_map)
            this_ens.inject_random(true_pose, scan, this_sonar, this_map, 10)
            this_ens.show_map_scan(scan=scan
                                   , this_map=this_map
                                   , pose=true_pose
            )
            plt.draw()
