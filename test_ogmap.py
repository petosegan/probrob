import ogmap
import numpy as np
from math import pi
import pytest

def setup_module(module):
    """setup state for tests"""
    global pose
    global th
    global rs
    global scan
    global sonar

    pose = (0, 0, 0) #pose
    th = np.linspace(0, 2*pi, 10)
    rs = np.linspace(0, 100, 10)
    scan = ogmap.Scan(pose, th, rs)
    sonar = ogmap.Sonar()


class TestOGMap():
    def test_cross(self):
        a = np.array([1, 1])
        b = np.array([2, 2])
        assert ogmap.cross(a, b) == 0
        c = np.array([1, -1])
        assert ogmap.cross(a, c) == -2


class TestRect():
    def test_init(self):
        """test instantiation of Rect"""
        r = ogmap.Rect(0, 0, 10, 10)
        assert r.x0 == 0
        assert r.y0 == 0
        assert r.width == 10
        assert r.height == 10

    def test_collision(self):
        """test collision detection"""
        r = ogmap.Rect(0, 0, 10, 10)
        assert r.collision(5, 5)
        assert r.collision(0, 0)
        assert r.collision(0, 10)
        assert r.collision(10, 0)
        assert r.collision(10, 10)
        assert not r.collision(50, 50)

class TestScan():
    def test_init(self):
        """test instantiation of Scan"""
#        p = (0, 0, 0) #pose
#        th = np.linspace(0, 2*pi, 10)
#        rs = np.linspace(0, 100, 10)
#        s = ogmap.Scan(p, th, rs)
        assert scan.pose == pose
        assert scan.x0 == 0
        assert scan.y0 == 0
        assert scan.phi == 0
        np.testing.assert_array_equal(scan.thetas, th)
        np.testing.assert_array_equal(scan.rs, rs)
        assert scan.pings == zip(th, rs)
        assert 101 not in scan.rs

class TestSonar():
    def test_init(self):
        """test instantiation of Sonar"""

        # test for an unfortunate and hard to notice bug
	# namely, that probability densities integrate to one
        assert round(np.sum(sonar.p_exp) - 1, 10) == 0
        assert round(np.sum(sonar.p_uni) - 1, 10) == 0
        assert round(np.sum(sonar.p_max) - 1, 10) == 0
        assert round(np.sum(sonar.p_min) - 1, 10) == 0

    def test_maxmin_filter(self):
        """docstring for test_maxmin_filter"""
        scan_filt = sonar.maxmin_filter(scan)
        assert 0 not in scan_filt.rs
        assert 100 not in scan_filt.rs
        with pytest.raises(ogmap.BadScanError):
            bad_scan = ogmap.Scan(pose, [0], [0])
            bad_scan_filt = sonar.maxmin_filter(bad_scan)

    def test_simulate_scan(self):
        """test simulation of sonar scans"""
        pass
