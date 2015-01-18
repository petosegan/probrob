import locate
import numpy as np

class TestLocate():
    def test_find_nearest(self):
        array = np.array([1, 2, 3])
        value = 0
        assert locate.find_nearest(array, value) == 1


