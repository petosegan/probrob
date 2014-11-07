import ogmap

class TestOGMap():
    def test_cross(self):
        a = np.array(1, 1)
        b = np.array(2, 2)
        assert ogmap.cross(a, b) == 0
        c = np.array(1, -1)
        assert ogmap.cross(a, c) == -2
