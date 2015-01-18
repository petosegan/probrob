from random import randrange

import ogmap


NTHETA = 200 # Number of angles in scan

def mapdef():

    test = ogmap.OGMap(100, '/home/will/projects/probrob/trace_cache.npy')
    test.rect(95, 0, 10, 50)
    test.rect(0, 0, 10, 50)
    test.rect(10, 70, randrange(10, 70), 10)

    test.rect(-1, -1, 1, 102)
    test.rect(100, -1, 1, 102)
    test.rect(0, 100, 100, 1)
    test.rect(0, -1, 100, -1)
   
    return test
