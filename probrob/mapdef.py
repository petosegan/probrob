from random import randrange

import ogmap


NTHETA = 200 # Number of angles in scan

def mapdef():

    test = ogmap.OGMap(100)
    test.rect(randrange(75, 95), 0, 1, randrange(10, 50))
    test.rect(randrange(10, 40), 0, 1, randrange(10, 50))
    test.rect(15, 70, randrange(10, 70), 1)
    test.rect(40, 55, 1, 20)

    test.rect(-1, -1, 1, 102)
    test.rect(100, -1, 1, 102)
    test.rect(0, 100, 100, 1)
    test.rect(0, -1, 100, -1)
   
    return test
