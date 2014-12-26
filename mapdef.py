import ogmap
import os

NTHETA = 200 # Number of angles in scan

def mapdef():

    test = ogmap.OGMap(100, '/home/will/projects/probrob/trace_cache.npy')
    test.rect(95, 0, 10, 50)
    test.rect(0, 0, 10, 50)
    test.rect(10, 70, 70, 10)
    
    # Check if map has changed. If so, precache the ray traces
    if os.path.isfile('mapchange.time'):
        with open("mapchange.time", 'r') as f:
            last_change_time = float(f.read())
            new_change_time = os.path.getmtime('mapdef.py')
        if new_change_time > last_change_time + 1:
            with open("mapchange.time", "w") as f:
                f.write(str(new_change_time)) # Update the map change monitor
            test.cache_traces(test.cache_file, NUM_THETA = NTHETA)
    else:
        with open("mapchange.time", "w") as f:
            f.write(str(os.path.getmtime('mapdef.py')))  # initialize the map change monitor
    
    return test
