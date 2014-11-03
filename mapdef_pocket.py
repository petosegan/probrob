import ogmap
import os

NTHETA = 200 # Number of angles in scan

def mapdef():

    test = ogmap.OGMap(100, 'pocket_trace_cache.npy')
    test.rect(60, 30, 10, 40)
    test.rect(30, 30, 30, 5)
    test.rect(30, 65, 30, 5)
    
    # Check if map has changed. If so, precache the ray traces
    if os.path.isfile('pocketmapchange.time'):
        with open("pocketmapchange.time", 'r') as f:
            last_change_time = float(f.read())
            new_change_time = os.path.getmtime('mapdef_pocket.py')
        if new_change_time > last_change_time + 1:
            with open("pocketmapchange.time", "w") as f:
                f.write(str(new_change_time)) # Update the map change monitor
            test.cache_traces(test.cache_file, NUM_THETA = NTHETA)
    else:
        with open("pocketmapchange.time", "w") as f:
            f.write(str(os.path.getmtime('mapdef_pocket.py')))  # initialize the map change monitor
    
    return test

