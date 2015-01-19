import cython
from math import sin, cos

@cython.cdivision(True)
def ray_trace(edges, pose, theta, rmax):
    ''' Test for intersection of a ray with edges in the map
    
    Args:
      pose (1x3 array): robot pose, as (x,y) position and heading (rad)
      theta (radian): heading of ray_trace, in the robot frame
      rmax (int): maximum range of ray tracing
    '''
    dists = []

    cdef float r0, r1, q0, q1
    cdef float den, pr0, pr1, cross_pr_s
    cdef float x0, y0, phi, s0, s1
    cdef float pr_dot_s, u, cross_pr_q, t

    x0, y0, phi = pose
    s0, s1 = cos(theta + phi), sin(theta + phi)
    for edge in edges:
        r0, r1 = edge[0], edge[1]
        q0, q1 = edge[2] - r0, edge[3] - r1
        den = q0*s1 - q1*s0
        pr0, pr1 = x0 - r0, y0 - r1
        cross_pr_s = pr0*s1 - pr1*s0
        if den == 0.0:
            if cross_pr_s == 0.0:
                pr_dot_s = pr0*s0 + pr1*s1
                if pr_dot_s < 0.0:
                    dists.append((pr0*pr0 + pr1*pr1)**0.5)
                    # print('parallel, intersecting')
                    continue
                dists.append(rmax)
                # print('parallel, non-intersecting')
                continue
        u = cross_pr_s / den
        if u > 1.0 or u < 0.0:
            dists.append(rmax)
            # print ('non-intersecting')
            continue
        cross_pr_q = pr0*q1 - pr1*q0
        t = cross_pr_q / den
        if t < 0.0:
            dists.append(rmax)
            # print('wrong side')
            continue
        dists.append(t)
        # print('intersection')
        # print t
    return min(dists)
 
