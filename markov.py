import numpy as np
import random

S = 0
C = 1
R = 2

TT = np.matrix('.8 .2 0; .4 .4 .2; .2 .6 .2')

N = 10000

counts = np.array([0.0, 0.0, 0.0])

state = R

random.seed()

for i in range(N):
    r = random.random()
    if r < TT[state, S]:
        state = S
        counts[S] += 1
    elif r < (TT[state, S] + TT[state, C]):
        state = C
        counts[C] += 1
    else:
        state = R
        counts[R] += 1

print counts / N