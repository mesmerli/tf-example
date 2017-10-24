# mondecarlo-numpy.py

import random
import math

@profile
def mondecarlo_pi_numpy(iteration):
    count_inside = 0

    for count in range(0, iteration):
        d = math.hypot(random.random(), random.random())
        if d < 1: count_inside += 1
        count += 1

mondecarlo_pi_numpy(10000)
