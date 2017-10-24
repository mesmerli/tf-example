# mondecarlo-numpy.py

import random
import math
import time

count_inside = 0

start = time.time()

for count in range(0, 10000000):
    d = math.hypot(random.random(), random.random())
    if d < 1: count_inside += 1
count += 1

end = time.time()

print("Time taken: {} s".format(end-start))
print("Value for Pi: {}".format(4.0 * count_inside / count))
