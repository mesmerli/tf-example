# montecarlo-tf.py

import tensorflow as tf
import matplotlib.pyplot as plt
import time

trials = 10000000
hits = 0

x = tf.random_uniform([1],minval=-1,maxval=1,dtype=tf.float32)
y = tf.random_uniform([1],minval=-1,maxval=1,dtype=tf.float32)

sess = tf.Session()

start = time.time()

with sess.as_default():
    for i in range(1,trials):
        if x.eval()**2 + y.eval()**2 < 1 :
            hits = hits + 1

end = time.time()

print("Time taken: {} s".format(end-start))
print("Value for Pi: {}".format(4 * float(hits)/trials))
