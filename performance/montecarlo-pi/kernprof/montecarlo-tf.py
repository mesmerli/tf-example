# montecarlo-tf.py

import tensorflow as tf
import matplotlib.pyplot as plt

@profile
def mondecarlo_pi_tf(iteration):
    trials = iteration
    hits = 0

    x = tf.random_uniform([1],minval=-1,maxval=1,dtype=tf.float32)
    y = tf.random_uniform([1],minval=-1,maxval=1,dtype=tf.float32)

    sess = tf.Session()

    with sess.as_default():
        for i in range(1,trials):
            if x.eval()**2 + y.eval()**2 < 1 :
                hits = hits + 1

mondecarlo_pi_tf(10000)
