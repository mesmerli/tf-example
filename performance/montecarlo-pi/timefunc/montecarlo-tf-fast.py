# montecarlo-tf-fast.py

import tensorflow as tf
import numpy as np
import time

n_trials = 10000

tf.reset_default_graph()

hit = tf.Variable(0, name='hit')

def body(ctr):
    x = tf.random_uniform(shape=[2], name='x')
    r = tf.sqrt(tf.reduce_sum(tf.square(x)))
    is_inside = tf.cond(tf.less(r,1), lambda: tf.constant(1), lambda: tf.constant(0))
    hit_op = hit.assign_add(is_inside)
    with tf.control_dependencies([hit_op]):
        return ctr + 1

def condition(ctr):
    return ctr < n_trials

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    result = tf.while_loop(condition, body, [tf.constant(0)])

    start = time.time()
    sess.run(result)
    end = time.time()

    hits = hit.eval()

    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter\
        ("/tmp/montecarlo_tf_fast_logs",sess.graph)

print("Time taken {} s".format(end-start))
print("Value of Pi: {}".format(4.*hits/n_trials))
