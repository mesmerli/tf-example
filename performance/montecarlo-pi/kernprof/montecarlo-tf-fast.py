# montecarlo-tf-fast.py

import tensorflow as tf
import numpy as np

n_trials = 10000

tf.reset_default_graph()

hit = tf.Variable(0, name='hit')

@profile
def body(ctr):
    x = tf.random_uniform(shape=[2], name='x')
    r = tf.sqrt(tf.reduce_sum(tf.square(x)))
    is_inside = tf.cond(tf.less(r,1), lambda: tf.constant(1), lambda: tf.constant(0))
    hit_op = hit.assign_add(is_inside)
    with tf.control_dependencies([hit_op]):
        return ctr + 1

@profile
def condition(ctr):
    return ctr < n_trials

@profile
def mondecarlo_pi_tf_fast(iteration):
    n_trials = iteration
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        result = tf.while_loop(condition, body, [tf.constant(0)])
        sess.run(result)
        hits = hit.eval()

mondecarlo_pi_tf_fast(10000)
