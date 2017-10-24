# montecarlo-tf-fast.py

import tensorflow as tf
import numpy as np
from tensorflow.python.client import timeline

n_trials = 10000

tf.reset_default_graph()

hit = tf.Variable(0, name='hit')

# @profile
def body(ctr):
    x = tf.random_uniform(shape=[2], name='x')
    r = tf.sqrt(tf.reduce_sum(tf.square(x)))
    is_inside = tf.cond(tf.less(r,1), lambda: tf.constant(1), lambda: tf.constant(0))
    hit_op = hit.assign_add(is_inside)
    with tf.control_dependencies([hit_op]):
        return ctr + 1

# @profile
def condition(ctr):
    return ctr < n_trials

# @profile
def mondecarlo_pi_tf_fast(iteration):
    n_trials = iteration
    with tf.Session() as sess:
        # add additional options to trace the session execution
        options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        tf.global_variables_initializer().run()
        result = tf.while_loop(condition, body, [tf.constant(0)])
        sess.run(result, options=options, run_metadata=run_metadata)
        hits = hit.eval()
        # Create the Timeline object, and write it to a json file
        fetched_timeline = timeline.Timeline(run_metadata.step_stats)
        chrome_trace = fetched_timeline.generate_chrome_trace_format()
        with open('timeline_01.json', 'w') as f:
            f.write(chrome_trace)

mondecarlo_pi_tf_fast(10000)
