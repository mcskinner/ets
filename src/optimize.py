import datasets
import ets
import models

import numpy as np
import tensorflow as tf


# Meta-meta-parameters and debugging knobs.
training_epochs = 10000
half_life = 750
display_step = 100
gradient_clip = 2.0
hot_ticket = 1
cost_growth = 1.5
base_learn = 0.0025
peak_param_cost = 1


# Meta-parameters for the optimization, themselves governed by the meta-meta-parameters above.
global_step = tf.Variable(0)
learning_rate = tf.minimum(
    tf.train.exponential_decay(base_learn * 2, global_step, half_life * hot_ticket, 0.5),
    tf.train.exponential_decay(base_learn * (2**hot_ticket), global_step, half_life, 0.5))
cost_weight = tf.train.exponential_decay(peak_param_cost / (cost_growth**hot_ticket), global_step, half_life, cost_growth)


# Set up the input data.
ys_raw = datasets.ukcars
mean_ys = np.mean(ys_raw)
ys = ys_raw / mean_ys
data = tf.placeholder('float', ys.shape)


# And then build the model.
state0, varz, cost = models.BaselineState(cost_weight)(data)


# Gradient descent, with decaying learning rate.
optimizer = tf.train.AdamOptimizer(learning_rate, beta2=0.9).minimize(cost, global_step=global_step)


# Dump the state to the screen.
# TODO: does repeatedly calling sess.run(...) perturb the state?
def PrintDiagnostics(sess, ys, title):
    c = sess.run(cost, feed_dict={data: ys})
    print title, \
        "cost=", "{:.9f}".format(c), \
        "state0=", sess.run(tf.reshape(state0, [-1])), \
        "varz=", dict((k, sess.run(v)) for k, v in varz.iteritems())


# Initializing the variables
init = tf.initialize_all_variables()


# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    PrintDiagnostics(sess, ys, 'Init')

    # Fit the data.    
    for epoch in range(training_epochs):
        sess.run(optimizer, feed_dict={data: ys})

        # Display logs per epoch step
        if epoch % display_step == 0:
            PrintDiagnostics(sess, ys, 'Epoch: %04d' % epoch)

    print "Optimization Finished!"
    sess.run(optimizer, feed_dict={data: ys})
    PrintDiagnostics(sess, ys, 'Training')
