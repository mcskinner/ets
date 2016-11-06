import datasets
import ets
import models

import numpy as np
import scipy.optimize
import tensorflow as tf


# Meta-meta-parameters and debugging knobs.
training_epochs = 10000
half_life = 1500
cost_growth = 1.5
base_learn = 0.005
base_cost_weight = 5.0

display_step = 100


# Set up the input data.
ys_raw = datasets.ukcars
mean_ys = np.mean(ys_raw)
ys = ys_raw / mean_ys
data = tf.placeholder('float', ys.shape)


# And then build the model.
global_step = tf.Variable(0, name = 'global_step')
cost_weight = tf.train.exponential_decay(base_cost_weight, global_step, half_life, cost_growth)
state0, varz, cost = models.BaselineState(cost_weight)(data)


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


# Gradient descent, with decaying learning rate.
def GradientDescent():
    # Meta-parameters for the optimization, themselves governed by the meta-meta-parameters above.
    learning_rate = tf.train.exponential_decay(base_learn, global_step, half_life, 0.5)
    optimizer = tf.train.AdamOptimizer(learning_rate, beta2=0.9).minimize(cost, global_step=global_step)

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
        PrintDiagnostics(sess, ys, 'Training')


GradientDescent()