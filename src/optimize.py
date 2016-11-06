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
data = tf.placeholder(tf.float32, ys.shape)


# And then build the model.
global_step = tf.Variable(0, name = 'global_step')
cost_weight = tf.train.exponential_decay(base_cost_weight, global_step, half_life, cost_growth)
state0, varz, cost = models.BaselineState(cost_weight)(data)


# Dump the state to the screen.
# TODO: does repeatedly calling sess.run(...) perturb the state?
def PrintDiagnostics(sess, ys, title, feeds):
    c = sess.run(cost, feed_dict=feeds)
    print title, \
        "cost=", "{:.9f}".format(c), \
        "state0=", sess.run(tf.reshape(state0, [-1]), feed_dict=feeds), \
        "varz=", dict((k, sess.run(v, feed_dict=feeds)) for k, v in varz.iteritems())


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

        # Fit the data.    
        for epoch in range(training_epochs):
            sess.run(optimizer, feed_dict={data: ys})

            # Display logs per epoch step
            if epoch % display_step == 0:
                PrintDiagnostics(sess, ys, 'Epoch: %04d' % epoch, {data: ys})

        print "Optimization Finished!"
        PrintDiagnostics(sess, ys, 'Training', {data: ys})


# Nelder-Mead Simplex algorithm, heuristic but efficient for low dimensionality.
def NelderMead():
    # Mild hack to exclude the global step / cost variables.
    all_varz = [var for var in tf.trainable_variables() if not var.name.startswith('global_step')]
    var_holders = {}
    var_assignments = []
    for var in all_varz:
        holder = tf.placeholder(var.dtype, var.get_shape())
        var_holders[var.name] = holder
        var_assignments.append(var.assign(holder))

    def GetFeeds(x):
        idx = 0
        feeds = {data: ys}
        for var in all_varz:
            shape = var.get_shape()
            n = int(np.product(shape))
            feeds[var.name] = np.reshape(x[idx:idx+n], shape)
            idx += n
        return feeds
        
    with tf.Session() as sess:
        def Cost(x):
            return sess.run(cost, feed_dict=GetFeeds(x))

        sess.run(init)
        packed = [tf.reshape(var, [-1]) for var in all_varz]
        x = scipy.optimize.fmin(Cost, np.concatenate(sess.run(packed)))
        PrintDiagnostics(sess, ys, 'Nelder-Mead', GetFeeds(x))


NelderMead()
