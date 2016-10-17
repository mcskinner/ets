import datasets
import ets
import models
import tensorflow as tf


# Meta-meta-parameters and debugging knobs.
training_epochs = 4000
half_life = 300
display_step = 100
gradient_clip = 0.5
hot_ticket = 2
cost_growth = 1.5
base_learn = 0.02
peak_param_cost = 5.0


# Meta-parameters for the optimization, themselves governed by the meta-meta-parameters above.
global_step = tf.Variable(0)
learning_rate = tf.minimum(
    tf.train.exponential_decay(base_learn * 2, global_step, half_life * hot_ticket, 0.5),
    tf.train.exponential_decay(base_learn * (2**hot_ticket), global_step, half_life, 0.5))
cost_weight = tf.train.exponential_decay(peak_param_cost / (cost_growth**hot_ticket), global_step, half_life, cost_growth)


# Set up the input data.
ys = datasets.bonds
data = tf.placeholder('float', ys.shape)


# And then build the model.
state0, params, varz, cost = models.Triangular(cost_weight)(data)


# Gradient descent, with decaying learning rate, and with gradient clipping.
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
gradients, v = zip(*optimizer.compute_gradients(cost))
gradients, _ = tf.clip_by_global_norm(gradients, gradient_clip)
optimizer = optimizer.apply_gradients(zip(gradients, v), global_step=global_step)


# Dump the state to the screen.
# TODO: does repeatedly calling sess.run(...) perturb the state?
def PrintDiagnostics(sess, ys, title):
    c = sess.run(cost, feed_dict={data: ys})
    print title, \
        "cost=", "{:.9f}".format(c), \
        "state0=", sess.run(tf.reshape(state0, [2])), \
        "params=", sess.run(tf.reshape(params, [2])), \
        "varz=", dict((k, sess.run(v)) for k, v in varz.iteritems())


# Initializing the variables
init = tf.initialize_all_variables()


# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Fit the data.    
    for epoch in range(training_epochs):
        sess.run(optimizer, feed_dict={data: ys})

        # Display logs per epoch step
        if epoch % display_step == 0:
            PrintDiagnostics(sess, ys, 'Epoch: %04d' % epoch)

    print "Optimization Finished!"
    sess.run(optimizer, feed_dict={data: ys})
    PrintDiagnostics(sess, ys, 'Training')
