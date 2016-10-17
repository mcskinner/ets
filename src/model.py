import copy
import itertools
import numpy as np
import tensorflow as tf
rng = np.random

# Recursively flatten a list.
def flatten(l):
    if l == []:
        return l
    if isinstance(l[0], list):
        return flatten(l[0]) + flatten(l[1:])
    return l[:1] + flatten(l[1:])


# Build out the TensorFlow model for an exponential state space model.
def ModelETS(w_in, F_in, var_init = {}, var_bounds = {}, soft_cost_weight = None, start_state = None, start_params = None):
    w_in = flatten(w_in)
    F_in = flatten(F_in)
    var_names = [v for v in itertools.chain(w_in, F_in) if isinstance(v, str)]
    n = len(w_in)

    var_init = copy.copy(var_init)
    var_bounds = copy.copy(var_bounds)
    for var in var_names:
        if var not in var_init:
            var_init[var] = rng.uniform(0, 1)
        if var not in var_bounds:
            var_bounds[var] = (0, 1)

    def model(data):
        num_steps = data.get_shape()[0]
        varz = dict((var, tf.Variable(var_init[var], name = var, dtype = tf.float32)) for var in var_names)
        get_var = lambda v: (isinstance(v, str) and varz[v]) or tf.constant(v, dtype = tf.float32)
        w_varz = tf.pack([get_var(v) for v in w_in])
        F_varz = tf.pack([get_var(v) for v in F_in])

        # ETS params
        if start_state is None:
            state0 = tf.Variable(tf.random_uniform([n, 1], -0.1, 0.1), dtype = tf.float32)
        else:
            state0 = tf.Variable(tf.cast(tf.reshape(flatten(start_state), [n, 1]), dtype = tf.float32))

        if start_params is None:
            params = tf.Variable(tf.random_uniform([n, 1], 0, 1), dtype = tf.float32)
        else:
            params = tf.Variable(tf.cast(tf.reshape(flatten(start_params), [n, 1]), dtype = tf.float32))
        w = tf.reshape(w_varz, shape = [n, 1])
        F = tf.reshape(F_varz, shape = [n, n])

        # Unrolled ETS loop
        outputs = []
        state = state0
        for time_step in range(num_steps):
            output = tf.matmul(tf.transpose(w), state)
            state = tf.matmul(F, state) + params * (data[time_step] - output)
            outputs.append(output)

        def softcost(var, bounds = (0, 1)):
            lo, hi = bounds
            neg_cost = -tf.minimum(var - lo, 0)
            big_cost = -tf.minimum(hi - var, 0)
            return neg_cost + big_cost

        # Sum of squared errors (SSE)
        pack_out = tf.reshape(tf.pack(outputs), data.get_shape())
        error_cost = tf.reduce_sum(tf.pow(pack_out - data, 2))
        if soft_cost_weight is not None:
            varz_cost = sum([softcost(v, var_bounds[k]) for k, v in varz.iteritems()])
            params_cost = tf.reduce_sum(softcost(params))
            cost = error_cost + soft_cost_weight * (varz_cost + params_cost)
        else:
            cost = error_cost
        return state0, params, varz, cost

    return model


# Basic models that you can get from the R forecast package.
SimpleETS = ModelETS([1], [[1]])
HoltsLinear = ModelETS([1, 1], [[1, 1], [0, 1]])
DampedTrend = ModelETS(
    [1, 'phi'],
    [[1, 'phi'], [0, 'phi']],
    var_bounds = {'phi': (0.8, 0.98)},
    soft_cost_weight = 5)

# Meta-meta-parameters and debugging knobs.
training_epochs = 4000
half_life = 300
display_step = 100
gradient_clip = 0.5
hot_ticket = 2
cost_growth = 1.5
base_learn = 0.0002
peak_param_cost = 5.0

# Test data, taken from the R expsmooth package (expsmooth::bonds).
bonds = np.asarray([
    5.83, 6.06, 6.58, 7.09, 7.31, 7.23, 7.43, 7.37, 7.6, 7.89, 8.12, 7.96,
    7.93, 7.61, 7.33, 7.18, 6.74, 6.27, 6.38, 6.6, 6.3, 6.13, 6.02, 5.79,
    5.73, 5.89, 6.37, 6.62, 6.85, 7.03, 6.99, 6.75, 6.95, 6.64, 6.3, 6.4,
    6.69, 6.52, 6.8, 7.01, 6.82, 6.6, 6.32, 6.4, 6.11, 5.82, 5.87, 5.89,
    5.63, 5.65, 5.73, 5.72, 5.73, 5.58, 5.53, 5.41, 4.87, 4.58, 4.89, 4.69,
    4.78, 4.99, 5.23, 5.18, 5.54, 5.9, 5.8, 5.94, 5.91, 6.1, 6.03, 6.26,
    6.66, 6.52, 6.26, 6, 6.42, 6.1, 6.04, 5.83, 5.8, 5.74, 5.72, 5.23,
    5.14, 5.1, 4.89, 5.13, 5.37, 5.26, 5.23, 4.97, 4.76, 4.55, 4.61, 5.07,
    5, 4.9, 5.28, 5.21, 5.15, 4.9, 4.62, 4.24, 3.88, 3.91, 4.04, 4.03,
    4.02, 3.9, 3.79, 3.94, 3.56, 3.32, 3.93, 4.44, 4.29, 4.27, 4.29, 4.26,
    4.13, 4.06, 3.81, 4.32, 4.7])

# Meta-parameters for the optimization, themselves governed by the meta-meta-parameters above.
global_step = tf.Variable(0)
learning_rate = tf.minimum(
    tf.train.exponential_decay(base_learn * 2, global_step, half_life * hot_ticket, 0.5),
    tf.train.exponential_decay(base_learn * (2**hot_ticket), global_step, half_life, 0.5))
cost_weight = tf.train.exponential_decay(peak_param_cost / (cost_growth**hot_ticket), global_step, half_life, cost_growth)

# A wide variety of less traditional models implemented after verifying that the basic results were sound.

# Simple damped model, no boundary constraints
#Model = ModelETS(
#    [1, 'phi'],
#    [[1, 'phi'], [0, 'phi']],
#    soft_cost_weight = None)

# Also simple model, basic boundary constraints but nothing special on phi.
#Model = ModelETS(
#    [1, 'phi'],
#    [[1, 'phi'], [0, 'phi']],
#    soft_cost_weight = cost_weight)

# More interesting damped model, with different damping rates for different places.
#Model = ModelETS(
#    [1, 'phi_w'],
#    [[1, 'phi_F'], [0, 'phi_F']],
#    soft_cost_weight = cost_weight)

# All the way loosened
#Model = ModelETS(
#    [1, 1],
#    [['a', 'b'], ['c', 'd']],
#    start_state = [2.05, 3.85],
#    start_params = [0.975, 0.25],
#    var_init = {'a': 0.815, 'b': 0.195, 'c': 0, 'd': 1},
#    soft_cost_weight = cost_weight)

# Hinted at by the results of the free parameter optimization
#Model = ModelETS(
#    [1, 1],
#    [[1, 'phi_b'], [0, 'phi_d']],
#    soft_cost_weight = cost_weight)

# Also hinted at as the best two-parameter form
#Model = ModelETS(
#    [1, 1],
#    [['a', 'b'], [0, 1]],
#    start_state = [1.9, 3.97],
#    start_params = [1.12, 0.05],
#    var_init = {'a': 0.827, 'b': 0.16},
#    soft_cost_weight = cost_weight)

# Probably the best formulation for expsmooth::bonds, in terms of AIC.
Model = ModelETS(
    [1, 1],
    [['a', 'b'], [0, 'd']],
    start_state = [1.9, 3.97],
    start_params = [1.12, 0.05],
    var_init = {'a': 0.827, 'b': 0.16, 'd': 0.9949},
    soft_cost_weight = cost_weight)

# Set up the placeholder for the input data, then build the model.
data = tf.placeholder('float', bonds.shape)
state0, params, varz, cost = Model(data)

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
        sess.run(optimizer, feed_dict={data: bonds})

        # Display logs per epoch step
        if epoch % display_step == 0:
            PrintDiagnostics(sess, bonds, 'Epoch: %04d' % epoch)

    print "Optimization Finished!"
    sess.run(optimizer, feed_dict={data: bonds})
    PrintDiagnostics(sess, bonds, 'Training')
