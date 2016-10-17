import copy
import itertools
import numpy
import tensorflow as tf


# Recursively flatten a list.
def flatten(l):
    if l == []:
        return l
    if isinstance(l[0], list):
        return flatten(l[0]) + flatten(l[1:])
    return l[:1] + flatten(l[1:])


# Build out the TensorFlow model for an exponential state space model.
def Model(w_in, F_in, var_init = {}, var_bounds = {}, soft_cost_weight = None, start_state = None, start_params = None):
    w_in = flatten(w_in)
    F_in = flatten(F_in)
    var_names = [v for v in itertools.chain(w_in, F_in) if isinstance(v, str)]
    n = len(w_in)

    var_init = copy.copy(var_init)
    var_bounds = copy.copy(var_bounds)
    for var in var_names:
        if var not in var_init:
            var_init[var] = numpy.random.uniform(0, 1)
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
