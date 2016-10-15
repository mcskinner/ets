import itertools
import numpy as np
import tensorflow as tf
rng = np.random

# recursively flatten a list
def flatten(l):
    if l == []:
        return l
    if isinstance(l[0], list):
        return flatten(l[0]) + flatten(l[1:])
    return l[:1] + flatten(l[1:])


# Build out the TensorFlow model for Holt's linear method.
def ModelETS(w_in, F_in):
    w_in = flatten(w_in)
    F_in = flatten(F_in)
    var_names = [v for v in itertools.chain(w_in, F_in) if isinstance(v, str)]
    n = len(w_in)

    def model(data):
        num_steps = data.get_shape()[0]
        varz = dict((var, tf.Variable(rng.uniform(0, 1), name = var, dtype = tf.float32)) for var in var_names)
        get_var = lambda v: (isinstance(v, str) and varz[v]) or tf.constant(v, dtype = tf.float32)
        w_varz = tf.pack([get_var(v) for v in w_in])
        F_varz = tf.pack([get_var(v) for v in F_in])

        # ETS params
        state0 = tf.Variable(tf.random_uniform([n, 1], -0.1, 0.1), dtype = tf.float32)
        params = tf.Variable(tf.random_uniform([n, 1], 0.1, 0.2), dtype = tf.float32)
        w = tf.reshape(w_varz, shape = [n, 1])
        F = tf.reshape(F_varz, shape = [n, n])

        # Unrolled ETS loop
        outputs = []
        state = state0
        for time_step in range(num_steps):
            output = tf.matmul(tf.transpose(w), state)
            state = tf.matmul(F, state) + params * (data[time_step] - output)
            outputs.append(output)

        # Sum of squared errors (SSE)
        pack_out = tf.reshape(tf.pack(outputs), data.get_shape())
        cost = tf.reduce_sum(tf.pow(pack_out - data, 2))
        return state0, params, varz, cost

    return model


SimpleETS = ModelETS([1], [[1]])
HoltsLinear = ModelETS([1, 1], [[1, 1], [0, 1]])
DampedTrend = ModelETS([1, 'phi'], [[1, 'phi'], [0, 'phi']])


# Meta-parameters and debugging knobs
learning_rate = 0.001
training_epochs = 2000
display_step = 50

# Test data
ys = np.asarray([1, 1, 1, 2, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10])
data = tf.placeholder('float', ys.shape)

# Gradient descent
state0, params, varz, cost = DampedTrend(data)
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)


def PrintDiagnostics(sess, ys, epoch_str):
    c = sess.run(cost, feed_dict={data: ys})
    print "Epoch:", epoch_str, \
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
        if (epoch + 1) % display_step == 0:
            PrintDiagnostics(sess, ys, '%04d' % (epoch+1))

    print "Optimization Finished!"
    PrintDiagnostics(sess, ys, 'Final')
