import numpy as np
import tensorflow as tf

# Build out the TensorFlow model for Holt's linear method.
def ModelETS(w_in, F_in):
    n = len(w_in)

    def model(data):
        num_steps = data.get_shape()[0]

        # ETS params
        state0 = tf.Variable(tf.random_uniform([n, 1], -0.1, 0.1), dtype = tf.float32)
        params = tf.Variable(tf.random_uniform([n, 1], 0.1, 0.2), dtype = tf.float32)
        w = tf.constant(w_in, shape = [n, 1], dtype = tf.float32)
        F = tf.constant(F_in, shape = [n, n], dtype = tf.float32)

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
        return state0, params, cost

    return model


SimpleETS = ModelETS([1], [[1]])
HoltsLinear = ModelETS([1, 1], [[1, 1], [0, 1]])

# Meta-parameters and debugging knobs
learning_rate = 0.01
training_epochs = 2000
display_step = 50

# Test data
ys = np.asarray([1, 1, 1, 2, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10])
data = tf.placeholder('float', ys.shape)

# Gradient descent
state0, params, cost = HoltsLinear(data)
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

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
            c = sess.run(cost, feed_dict={data: ys})
            print "Epoch:", '%04d' % (epoch+1), \
                "cost=", "{:.9f}".format(c), \
                "state0=", sess.run(tf.reshape(state0, [2])), \
                "params=", sess.run(tf.reshape(params, [2]))

    print "Optimization Finished!"
    training_cost = sess.run(cost, feed_dict={data: ys})
    print "Training cost=", training_cost, \
        "state0=", sess.run(tf.reshape(state0, [2])), \
        "params=", sess.run(tf.reshape(params, [2]))
