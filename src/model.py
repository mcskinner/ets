import numpy as np
import tensorflow as tf
rng = np.random

# Meta-parameters and debugging knobs
learning_rate = 0.01
training_epochs = 2000
display_step = 50

# Test data
y = np.asarray([1, 1, 1, 2, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10])
num_steps = y.shape[0]

# Input data placeholders
data_in = tf.placeholder('float')
data_out = tf.placeholder('float')

# ETS params
level0 = tf.Variable(0.1 * rng.randn(), name = 'level0', dtype = tf.float32)
pace0 = tf.Variable(0.1 * rng.randn(), name = 'pace0', dtype = tf.float32)
alpha = tf.Variable(0.5, name = 'alpha', dtype = tf.float32)
beta = tf.Variable(0.1, name = 'beta', dtype = tf.float32)

# Definition of the ETS update
def update(y, level, pace):
    output = level + pace
    new_level = output + alpha * (y - output)
    new_pace = pace + beta * (y - output)
    return output, new_level, new_pace

# Unrolled ETS loop
outputs = []
level, pace = level0, pace0
for time_step in range(num_steps):
    output, level, pace = update(data_in[time_step], level, pace)
    outputs.append(output)

# Mean squared error
cost = tf.reduce_sum(tf.pow(tf.pack(outputs) - data_out, 2))

# Gradient descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Fit the data.    
    for epoch in range(training_epochs):
        sess.run(optimizer, feed_dict={data_in: y, data_out: y})

        # Display logs per epoch step
        if (epoch + 1) % display_step == 0:
            c = sess.run(cost, feed_dict={data_in: y, data_out: y})
            print "Epoch:", '%04d' % (epoch+1), \
                "cost=", "{:.9f}".format(c), \
                "level0=", sess.run(level0), \
                "pace0=", sess.run(pace0), \
                "alpha=", sess.run(alpha), \
                "beta=", sess.run(beta)

    print "Optimization Finished!"
    training_cost = sess.run(cost, feed_dict={data_in: y, data_out: y})
    print "Training cost=", training_cost, \
        "level0=", sess.run(level0), \
        "pace0=", sess.run(pace0), \
        "alpha=", sess.run(alpha), \
        "beta=", sess.run(beta)
