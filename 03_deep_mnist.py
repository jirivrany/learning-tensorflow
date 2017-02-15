"""
Based on
https://www.tensorflow.org/tutorials/mnist/pros/
"""

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

import tensorflow as tf
sess = tf.InteractiveSession()

# placeholders for data
# image 28 by 28 pixel flattened into row vector
# none means - any value, unkown on start
x = tf.placeholder(tf.float32, shape=[None, 784])
# classes 0-9
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# variables = model parameters
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

# run the InteractiveSession
init = tf.initialize_all_variables()
sess.run(init)

#model
y = tf.matmul(x,W) + b
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))

#train the model
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
for i in range(1000):  
    # 100 samples per iteration
    batch = mnist.train.next_batch(100)
    # replace empty placeholders with data
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})

# model validation
# tf.argmax gives the index of the highest entry in a tensor along some axis
# compared predicted and real value
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

# cast the list of booleans to numbers and reduce mean
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# print the result
print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

