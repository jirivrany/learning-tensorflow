"""
Based on second part of
https://www.tensorflow.org/tutorials/mnist/pros/
"""

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf


def weight_variable(shape):
    """
    Tnitialize weights with a small amount of noise for 
    symmetry breaking, and to prevent 0 gradients. 
    """
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    """
     Since we're using ReLU neurons, it is also good 
     practice to initialize them with a slightly 
     positive initial bias to avoid "dead neurons".
    """
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    """
    Our convolutions uses a stride of one and are zero padded 
    so that the output is the same size as the input.
    """
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    """
    Max pooling over 2x2 blocks        
    """
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], 
        strides=[1, 2, 2, 1], padding='SAME')    

def main():
    """
    We can now implement our CNN
    """    
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

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

    # first convolution layer
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    #  reshape x to a 4d tensor, with the second and third dimensions 
    # corresponding to image width and height, and the final dimension
    # corresponding to the number of color channels.
    x_image = tf.reshape(x, [-1,28,28,1])
    # We then convolve x_image with the weight tensor, add the bias, 
    # apply the ReLU function, and finally max pool. 
    # The max_pool_2x2 method will reduce the image size to 14x14.
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # Second conv. layer
    # The second layer will have 64 features for each 5x5 patch.
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # Fully connected layer
    # with 1024 neurons to allow processing on the entire image.
    # igame has been reduced to 7x7 (initaly 28, then 14 after first and 7 after second layer)

    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # dropout to reduce overfitting
    # dropout works on fully connected layer
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # final read out Softmax layer
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    # now we can train and test the model

    # new in this code
    # We will replace the steepest gradient descent optimizer with the more sophisticated ADAM optimizer.
    # We will include the additional parameter keep_prob in feed_dict to control the dropout rate.
    # We will add logging to every 100th iteration in the training process.



    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # run the InteractiveSession
    init = tf.initialize_all_variables()
    sess.run(init)

    for i in range(20000):
        batch = mnist.train.next_batch(50)
        if i%100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x:batch[0], y_: batch[1], keep_prob: 1.0})
            print("step %d, training accuracy %g"%(i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    print("test accuracy %g"%accuracy.eval(feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

if __name__ == "__main__":
    main()    