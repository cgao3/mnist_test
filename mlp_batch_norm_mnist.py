import tensorflow as tf
import numpy as np

epsilon = 1e-4
x = tf.placeholder(shape=[None, 784], dtype=tf.float32)
y_ = tf.placeholder(shape=[None, 10], dtype=tf.float32)

random_norm_init=tf.random_normal_initializer(mean=0.0, stddev=1)
w1=tf.get_variable(name='w1_batch_norm', shape=[784,100], initializer=random_norm_init)

z1=tf.matmul(x,w1)

batch_mean1, batch_var1=tf.nn.moments(z1,[0])

gamma1=tf.get_variable(name='gamma1_batch_norm', shape=[100], initializer=random_norm_init)
beta1=tf.get_variable(name='beta1_batch_norm', shape=[100], initializer=random_norm_init)

b1=tf.nn.batch_normalization(z1, batch_mean1, batch_var1, beta1, gamma1, epsilon)
l1=tf.nn.relu(b1)

w2=tf.get_variable(name='w2_batch_norm', shape=[100,100], initializer=random_norm_init)
z2=tf.matmul(l1, w2)
batch_mean2, batch_var2=tf.nn.moments(z2, [0])

gamm2=tf.get_variable(name='gamma2_batch_norm', shape=[100], initializer=random_norm_init)
beta2=tf.get_variable(name='beta2_batch_norm', shape=[100], initializer=random_norm_init)

b2=tf.nn.batch_normalization(z2,batch_mean2, batch_var2, beta2, gamm2, epsilon)
l2=tf.nn.relu(b2)

w3=tf.get_variable(name='w3_batch_norm', shape=[100,10], initializer=random_norm_init)
b3=tf.get_variable(name='b3_batch_norm', shape=[10], initializer=random_norm_init)

z3=tf.matmul(l2, w3) + b3


loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=z3, labels=y_))

accuracy=tf.reduce_mean(
    tf.cast(
        tf.equal(tf.arg_max(y_, 1), tf.arg_max(z3, 1))
    , tf.float32)
)

learning_rate=tf.placeholder(dtype=tf.float32)
optimizer=tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)


with tf.Session() as sess:
    from tensorflow.examples.tutorials.mnist import input_data
    mnist=input_data.read_data_sets('MNIST_data', one_hot=True)
    lr=0.5
    sess.run(tf.global_variables_initializer())
    for step in range(10000):
        batch=mnist.train.next_batch(100)
        sess.run(optimizer, feed_dict={x:batch[0], y_:batch[1], learning_rate:lr})
        acc=sess.run(accuracy, feed_dict={x:batch[0], y_:batch[1]})
        if step%100==0:
            print("step ", step, " accuracy is", acc)
            lr = lr*0.99
    test_acc=sess.run(accuracy, feed_dict={x:mnist.test.images, y_:mnist.test.labels})
    print("Test accuracy: ", test_acc)