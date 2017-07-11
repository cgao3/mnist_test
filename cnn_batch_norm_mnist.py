import tensorflow as tf
import numpy as np

'''
an convolutional neural net architecture + batch normalization

sepcification:

input [None, 784] => reshape into [None, 28, 28, 1]

first convolution layer

3x3, 32 convolutions



'''

epsilon = 1e-3


x = tf.placeholder(shape=[None, 784], dtype=tf.float32)
y_ = tf.placeholder(shape=[None, 10], dtype=tf.float32)

random_norm_init = tf.random_normal_initializer()

keep_probility=tf.placeholder(tf.float32)

def batch_norm_wrapper(inputs, var_name_prefix, is_training=True):
    pop_mean = tf.get_variable(name=var_name_prefix+'_pop_mean', shape=[inputs.get_shape()[-1]], dtype=tf.float32, trainable=False)
    pop_var = tf.get_variable(name=var_name_prefix+'_pop_var', shape=[inputs.get_shape()[-1]], dtype=tf.float32, trainable=False)

    gamma = tf.get_variable(name=var_name_prefix+'_gamma_batch_norm', shape=[inputs.get_shape()[-1]], initializer=random_norm_init)
    beta = tf.get_variable(name=var_name_prefix+'_beta1_batch_norm', shape=[inputs.get_shape()[-1]], initializer=random_norm_init)

    if is_training:
        batch_mean, batch_var = tf.nn.moments(inputs, [0,1,2])
        train_mean=tf.assign(pop_mean, pop_mean*0.999+batch_mean*(1-0.999))
        train_var=tf.assign(pop_var, pop_var*0.999+batch_var*(1-0.999))
        with tf.control_dependencies([train_mean,train_var]):
            b1 = tf.nn.batch_normalization(inputs, batch_mean, batch_var, beta, gamma, epsilon)
    else:
        b1=tf.nn.batch_normalization(inputs, pop_mean, pop_var, beta, gamma, epsilon)

    return b1

def build_graph(is_training=True):

    x_=tf.reshape(x, shape=[-1,28,28,1], name="input_images")

    w1=tf.get_variable(name='w1', shape=[5,5,1,32],initializer=random_norm_init)
    z1=tf.nn.conv2d(x_,w1,strides=(1,1,1,1), padding="VALID")
    pool1=tf.nn.max_pool(z1,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
    print(pool1)
    z_hat1=batch_norm_wrapper(pool1,var_name_prefix="1", is_training=is_training)
    l1=tf.nn.relu(z_hat1)

    w2=tf.get_variable(name='w2', shape=[5,5,32,32],initializer=random_norm_init)
    z2=tf.nn.conv2d(l1,w2,strides=(1,1,1,1), padding="VALID")
    z_hat2=batch_norm_wrapper(z2,var_name_prefix="2", is_training=is_training)
    l2=tf.nn.relu(z_hat2)

    dropout_l2=tf.nn.dropout(l2, keep_prob=keep_probility)
    w3=tf.get_variable(name='w3', shape=[8*8*32, 10], initializer=random_norm_init)
    b3=tf.get_variable(name='b3', shape=[10], initializer=random_norm_init)
    z3=tf.matmul(tf.reshape(dropout_l2, shape=[-1, 8*8*32]), w3) + b3

    return z3

z=build_graph()
print(z)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=z, labels=y_))
accuracy=tf.reduce_mean(
    tf.cast(
        tf.equal(tf.arg_max(y_, 1), tf.arg_max(z, 1))
    , tf.float32)
)

optimizer=tf.train.AdamOptimizer().minimize(loss)
with tf.Session() as sess:
    from tensorflow.examples.tutorials.mnist import input_data
    mnist=input_data.read_data_sets('MNIST_data', one_hot=True)
    sess.run(tf.global_variables_initializer())
    for step in range(50000):
        batch=mnist.train.next_batch(100)
        sess.run(optimizer, feed_dict={x:batch[0], y_:batch[1], keep_probility:0.5})
        acc=sess.run(accuracy, feed_dict={x:batch[0], y_:batch[1], keep_probility:1.0})
        if step%100==0:
            print("step ", step, " accuracy is", acc)

    print("traning finished.")
    tf.get_variable_scope().reuse_variables()
    z_test=build_graph(is_training=False)
    acc_test = tf.reduce_mean(
        tf.cast(
            tf.equal(tf.arg_max(y_, 1), tf.arg_max(z_test, 1))
            , tf.float32)
    )
    acc_test_value=sess.run(acc_test, feed_dict={x:mnist.test.images, y_:mnist.test.labels, keep_probility:1.0})
    print("Using population mean/variance")
    print("Test accuracy: ", acc_test_value)
