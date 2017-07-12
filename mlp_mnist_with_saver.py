import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops

from tensorflow.examples.tutorials.mnist import input_data

mnist=input_data.read_data_sets('MNIST_data', one_hot=True)

'''
architecture description:

three FC layers,

layer 1:
input in the shape [None, 784] x [784, 100]

layer 2:
[None, 100] x [100, 100]

layer 3:
[None, 100] x [100, 10]

output [None, 10]

then apply softmax,
loss function is cross entropy, loss(p,q) = - \sum_{i} p_i log (q_i)

use GradientDescent to minimize this loss.

Learning rate is a small number 0<lr<1.0
learning rate should decay because of the convergence condition \sum_{lr_i} < infinity
'''
def selu(x):
    with ops.name_scope('elu') as scope:
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale*tf.where(x>=0.0, x, alpha*tf.nn.elu(x))


x=tf.placeholder(tf.float32, shape=[None, 784], name='x_inputs')
y_ =tf.placeholder(tf.float32, shape=[None, 10], name='y_labels')

tf.add_to_collection('x_inputs_node', x)
tf.add_to_collection('y_labels_node', y_)

w1=tf.get_variable(name='nobatch_normal_w1', initializer=tf.random_normal(shape=[784,100], mean=0, stddev=1), dtype=tf.float32)
b1=tf.get_variable(name='nobatch_normal_b1', initializer=tf.random_normal(shape=[100], mean=0, stddev=1), dtype=tf.float32)

z1=tf.matmul(x, w1) + b1
l1=selu(z1)
#l1=tf.nn.relu(l1)

w2=tf.get_variable(name='nobatch_normal_w2', initializer=tf.random_normal(shape=[100,100], mean=0, stddev=1), dtype=tf.float32)
b2=tf.get_variable(name='nobatch_normal_b2', initializer=tf.random_normal(shape=[100], mean=0, stddev=1), dtype=tf.float32)
z2=tf.matmul(l1, w2) + b2
l2=selu(z2)
#l2=tf.nn.relu(z2)

keep_prob=tf.placeholder(tf.float32)
h_dropout=tf.nn.dropout(l2, keep_prob=keep_prob)
w3=tf.get_variable(name='nobatch_normal_w3', initializer=tf.random_normal(shape=[100,10], mean=0, stddev=1), dtype=tf.float32)
b3=tf.get_variable(name='nobatch_normal_b3', initializer=tf.random_normal(shape=[10], mean=0, stddev=1), dtype=tf.float32)
z3=tf.matmul(l2, w3) + b3

loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=z3))
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(z3,1), tf.arg_max(y_,1)),tf.float32), name='accuracy_eva')

tf.add_to_collection('accuracy_op', accuracy)

optimizer=tf.train.AdamOptimizer().minimize(loss)
saver=tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    lr=0.5
    for step in range(10000):
        batch=mnist.train.next_batch(100)
        sess.run(optimizer,feed_dict={x:batch[0], y_:batch[1], keep_prob:0.5})
        acc=sess.run(accuracy, feed_dict={x:batch[0], y_:batch[1]})
        if step%100==0:
            print("step", step, "train accuracy is", acc)
            lr=lr*0.99

    test_acc=sess.run(accuracy, feed_dict={x:mnist.test.images, y_:mnist.test.labels, keep_prob:1.0})
    print("Test accuracy is", test_acc)
    print("saving model and graph")
    saver.save(sess, '/tmp/mlp_mnist_model/', global_step=step)
    tf.train.write_graph(sess.graph_def, '/tmp/mlp_mnist_graph/', 'mlp.txt', as_text=True)
    tf.train.write_graph(sess.graph_def, '/tmp/mlp_mnist_graph/', 'mlp.pb', as_text=False)
    print("Done.")

