import tensorflow as tf

'''
evaluate a trained neural net model from checkpoint

rebuild the computation graph from saved .meta file
'''
def evaluate(meta_graph_file, checkpoint_file, op_name_input,
             op_name_label, op_name_to_run, feed_input, feed_label):

    sess=tf.Session()
    print("import meta graph, restore from checkpoint")
    saver=tf.train.import_meta_graph(meta_graph_file)
    saver.restore(sess, checkpoint_file)
    print("restored.")


    op_input=tf.get_collection(op_name_input)[0]
    op_label=tf.get_collection(op_name_label)[0]

    op_run=tf.get_collection(op_name_to_run)[0]

    result=sess.run(op_run, feed_dict={op_input:feed_input, op_label:feed_label})
    print("Result is ", result)

def evaluate_mnist_mlp(meta_graph, checkpoint):
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    input_node_name="x_inputs_node"
    label_node_name="y_labels_node"
    run_node_name="accuracy_op"
    evaluate(meta_graph, checkpoint, input_node_name, label_node_name, run_node_name,
             mnist.test.images, mnist.test.labels)


if __name__ == "__main__":
    import argparse
    parser= argparse.ArgumentParser()
    parser.add_argument('--meta_graph', type=str, default='', help='path to graph')
    parser.add_argument('--checkpoint', type=str, default='', help='path to checkpoint')
    parser.add_argument('--op_name_run', type=str, default='accuracy_op', help='op name')
    args=parser.parse_args()

    import os.path
    if not os.path.isfile(args.meta_graph):
        print("use --help to usage")
        exit(0)
    evaluate_mnist_mlp(args.meta_graph, args.checkpoint)

