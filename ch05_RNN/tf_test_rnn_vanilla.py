import tensorflow as tf
from tensorflow.examples.tutorials.mnist import  input_data

# Import mnist data
mnist = input_data.read_data_sets('./data/', one_hot=True)

# define params
element_size = 28
time_steps = 28
num_classes = 10
batch_size = 128
hidden_layer_size = 128

# where to save TensorBoard model summary
LOG_DIR = 'logs/RNN_with_summaries'

# create placeholder for inputs, labels
_input = tf.placeholder(tf.float32,
                        shape=[None, time_steps, element_size],
                        name='inputs')
y = tf.placeholder(tf.float32,
                   shape=[None, num_classes],
                   name='inputs')

# helper function taken from official tensorflow docs
# add some ops that take care of logging summaries
def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stdev'):
            stdev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stdev', stdev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


# weights and bias for input and hidden layer
with tf.name_scope('rnn_weights'):
    with tf.name_scope('W_x'):
        Wx = tf.Variable(tf.zeros([element_size, hidden_layer_size]))
        variable_summaries(Wx)
    with tf.name_scope('W_h'):
        Wh = tf.Variable(tf.zeros([hidden_layer_size, hidden_layer_size]))
        variable_summaries(Wh)
    with tf.name_scope('bias'):
        b_rnn = tf.Variable(tf.zeros([hidden_layer_size]))
        variable_summaries(b_rnn)


def rnn_steps(previous_hidden_state, x):

    current_hidden_state = tf.tanh(
        tf.matmul(previous_hidden_state, Wh) +
        tf.matmul(x, Wx) +
        b_rnn
    )

    return current_hidden_state


# processing inputs to work with scan function
# current input shape : (batch_size, time_step, element_size)
processed_input = tf.transpose(_input, perm=[1, 0, 2])
# current input shape : (time_step, batch_size, element_size)


initial_hidden = tf.zeros([batch_size, hidden_layer_size])
# getting all state vectors across time
all_hidden_states = tf.scan(rnn_steps,
                            processed_input,
                            initializer=initial_hidden,
                            name='states')


# weights for output layers
with tf.name_scope('linear_layer_weights') as scope:
    with tf.name_scope('W_linear'):
        Wl = tf.Variable(tf.truncated_normal([hidden_layer_size, num_classes],
                                             mean=0,
                                             stddev=0.01))
        variable_summaries(Wl)

    with tf.name_scope('Bias_linear'):
        bl = tf.Variable(tf.truncated_normal([num_classes],
                                             mean=0,
                                             stddev=0.01))
        variable_summaries(bl)

# apply linear layer to state vector
def get_linear_layer(hidden_state):
    return tf.matmul(hidden_state, Wl) + bl

with tf.name_scope('linear_layer_weights') as scope:
    # Iterate across time, apply linear layer to all RNN outputs
    all_outputs = tf.map_fn(get_linear_layer, all_hidden_states)
    # get last output -- h_28
    output = all_outputs[-1]
    tf.summary.histogram('outputs', output)


with tf.name_scope('cross_entropy'):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output,
                                                                              labels=y))
    tf.summary.scalar('cross_entropy', cross_entropy)


with tf.name_scope('train'):
    # Using RMSPropOptimizer
    train_step = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cross_entropy)


with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(output, 1))
    accuracy = (tf.reduce_mean(tf.cast(correct_prediction, tf.float32))) * 100
    tf.summary.scalar('accuracy', accuracy)

# merge all the summaries
merged = tf.summary.merge_all()


# get a small test set
test_data = mnist.test.images[:batch_size].reshape((-1, time_steps, element_size))
test_label = mnist.test.labels[:batch_size]


with tf.Session() as sess:
    # write summaries to LOG_DIR -- used by TensorBoard
    train_writer = tf.summary.FileWriter(LOG_DIR + '/train',
                                         graph=tf.get_default_graph())
    test_writer = tf.summary.FileWriter(LOG_DIR + '/test',
                                        graph=tf.get_default_graph())

    sess.run(tf.global_variables_initializer())

    for i in range(10000):

        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # reshape data to get 28 sequences of 18 pixels
        batch_x = batch_x.reshape((batch_size, time_steps, element_size))
        summary, _ = sess.run([merged, train_step],
                              feed_dict={_input: batch_x, y: batch_y})

        # add to summaries
        train_writer.add_summary(summary, i)

        if i % 1000 == 0:
            acc, loss, = sess.run([accuracy, cross_entropy],
                                    feed_dict={_input: batch_x, y: batch_y})
            print(f'Iter {str(i)}, Minibatch Loss={loss}, training accuracy={acc}')

        if i % 100 == 0:
            # calculate accuracy for 128 mnist test images and add to summaries
            summary, acc = sess.run([merged, accuracy],
                                    feed_dict={_input: test_data,
                                               y: test_label})
            test_writer.add_summary(summary, i)

    test_acc = sess.run(accuracy, feed_dict={_input: test_data, y: test_label})
    print(f'Test accuracy: {test_acc}')
