import tensorflow as tf

# Linear Regression
from sklearn import datasets, preprocessing
boston = datasets.load_boston()
x_data = preprocessing.StandardScaler().fit_transform(boston.data)
y_data = boston.target

x = tf.placeholder(tf.float64, shape=(None, 13))
y_true = tf.placeholder(tf.float64, shape=(None))

# from IPython import embed; embed()

with tf.name_scope('inference') as scope:
    w = tf.Variable(tf.zeros([1, 13], dtype=tf.float64), name='weights')
    b = tf.Variable(0, dtype=tf.float64, name='bias')
    y_pred = tf.matmul(w, tf.transpose(x)) + b

with tf.name_scope('loss') as scope:
    loss = tf.reduce_mean(tf.square(y_true-y_pred))

with tf.name_scope('train') as scope:
    learning_rate = 0.1
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for step in range(100):
        sess.run(train, feed_dict={x: x_data, y_true: y_data})

        if step % 10 == 0:
            mse = sess.run(loss, feed_dict={x: x_data, y_true: y_data})
            print(f'step : {step}, mse:{mse}')

    MSE = sess.run(loss, feed_dict={x: x_data, y_true: y_data})

print(MSE)