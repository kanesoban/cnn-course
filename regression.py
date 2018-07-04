import tensorflow as tf
import numpy


class Regression:
    def __init__(self, input_size, classes=2, learning_rate=1e-3):
        self.learning_rate = learning_rate
        self.session = tf.Session()
        self.classes = classes
        self.input_var = tf.placeholder(dtype=tf.float32, shape=(None, input_size), name='input_var')
        self.target_var = tf.placeholder(dtype=tf.float32, shape=(None, self.classes), name='labels_var')
        self.weights = tf.Variable(numpy.zeros((input_size, classes), dtype=numpy.float32))
        self.bias = tf.Variable(numpy.zeros((classes,), dtype=numpy.float32))
        self.forward_op = tf.nn.softmax(tf.matmul(self.input_var, self.weights) + self.bias)
        self.cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=self.forward_op,
                labels=self.target_var
            )
        )
        self.train_op = tf.train.RMSPropOptimizer(self.learning_rate, decay=0.9999, momentum=0.99).minimize(self.cost)
        self.session.run(tf.global_variables_initializer())

        correct_preds = tf.equal(tf.argmax(self.forward_op, 1), tf.argmax(self.target_var, 1))
        self.accuracy_op = tf.reduce_mean(tf.cast(correct_preds, tf.float32), axis=0)

    def fit(self, data, targets, batch_size=None):
        data = data.reshape((data.shape[0], -1))
        if batch_size:
            n_batches = data.shape[0] // batch_size + 1
            for i in range(n_batches):
                lower_bound = i * batch_size
                upper_bound = min((i+1) * batch_size, data.shape[0])
                if lower_bound < upper_bound:
                    self.session.run(self.train_op, feed_dict={self.input_var: data[lower_bound:upper_bound],
                                                               self.target_var: targets[lower_bound:upper_bound]})
        else:
            self.session.run(self.train_op, feed_dict={self.input_var: data, self.target_var: targets})

    def accuracy(self, data, targets):
        data = data.reshape((data.shape[0], -1))
        return self.session.run(self.accuracy_op, feed_dict={self.input_var: data, self.target_var: targets})

    def loss(self, data, targets):
        data = data.reshape((data.shape[0], -1))
        return self.session.run(self.cost, feed_dict={self.input_var: data, self.target_var: targets})
