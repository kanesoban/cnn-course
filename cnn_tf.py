import numpy
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)


class DenseLayer:
    def __init__(self, input_var, output_size, nonlinearity=None):
        self.input_var = input_var
        self.weights = tf.Variable(numpy.zeros((input_var.shape[1], output_size), dtype=numpy.float32))
        self.bias = tf.Variable(numpy.zeros((output_size,), dtype=numpy.float32))
        self.nonlinearity = nonlinearity
        self.op = self.nonlinearity(tf.matmul(self.input_var, self.weights) + self.bias)
        if self.nonlinearity:
            self.op = self.nonlinearity(self.op)
        self.shape = self.op.shape


class ConvPoolLayer:
    def __init__(self, input_var, id, filters_in, filters_out, conv_size=5):
        with tf.name_scope('conv' + str(id)) as scope:
            kernel = tf.Variable(tf.truncated_normal([conv_size, conv_size, filters_in, filters_out], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(input_var, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[filters_out], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.op = tf.nn.relu(out, name=scope)

        self.op = tf.nn.max_pool(self.op, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool' + str(id))


class LeNet5:
    def __init__(self, input_size=(32, 32, 3), classes=2, learning_rate=1e-4):
        self.classes = classes
        self.learning_rate = learning_rate
        self.session = tf.Session()
        width, height, color_channels = input_size
        self.layers = []
        self.input_var = tf.placeholder(dtype=tf.float32, shape=(None, width, height, color_channels),
                                        name='input_var')
        self.target_var = tf.placeholder(dtype=tf.float32, shape=(None, self.classes), name='labels_var')

        self.layers.append(ConvPoolLayer(self.input_var, 0, filters_in=1, filters_out=6))
        self.layers.append(ConvPoolLayer(self.layers[-1].op, 1, filters_in=6, filters_out=16))
        flattened_input = tf.layers.flatten(self.layers[-1].op)
        self.layers.append(DenseLayer(flattened_input, 120, nonlinearity=tf.nn.relu))
        self.layers.append(DenseLayer(self.layers[-1].op, 84, nonlinearity=tf.nn.relu))
        self.layers.append(DenseLayer(self.layers[-1].op, self.classes, nonlinearity=tf.nn.softmax))

        self.forward_op = self.layers[-1].op

        self.cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=self.forward_op,
                labels=self.target_var
            )
        )
        self.train_op = tf.train.RMSPropOptimizer(self.learning_rate, decay=0.9999, momentum=0.99).minimize(self.cost)

        correct_preds = tf.equal(tf.argmax(self.forward_op, 1), tf.argmax(self.target_var, 1))
        self.accuracy_op = tf.reduce_mean(tf.cast(correct_preds, tf.float32), axis=0)

        self.session.run(tf.global_variables_initializer())

    def fit(self, data, targets, batch_size=None):
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

    def predict(self, data):
        return numpy.argmax(self.session.run(self.layers[-1].op, feed_dict={self.input_var: data}), axis=1)

    def accuracy(self, data, targets):
        return self.session.run(self.accuracy_op, feed_dict={self.input_var: data, self.target_var: targets})

    def loss(self, data, targets):
        return self.session.run(self.cost, feed_dict={self.input_var: data, self.target_var: targets})
