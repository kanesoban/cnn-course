import numpy
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)


class DenseLayer:
    def __init__(self, input_var, output_size, nonlinearity=None):
        self.input_var = input_var
        self.weights = tf.Variable(numpy.zeros((input_var.shape[1], output_size), dtype=numpy.float32))
        self.bias = tf.Variable(numpy.zeros((1,), dtype=numpy.float32))
        self.nonlinearity = nonlinearity
        self.op = self.nonlinearity(tf.matmul(self.input_var, self.weights) + self.bias)
        if self.nonlinearity:
            self.op = self.nonlinearity(self.op)
        self.shape = self.op.shape


class ConvPoolLayer:
    def __init__(self, input_var, conv_size, filters_out, pool_size, nonlinearity=None):
        self.width = input_var.shape[1]
        self.heigth = input_var.shape[2]
        self.filters_in = input_var.shape[3]
        self.filters_out = filters_out
        self.input_var = input_var
        self.pool_size = pool_size
        self.weights = tf.Variable(numpy.zeros((conv_size, conv_size, self.filters_in, self.filters_out),
                                               dtype=numpy.float32))
        self.bias = tf.Variable(numpy.zeros((1,), dtype=numpy.float32))
        self.nonlinearity = nonlinearity

        self.op = tf.nn.conv2d(self.input_var, self.weights, strides=[1, 1, 1, 1], padding='VALID')
        self.op = self.op + self.bias
        self.op = tf.nn.max_pool(
            self.op,
            ksize=[1, self.pool_size, self.pool_size, 1],
            strides=[1, self.pool_size, self.pool_size, 1],
            padding='VALID'
        )
        if self.nonlinearity:
            self.op = self.nonlinearity(self.op)
        self.shape = self.op.shape


class LeNet5:
    def __init__(self, input_size=(32, 32, 3), classes=2, learning_rate=1e-3):
        self.classes = classes
        self.learning_rate = learning_rate
        self.session = tf.Session()
        width, height, color_channels = input_size
        with tf.variable_scope('LeNet5', reuse=tf.AUTO_REUSE):
            self.layers = []
            self.input_var = tf.placeholder(dtype=tf.float32, shape=(None, width, height, color_channels),
                                            name='input_var')
            self.target_var = tf.placeholder(dtype=tf.float32, shape=(None, self.classes), name='labels_var')
            self.layers.append(ConvPoolLayer(self.input_var, conv_size=5, filters_out=6, pool_size=2,
                                             nonlinearity=tf.nn.tanh))
            self.layers.append(ConvPoolLayer(self.layers[-1].op, conv_size=5, filters_out=16, pool_size=2,
                                             nonlinearity=tf.nn.tanh))
            flattened_input = tf.layers.flatten(self.layers[-1].op)
            self.layers.append(DenseLayer(flattened_input, 120, nonlinearity=tf.nn.relu).op)
            self.layers.append(DenseLayer(self.layers[-1], 84, nonlinearity=tf.nn.relu).op)
            self.layers.append(DenseLayer(self.layers[-1], 2, nonlinearity=tf.nn.tanh).op)

        self.forward_op = self.layers[-1]

        self.cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=self.forward_op,
                labels=self.target_var
            )
        )
        self.train_op = tf.train.RMSPropOptimizer(self.learning_rate, decay=0.9999).minimize(self.cost)
        self.session.run(tf.global_variables_initializer())

    def fit(self, data, targets):
        return self.session.run(self.train_op, feed_dict={self.input_var: data, self.target_var: targets})

    def predict(self, data):
        return numpy.argmax(self.session.run(self.layers[-1], feed_dict={self.input_var: data}))

    def loss(self, data, targets):
        return self.session.run(self.cost, feed_dict={self.input_var: data, self.target_var: targets})
