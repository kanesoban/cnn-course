import numpy
import tensorflow as tf


class DenseLayer:
	def __init__(self, input_var, output_size, nonlinearity=None):
		self.input_var = input_var
		self.weights = tf.get_variable('weights', shape=(input_size, output_size), initializer=tf.zeros_initializer(), dtype=tf.float32)
 		self.bias = tf.get_variable('bias', shape=(None,), initializer=tf.zeros_initializer(), dtype=tf.float32)
 		self.nonlinearity = nonlinearity

 	def forward_op():
 		op = self.nonlinearity(tf.nn.bias_add(tf.matmul(self.weights, self.input_var), self.bias))
 		if self.nonlinearity:
 			op = self.nonlinearity(op)
 		return op


class ConvPoolLayer:
	def __init__(self, input_var, conv_size, filters_out, pool_size):
		self.width = input_var.shape[1]
		self.heigth = input_var.shape[2]
		self.filters_in = input_var.shape[3]
		self.filters_out = filters_out
		self.input_var = input_var
		self.weights = tf.get_variable('weights', shape=(self.filters_in, conv_size, conv_size, self.filters_out), initializer=tf.zeros_initializer(), dtype=tf.float32)
 		self.bias = tf.get_variable('bias', shape=(None,), initializer=tf.zeros_initializer(), dtype=tf.float32)
 		self.nonlinearity = tf.tanh

	def forward_op():
        op = tf.nn.conv2d(self.input_var, self.weigths, strides=[1, 1, 1, 1], padding='VALID')
        op = tf.nn.bias_add(op, self.bias)
        op = tf.nn.max_pool(
        	op,
        	ksize=[1, self.pool_size, self.pool_size, 1],
        	strides=[1, self.pool_size, self.pool_size, 1],
        	padding='VALID'
        )
        return tf.tanh(op)


class LeNet5:
	def __init__(self, classes, color_channels, learning_rate):
		self.classes = classes
		self.learning_rate = learning_rate
		self.session = tf.Session()
		with tf.variable_scope('LeNet5', reuse=tf.AUTO_REUSE):
			self.layers = []
			self.input_var = tf.placeholder(dtype=tf.float32, shape=(None, 32, 32, color_channels), name='input_var')
			self.labels_var = tf.placeholder(dtype=float32, shape=(None, self.classes), name='labels_var')
			self.layers.append(ConvPoolLayer(self.input_var, conv_size=5, filters_out=6, pool_size=2))
			self.layers.append(ConvPoolLayer(layers[-1], conv_size=5, filters_out=16, pool_size=2))
			flattened_input = tf.layers.flatten(self.layers[-1].forward_op())
			self.layers.append(DenseLayer(flattened_input, 120))
			self.layers.append(DenseLayer(layers[-1], 84))
			self.layers.append(DenseLayer(layers[-1], 2))
		self.session.run(tf.global_variables_initializer())

		self.forward_op = self.layers[-1]

		self.cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=self.forward_op,
                labels=self.labels_var
            )
        )
		self.train_op = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.cost)

	def fit(self, data, targets):
		return self.session.run(self.train_op, feed_dict={self.input_var: data, self.target_var: targets})

	def predict(self, data):
		return numpy.argmax(self.session.run(self.layers[-1], feed_dict={self.input_var: data}))

	def loss(self, data, targets):
		return self.session(self.cost, feed_dict={self.input_var: data, self.target_var: targets})


