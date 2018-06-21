import numpy
import tensorflow as tf


class DenseLayer:
	def __init__(self, input_size, output_size, nonlinearity=None):
		self.input_var = tf.placeholder(dtype=tf.float32, shape=(None, input_size), name='input_var')
		self.weights = tf.get_variable('weights', shape=(input_size, output_size), initializer=tf.zeros_initializer(), dtype=tf.float32)
 		self.bias = tf.get_variable('bias', shape=(None,), initializer=tf.zeros_initializer(), dtype=tf.float32)
 		self.nonlinearity = nonlinearity

 	def forward_op():
 		op = self.nonlinearity(tf.nn.bias_add(tf.matmul(self.weights, self.input_var), self.bias))
 		if self.nonlinearity:
 			op = self.nonlinearity(op)
 		return op


class ConvLayer:
	def __init__(self, heigth, width, conv_size, filters_in, filters_out):
		self.heigth = heigth
		self.width = width
		self.filters_in = filters_in
		self.filters_out = filters_out
		self.input_var = tf.placeholder(dtype=tf.float32, shape=(None, self.heigth, self.width, self.filters_in), name='input_var')
		self.weights = tf.get_variable('weights', shape=(self.filters_in, conv_size, conv_size, self.filters_out), initializer=tf.zeros_initializer(), dtype=tf.float32)
 		self.bias = tf.get_variable('bias', shape=(None,), initializer=tf.zeros_initializer(), dtype=tf.float32)
 		self.nonlinearity = tf.tanh

	def forward_op():
        op = tf.nn.conv2d(self.input_var, self.weigths, strides=[1, 1, 1, 1], padding='VALID')
        op = tf.nn.bias_add(op, self.bias)
        return tf.tanh(op)

class CNN:
	def __init__(self):
		pass
