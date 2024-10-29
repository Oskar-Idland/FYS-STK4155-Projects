import math
import autograd.numpy as np
import sys
import warnings
from autograd import grad, elementwise_grad
from random import random, seed
from copy import deepcopy, copy
from typing import Tuple, Callable
from sklearn.utils import resample

from activation_funcs import *
from cost_funcs import *
from Schedulers import *

warnings.simplefilter("error")

class FFNN:
	"""
	Description:
	------------
		Feed Forward Neural Network with interface enabling flexible design of a
		neural networks architecture and the specification of activation function
		in the hidden layers and output layer respectively. This model can be used
		for both regression and classification problems, depending on the output function.

	Attributes:
	------------
		dimensions (tuple[int]): A list of positive integers, which specifies the
			number of nodes in each of the networks layers. The first integer in the array
			defines the number of nodes in the input layer, the second integer defines number
			of nodes in the first hidden layer and so on until the last number, which
			specifies the number of nodes in the output layer.

		hidden_func (Callable): The activation function for the hidden layers

		output_func (Callable): The activation function for the output layer

		cost_func (Callable): Our cost function

		seed (int): Sets random seed, makes results reproducible
	"""
	def __init__(
		self,
		dimensions: tuple[int],
		hidden_func: Callable = sigmoid,
		output_func: Callable = lambda x: x,
		cost_func: Callable = CostOLS,
		seed: int = None,
	):
		self.dimensions = dimensions
		self.hidden_func = hidden_func
		self.output_func = output_func
		self.cost_func = cost_func
		self.seed = seed
		self.weights = list()
		self.schedulers_weight = list()
		self.schedulers_bias = list()
		self.a_matrices = list()
		self.z_matrices = list()
		self.classification = None

		self.reset_weights()
		self._set_classification()

	def fit(
		self,
		X: np.ndarray,
		t: np.ndarray,
		scheduler: Scheduler,
		batches: int = 1,
		epochs: int = 100,
		lambd: float = 0,
		X_val: np.ndarray = None,
		t_val: np.ndarray = None,
	):
		# Setup
		if self.seed is not None:
			np.random.seed(self.seed) 
			 
		val_set = False
		if X_val is not None and t_val is not None:
			val_set = True
		
		# Creating arrays for score metrics
		train_errors = np.empty(epochs)
		train_errors.fill(np.nan)
		val_errors = np.empty(epochs)
		val_errors.fill(np.nan)
	
		train_accs = np.empty(epochs)
		train_accs.fill(np.nan)
		val_accs = np.empty(epochs)
		val_accs.fill(np.nan)

		self.schedulers_weight = list()
		self.schedulers_bias = list()

		batch_size = X.shape[0] // batches

		X, t = resample(X, t)
			 
		# This function returns a function valued only at x
		cost_function_train = self.cost_func(t)
		if val_set:
			cost_function_val = self.cost_func(t_val)

		# Create schedulers for each weight matrix
		for i in range(len(self.weights)):
			self.schedulers_weight.append(copy(scheduler))
			self.schedulers_bias.append(copy(scheduler))

		print(f"{scheduler.__class__.__name__}: Eta={scheduler.eta}, lambda={lambd}")
	
		try:
			for e in range(epochs):
				for i in range(batches):
					if i == batches - 1:
						# If the for loop has reached the last batch, take the remainding data
						X_batch = X[i * batch_size :, :]
						t_batch = t[i * batch_size :, :]	

					else:
						X_batch = X[i * batch_size : (i + 1) * batch_size, :]
						t_batch = t[i * batch_size : (i + 1) * batch_size, :]

					self._feedforward(X_batch)
					self._backpropagation(X_batch, t_batch, lambd)

				# Reset scheduler for each epoch
				for scheduler in self.schedulers_weight:
					scheduler.reset()

				for scheduler in self.schedulers_bias:
					scheduler.reset()
				
				# Computing performance metrics
				pred_train = self.predict(X)
				train_error = cost_function_train(pred_train)

				train_errors[e] = train_error
				if val_set:
					pred_val = self.predict(X_val)
					val_error = cost_function_val(pred_val)
					val_errors[e] = val_error

				if self.classification:
					train_acc = self._accuracy(self.predict(X), t)
					train_accs[e] = train_acc
					if val_set:
						val_acc = self._accuracy(pred_val, t_val)
						val_accs[e] = val_acc

				# Printing progress bar
				progression = e / epochs
				print_length = self._progress_bar(
					progression,
					train_error = train_errors[e],
					train_acc = train_accs[e],
					val_error = val_errors[e],
					val_acc = val_accs[e],
				)
		except KeyboardInterrupt:
			# Allow the user to interrupt the training and see the results
			pass
	
		# Visualize the training process
		sys.stdout.write("\r" + " " * print_length)
		sys.stdout.flush()
		self._progress_bar(
			1,
			train_error=train_errors[e],
			train_acc=train_accs[e],
			val_error=val_errors[e],
			val_acc=val_accs[e],
		)
		sys.stdout.write("")

		# return performance metrics for the entire run
		scores = dict()

		scores["train_errors"] = train_errors

		if val_set:
			scores["val_errors"] = val_errors

		if self.classification:
			scores["train_accs"] = train_accs

			if val_set:
				scores["val_accs"] = val_accs

		return scores
	
	def predict(self, X: np.ndarray, *, threshold = 0.5):

		predict = self._feedforward(X)

		if self.classification:
			return np.where(predict > threshold, 1, 0)
		else:
			return predict
		
	def reset_weights(self):
		"""
		Description:
		------------
			Resets the weights of the network in order to train the network from scratch.
		"""

		if self.seed is not None:
			np.random.seed(self.seed)

		self.weights = list()
		for i in range(len(self.dimensions) - 1):
			weight_array = np.random.randn(
				self.dimensions[i] + 1, self.dimensions[i + 1]
			)
			weight_array[0, :] = np.random.randn(self.dimensions[i + 1]) * 0.01

			self.weights.append(weight_array)

	def _feedforward(self, X: np.ndarray):

		# Resetting the matrices
		self.a_matrices = list()
		self.z_matrices = list()

		# Correcting the shape of the input
		if len(X.shape) == 1:
			X = np.reshape(X, (1, X.shape[0]))

		# Adding bias to the Design Matrix
		bias = np.ones((X.shape[0], 1)) * 0.01
		X = np.hstack((bias, X))

		# a^0, the nodes in the input layer (one a^0 for each row in X - where the
		# exponent indicates layer number).
		a = X
		self.a_matrices.append(a)
		self.z_matrices.append(a)

		# Do the feedforward
		for i in range(len(self.weights)):
			if i < len(self.weights) -1:
				z = a @ self.weights[i]
				self.z_matrices.append(z)
				a = self.hidden_func(z)

				bias = np.ones((a.shape[0], 1)) * 0.01
				a = np.hstack((bias, a))
				self.a_matrices.append(a)
			else:
				try:
					# a^L, the nodes in the output layer
					z = a @ self.weights[i]
					a = self.output_func(z)
					self.a_matrices.append(a)
					self.z_matrices.append(z)
				except Exception as OverflowError:
					print(
						"OverflowError in fit() in FFNN\nHOW TO DEBUG ERROR: Consider lowering your learning rate or scheduler"
					)
		# Return a^L
		return a

	def _backpropagation(self, X, t, lambd):

		out_derivative = derivate(self.output_func)
		hidden_derivative = derivate(self.hidden_func)

		for i in range(len(self.weights) -1, -1, -1):
			# Delta^L
			if i == len(self.weights) - 1:
				# multi class classification
				if (
					self.output_func == "softmax"
				):
					delta_matrix = self.a_matrices[i + 1] - t
				else:
				# single class classification
					cost_func_derivative = grad(self.cost_func(t))
					delta_matrix = out_derivative(
						self.z_matrices[i + 1]
						) * cost_func_derivative(self.a_matrices[i + 1])
			# delta matrix for hidden layers
			else:
				delta_matrix = (
					self.weights[i + 1][1:, :] @ delta_matrix.T
				).T * hidden_derivative(self.z_matrices[i + 1])

			# Computing the gradients
			gradient_weights = self.a_matrices[i][:,1:].T @ delta_matrix
			gradient_bias = np.sum(delta_matrix, axis=0).reshape(
				1, delta_matrix.shape[1]
			)

			# Regularization
			gradient_weights += lambd * self.weights[i][1:, :]

			# use scheduler to update weights		'
			update_matrix = np.vstack(
				[
					self.schedulers_bias[i].update_change(gradient_bias),
					self.schedulers_weight[i].update_change(gradient_weights),
				]
			)

			# update weights and bias
			self.weights[i] -= update_matrix

	def _accuracy(self, prediction: np.ndarray, target: np.ndarray):

		assert prediction.size == target.size
		return np.average((target == prediction))
	
	def _set_classification(self):

		self.classification = False
		if (
			self.cost_func.__name__ == "CostLogReg"
			or self.cost_func.__name__ == "CostCrossEntropy"
		):
			self.classification = True

	def _progress_bar(self, progression, **kwargs):
		"""
		Description:
		------------
			Displays progress of training 
		"""
		print_length = 40
		num_equals = int(progression * print_length)
		num_not = print_length - num_equals
		arrow = ">" if num_equals > 0 else ""
		bar = "[" + "=" * (num_equals - 1) + arrow + "-" * num_not + "]"
		perc_print = self._format(progression * 100, decimals=5)
		line = f"  {bar} {perc_print}% "

		for key in kwargs:
			if not np.isnan(kwargs[key]):
				value = self._format(kwargs[key], decimals=4)
				line += f"| {key}: {value} "
		sys.stdout.write("\r" + line)
		sys.stdout.flush()
		
		return len(line)
	
	def _format(self, value, decimals=4):
		"""
		Description:
		------------
			Formats decimal numbers for progress bar
		"""
		if value > 0:
			v = value
		elif value < 0:
			v = -10 * value
		else:
			v = 1
		n = 1 + math.floor(math.log10(v))
		if n >= decimals - 1:
			return str(round(value))
		
		return f"{value:.{decimals-n-1}f}"	