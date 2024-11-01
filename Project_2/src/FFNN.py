import autograd.numpy as np
from typing import Callable
import math
import sys
from copy import copy

from activation_funcs import *
from cost_funcs import *
from Schedulers import *
from functions import MSE, MSE_derivative, R2, create_X, FrankeFunction

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

# warnings.simplefilter("error")

class FFNN:
	def __init__(
		self,
		layer_sizes: tuple[int],
		hidden_funcs: tuple[Callable] = None,
		output_func: Callable = identity,
		cost_func: Callable = MSE,
		cost_der: Callable = MSE_derivative,
		seed: int = None,
	):
		"""Feed Forward Neural Network implementation.
		
		Parameters:
			layer_sizes: Tuple specifying number of nodes in each layer (input, hidden, output)
			hidden_funcs: Activation functions for hidden layers
			output_func: Activation function for output layer
			cost_func: Cost/Loss function
			cost_der: Derivative of cost function
			seed: Random seed for reproducibility
		"""
		# Set random seed for reproducibility
		if seed is not None:
			np.random.seed(seed)

		# Input validation
		assert len(layer_sizes) >= 2, "Need at least two layers (input and output)"

		n_hidden_layers = len(layer_sizes) - 2

		# Default to sigmoid activation if none provided
		if hidden_funcs is None:
			hidden_funcs = tuple([sigmoid] * n_hidden_layers)
		
		assert len(hidden_funcs) == n_hidden_layers, f"Expected {n_hidden_layers} hidden activation functions, got {len(hidden_funcs)}"

		self.layer_sizes = layer_sizes
		self.hidden_funcs = hidden_funcs	
		self.output_func = output_func
		self.cost_func = cost_func
		self.cost_der = cost_der

		# self.weights = list()
		# self.schedulers_weight = list()
		# self.schedulers_bias = list()

		self.classification = None

		self.reset_weights()
		self._set_classification()

	def fit(self, X, y, scheduler, epochs=100, batches=1, lmbda=0):
		"""
		Train the network using stochastic gradient descent.

		Paremeters:
			X: Training data
			y: Target values
			scheduler: Learning rate scheduler
			epochs: Number of epochs (Default = 100)
			batch_size: Size of mini-batches (1 = full batch)
			lmbda: Regularization parameter (Default = 0)	

		"""	
		batch_size = X.shape[0] // batches	

		training_scores = np.empty(epochs)
		training_scores.fill(np.nan)

		R2_scores = np.empty(epochs)
		R2_scores.fill(np.nan)

		train_accs = np.empty(epochs)
		train_accs.fill(np.nan)

		self.schedulers_weight = list()
		self.schedulers_bias = list()

		# create schedulers for each weight matrix
		for i in range(len(self.weights)):
			self.schedulers_weight.append(copy(scheduler))
			self.schedulers_bias.append(copy(scheduler))

		print(f"{scheduler.__class__.__name__}: Eta={scheduler.eta}, Lambda={lmbda}")

		for e in range(epochs):
			for i in range(0, X.shape[0], batch_size):
				# Mini-batch gradient descent
				if i == batches - 1:
					# Take the rest of the data at the last batch
					X_batch = X[i * batch_size]
					y_batch = y[i * batch_size]
				else:
					X_batch = X[i * batch_size : (i + 1) * batch_size]
					y_batch = y[i * batch_size : (i + 1) * batch_size]

				# Compute gradients
				self._backpropagation(X_batch, y_batch, lmbda)

				# # Update weights
				# for i, grad in enumerate(gradients):
				# 	self.weight[i] -= scheduler.update_change(grad)

			# reset schedulers for each epoch (some schedulers pass in this call)
			for scheduler in self.schedulers_weight:
				scheduler.reset()

			for scheduler in self.schedulers_bias:
				scheduler.reset()

			# Compute training scores
			pred_train = self.predict(X)

			training_score = self.cost_func(pred_train, y) 
			
			training_scores[e] = training_score
			
			if not self.classification:
				R2_score = R2(pred_train, y)
				R2_scores[e] = R2_score	
			else:
				accuracy = self._accuracy(X, y)
				train_accs[e] = accuracy
			
			 # printing progress bar
			progression = e / epochs
			print_length = self._progress_bar(
				progression,
				training_scores=training_scores[e],
				R2_scores=R2_scores[e],
				train_acc=train_accs[e],
			)

		# visualization of training progression (similiar to tensorflow progression bar)
		sys.stdout.write("\r" + " " * print_length)
		sys.stdout.flush()
		self._progress_bar(
			1,
			training_scores=training_scores[e],
			R2_scores=R2_scores[e],
			train_acc=train_accs[e],
		)
		sys.stdout.write("")

		# Return training scores for the entire run
		scores = dict()

		scores["cost"] = training_scores

		if not self.classification:
			scores["R2"] = R2_scores
		else:
			scores["accuracy"] = train_accs
		
		return scores

	def predict(self, X):
		"""Make predictions for given inputs."""
		return self._forward(X)
	
	def reset_weights(self):
		"""Reset weights and biases to random values."""
		self.weights = []
		for i in range(len(self.layer_sizes) - 1):
			# Initialize weight matrix with an extra column for bias
			layer_weight_matrix = np.random.randn(self.layer_sizes[i], self.layer_sizes[i + 1] + 1)
			# Initialize bias weights with small values
			layer_weight_matrix[:, 0] = np.random.randn(self.layer_sizes[i]) * 0.01
			
			self.weights.append(layer_weight_matrix)

	def _forward(self, inputs):
		"""Forward pass returning only final predictions."""
		activations, _ = self._forward_pass(inputs)
		return activations[-1]
	
	def _forward_pass(self, inputs):
		"""Forward pass that saves intermediate values for backpropagation.
		
		Parameters:
			inputs: Input data of shape (batch_size, input_nodes)
			
		Returns:
		tuple: Lists of activations and weighted sums for all layers
		"""
		activations = []
		weighted_sums = []
		
		# Add bias term to input
		bias = np.ones((inputs.shape[0])) * 0.01
		a = np.zeros((inputs.shape[0] + 1, inputs.shape[1]))
		a[1:] = inputs
		a[0] = bias
		activations.append(a)
		
		# Through hidden layers
		for (W, func) in zip(self.weights[:-1], self.hidden_funcs):
			weight, bias = W[1:], W[0]
			z = (a @ weight.T)[0] + bias
			weighted_sums.append(z)
			a = func(z)
			# Add bias term for next layer
			bias = np.ones((a.shape[0], 1)) * 0.01
			a = np.hstack([bias, a])
			activations.append(a)
		
		# Output layer
		z = a @ self.weights[-1]
		weighted_sums.append(z)
		activations.append(self.output_func(z))
		
		return activations, weighted_sums
	
	def _backpropagation(self, inputs, targets, lambd = 0):
		"""Computes gradients for network weights and biases using backpropagation.
		
		Parameters:
			inputs: Input data of shape (batch_size, input_nodes)
			targets: Target values of shape (batch_size, output_nodes)
			lambd: Regularization parameter (Default = 0)
		Returns:
			weight_gradients: List of weight gradients for each layer
			bias_gradients: List of bias gradients for each layer
		"""
		 # Forward pass to get activations and z-values
		activations, weighted_sums = self._forward_pass(inputs)
		
		 # Iterate through layers backwards
		for i in range(len(self.weights) - 1, -1, -1):
			# Compute delta for output layer
			if i == len(self.weights) - 1:
				if self.output_func.__name__ == "softmax":
					delta_matrix = activations[i + 1] - targets
				else:
					delta_matrix = derivate(self.output_func)(weighted_sums[-1])
					delta_matrix *= self.cost_der(activations[-1], targets)
			
			# Compute delta for hidden layers
			else:
				delta_matrix = (self.weights[i + 1][1:, :] @ delta_matrix.T).T
				print(i)
				delta_matrix *= derivate(self.hidden_funcs[i])(weighted_sums[i + 1])
			
			# Split activations into bias and rest
			a_bias = activations[i][:, 0].reshape(-1, 1)
			a_rest = activations[i][:, 1]
			
			# Calculate gradients
			gradient_weights = activations[i][:, 1:].T @ delta_matrix
			gradient_bias = np.sum(delta_matrix, axis=0).reshape(1, -1)
			
			# Add regularization (only to weights, not bias)
			gradient_weights += self.weights[i][1:, :] * lambd
			
			# Combine updates in format expected by schedulers
			update_matrix = np.vstack([
				self.schedulers_bias[i].update_change(gradient_bias),
				self.schedulers_weight[i].update_change(gradient_weights)
			])
			
			# Update weights
			self.weights[i] -= update_matrix
	
	def _accuracy(self, X, y):
		"""
		Calculate accuracy for binary classification predictions.
		"""
		predictions = self.predict(X)
		predictions = (predictions > 0.5).astype(int)  # Threshold at 0.5 for binary
		return np.mean(predictions == y)
	
	def _set_classification(self):
		"""
		Description:
		------------
			Decides if FFNN acts as classifier (True) og regressor (False),
			sets self.classification during init()
		"""
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
	
if __name__ == "__main__":
	from sklearn.preprocessing import StandardScaler
	from sklearn.model_selection import train_test_split

	seed = 42069
	np.random.seed(seed)


	N  = 100

	x = np.arange(0, 1, 1/N)
	y = np.arange(0, 1, 1/N)
	xx, yy = np.meshgrid(x, y)

	z = (FrankeFunction(xx, yy) + np.random.normal(0, 0.001, (N, N))).reshape(-1, 1) 

	poly_degree = 3
	X = create_X(x, y, poly_degree)

	X_train, X_test, y_train, y_test = train_test_split(X, z, test_size=0.2, random_state=seed)

	scaler = StandardScaler()
	X_train_scaled = scaler.fit_transform(X_train)
	X_test_scaled = scaler.transform(X_test)

	y_train_scaled = scaler.fit_transform(y_train)
	y_test_scaled = scaler.transform(y_test)

	input_shape = X_train_scaled.shape[1]
	hidden_shape = (4,2)
	output_shape = 1
	dims =  (input_shape, *hidden_shape, output_shape)

	model = FFNN(
		layer_sizes=dims,
		hidden_funcs=[sigmoid, sigmoid],
		output_func=identity,
		cost_func=MSE,
		cost_der=MSE_derivative,
		seed=seed
	)

	scheduler = Adagrad(eta=0.001)

	epochs = 1000
	batches = 10

	lmbda = 0.01

	scores = model.fit(
		X_train_scaled,
		y_train_scaled, 
		scheduler=scheduler,
		epochs=epochs,
		batches=batches,
		lmbda=lmbda,
		)

	z_pred_scaled = model.predict(X)
	z_pred = scaler.inverse_transform(z_pred_scaled)
	
	# Reshape prediction to match grid
	z_pred = z_pred.reshape(len(x), len(y))

	z2 = (FrankeFunction(xx, yy) + np.random.normal(0, 0.1, (N, N)))
	
	fig = plt.figure(figsize = (13, 7))
	axs = [fig.add_subplot(121, projection = "3d"), fig.add_subplot(122, projection = "3d")]

	surf_true = axs[0].plot_surface(xx, yy, z2, cmap = cm.coolwarm, linewidth = 0, antialiased = False)
	surf_predict = axs[1].plot_surface(xx, yy, z_pred, cmap = cm.coolwarm, linewidth = 0, antialiased = False)
	for i in range(2):
		axs[i].zaxis.set_major_locator(LinearLocator(10))
		axs[i].zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
		axs[i].set_xlabel(r"$x$")
		axs[i].set_ylabel(r"$y$")
	fig.colorbar(surf_true, shrink = 0.4, aspect = 10, label = r"$f(x,y)$")
	fig.colorbar(surf_predict, shrink = 0.4, aspect = 10, label = r"$f(x,y)+\varepsilon$")
	plt.tight_layout()
	# plt.savefig("../figs/a_Franke_surf.pdf")
	plt.show()