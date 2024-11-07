import autograd.numpy as np
from typing import Callable
import math
import sys
from copy import copy

from activation_funcs import identity, sigmoid, derivate
from Schedulers import *
from utils import MSE, MSE_derivative, R2, FrankeFunction, create_X

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
		self.seed = seed
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

		self.classification = None

		self.reset_weights()
		self._set_classification()

	def fit(self, X, y, scheduler, epochs=100, batches=1, lmbda=0, convergence_tol: float = None, X_test: np.ndarray = None, y_test: np.ndarray = None):
		
		"""
		Train the network using stochastic gradient descent.

		Parameters:
			X (np.ndarray): Training data.
			y (np.ndarray): Target values.
			scheduler: Learning rate scheduler.
			epochs (int, optional): Number of epochs (Default = 100).
			batches (int, optional): Number of mini-batches (Default = 1).
			lmbda (float, optional): Regularization parameter (Default = 0).
			convergence_tol (float, optional): Tolerance for convergence (Default = None).
			X_test (np.ndarray, optional): Test data (Default = None).
			y_test (np.ndarray, optional): Test target values (Default = None).

		Returns:
			dict: Dictionary containing training scores and optionally test scores and accuracies.
			int (optional): Epoch at which convergence was achieved, if convergence_tol is specified.	
		"""	
		if self.seed is not None:
			np.random.seed(self.seed)

		tes_set = False
		if X_test is not None and y_test is not None:
			tes_set = True
		

		batch_size = X.shape[0] // batches	

		training_scores = np.empty(epochs)
		training_scores.fill(np.nan)

		R2_scores = np.empty(epochs)
		R2_scores.fill(np.nan)

		test_scores = np.empty(epochs)
		test_scores.fill(np.nan)

		R2_test = np.empty(epochs)
		R2_test.fill(np.nan)

		train_accs = np.empty(epochs)
		train_accs.fill(np.nan)

		test_accs = np.empty(epochs)
		test_accs.fill(np.nan)

		self.schedulers_weight = list()
		self.schedulers_bias = list()

		# create schedulers for each weight matrix
		for i in range(len(self.layer_weights)):
			self.schedulers_weight.append(copy(scheduler))
			self.schedulers_bias.append(copy(scheduler))

		# X, y = resample(X, y, random_state=seed)
		print(f"{scheduler.__class__.__name__}: Eta={scheduler.eta}, Lambda={lmbda}")

		n_samples = X.shape[0]

		if convergence_tol is not None:
			convergence_check = True
			recent_scores = []

		for e in range(epochs):
			indices = np.arange(n_samples)
			np.random.shuffle(indices)
			X = X[indices]
			y = y[indices]
			for i in range(0, X.shape[0], batch_size):
				
				end = min(i + batch_size, n_samples)
				X_batch = X[i:end]
				y_batch = y[i:end]

				# Compute gradients
				self._backpropagation(X_batch, y_batch, lmbda)

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

				if tes_set:
					pred_test = self.predict(X_test)
					test_scores[e] = self.cost_func(pred_test, y_test)
					R2_test[e] = R2(pred_test, y_test)

			else:
				train_accuracy = self._accuracy(pred_train, y)
				train_accs[e] = train_accuracy

				if tes_set:
					pred_test = self.predict(X_test)
					test_scores[e] = self.cost_func(pred_test, y_test)
					test_accuracy = self._accuracy(pred_test, y_test)
					test_accs[e] = test_accuracy

			 # printing progress bar
			progression = e / epochs
			print_length = self._progress_bar(
				progression,
				training_scores=training_scores[e],
				R2_test=R2_test[e],
				test_scores=test_scores[e],
				R2_scores=R2_scores[e],
				train_acc=train_accs[e],
				test_acc=test_accs[e],
			)

			# Check for convergence
			if convergence_tol is not None and convergence_check:
				recent_scores.append(training_score)

				# Only check convergence after collecting enough epochs
				if len(recent_scores) >= 10:
					# Keep only the most recent values
					if len(recent_scores) > 10:
						recent_scores.pop(0)
						
					# Calculate mean and std of recent MSEs
					mean_mse = np.mean(recent_scores)
					# Check if variation is below tolerance
					if abs(mean_mse - training_score) / training_score <=  convergence_tol:
						convergence_epoch = e
						print(f"\nConverged at epoch {convergence_epoch} with MSE stability below {convergence_tol:.2e}")
						convergence_check = False
			

		# visualization of training progression (similiar to tensorflow progression bar)
		sys.stdout.write("\r" + " " * print_length)
		sys.stdout.flush()
		self._progress_bar(
			1,
			training_scores=training_scores[e],
			R2_test=R2_test[e],
			test_scores=test_scores[e],
			R2_scores=R2_scores[e],
			train_acc=train_accs[e],
			test_acc=test_accs[e],
		)
		sys.stdout.write("")

		# Return training scores for the entire run
		scores = dict()

		scores["cost"] = training_scores

		if not self.classification:
			scores["R2"] = R2_scores
			if tes_set:
				scores["test_cost"] = test_scores
				scores["test_R2"] = R2_test
		else:
			scores["train_accuracy"] = train_accs
			if tes_set:
				scores["test_cost"] = test_scores
				scores["test_accuracy"] = test_accs
		
		if convergence_tol is not None:
				return scores, convergence_epoch

		return scores

	def predict(self, X, threshold=0.5):
		"""Make predictions for given inputs."""
		predictions = self._forward(X)
		if self.classification:
			return (predictions > threshold).astype(int)
		return predictions
	
	def reset_weights(self):
		"""Reset weights and biases to random values."""

		if self.seed is not None:
			np.random.seed(self.seed)

		self.layer_weights = []
		self.layer_biases = []
		for i in range(len(self.layer_sizes) - 1):
			layer_weight = np.random.randn(self.layer_sizes[i], self.layer_sizes[i + 1])
			layer_biases = np.random.randn(self.layer_sizes[i + 1]) * 0.01
			
			self.layer_weights.append(layer_weight)
			self.layer_biases.append(layer_biases)

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

		z_values = [] # List of weighted sums for each layer
		activations = [inputs] # List of activations for each layer

		# Iterate through hidden layers
		for i, (W, b, func) in enumerate(zip(self.layer_weights[:-1], self.layer_biases[:-1], self.hidden_funcs)):
			z = activations[i] @ W + b
			a = func(z)

			z_values.append(z)
			activations.append(a)
			
		# Output layer
		z = activations[-1] @ self.layer_weights[-1] + self.layer_biases[-1]
		z_values.append(z)

		a = self.output_func(z)
		activations.append(a)
		
		return activations, z_values
	
	def _backpropagation(self, inputs, targets, lmbda = 0):
		"""Computes gradients for network weights and biases using backpropagation.
		
		Parameters:
			inputs: Input data of shape (batch_size, input_nodes)
			targets: Target values of shape (batch_size, output_nodes)
			lambd: Regularization parameter (Default = 0)
		Returns:
			weight_gradients: List of weight gradients for each layer
			bias_gradients: List of bias gradients for each layer
		"""
		if self.seed is not None:
			np.random.seed(self.seed)

		 # Forward pass to get activations and z-values
		activations, z_values = self._forward_pass(inputs)
		

		# Compute gradients for output layer
		if self.classification:
			delta = activations[-1] - targets
		else:
			delta = self.cost_der(activations[-1], targets)	
			delta *= derivate(self.output_func)(z_values[-1]) 
		

		 # Iterate backwards through layers 
		for i in range(len(self.layer_weights) - 1, -1, -1):
			# Compute gradients
			dW = activations[i].T @ delta
			db = np.sum(delta, axis=0, keepdims=True)

		
			# Propagate error to previous layer
			if i > 0: # Skip if input layer
				delta = delta @ self.layer_weights[i].T
				delta *= derivate(self.hidden_funcs[i-1])(z_values[i-1])

			# Add regularization term
			dW += lmbda * self.layer_weights[i]

			# Update weights and biases
			weight_update = self.schedulers_weight[i].update_change(dW)
			bias_update = self.schedulers_bias[i].update_change(db)

			self.layer_weights[i] -= weight_update
			self.layer_biases[i] -= bias_update.reshape(-1)

	def _accuracy(self, predictions, target):
		"""
		Calculate accuracy for binary classification predictions.
		"""
		assert predictions.size == target.size
		predictions = (predictions > 0.5).astype(int)
		return np.mean((target == predictions))

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
			print("Classification task detected")

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
		return f"{value:4>.{decimals-n-1}f}"
	
if __name__ == "__main__":
	from sklearn.preprocessing import StandardScaler
	from sklearn.model_selection import train_test_split

	seed = 42069
	np.random.seed(seed)


	N  = 100
	dx = 1/N

	x = np.linspace(0, 1, N)
	y = np.linspace(0, 1, N)
	xx, yy = np.meshgrid(x, y)

	xx = xx.flatten().reshape(-1,1)
	yy = yy.flatten().reshape(-1,1)

	zz = FrankeFunction(xx, yy)
	target = zz.reshape(-1,1)

	poly_degree = 4

	scaler_x = StandardScaler()
	scaler_y = StandardScaler()
	x_scaled = scaler_x.fit_transform(xx)
	y_scaled = scaler_y.fit_transform(yy)

	X = np.hstack([x_scaled, y_scaled])

	target = (target - np.mean(target)) / np.std(target)

	input_shape = X.shape[1]
	hidden_shape = [50, 50]
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

	model.reset_weights()

	# scheduler = Constant(eta=0.001)
	scheduler = Adam(eta=0.001, rho=0.9, rho2=0.999)

	epochs = 1000
	batches = 32

	lmbda = 0.001

	scores = model.fit(
		X,
		target, 
		scheduler=scheduler,
		epochs=epochs,
		batches=batches,
		lmbda=lmbda,
		)

	z_pred = model.predict(X)
	
	grid_size = int(np.sqrt(len(xx)))
	xx = xx.reshape((grid_size, grid_size))
	yy = yy.reshape((grid_size, grid_size))
	zz = zz.reshape((grid_size, grid_size))
	z_pred = z_pred.reshape((grid_size, grid_size))


	fig = plt.figure(figsize = (13, 7))
	axs = [fig.add_subplot(121, projection = "3d"), fig.add_subplot(122, projection = "3d")]

	surf_true = axs[0].plot_surface(xx, yy, zz, cmap = cm.coolwarm, linewidth = 0, antialiased = False)
	axs[0].set_title("True Franke function")
	axs[1].set_title("Predicted Franke function")
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

