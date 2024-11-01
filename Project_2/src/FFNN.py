import autograd.numpy as np
from typing import Callable

from activation_funcs import *
from cost_funcs import *
from Schedulers import *
from functions import MSE, MSE_derivative, R2

# warnings.simplefilter("error")

class FFNN:
	def __init__(
		self,
		dimensions: tuple[int],
		hidden_funcs: tuple[Callable] = None,
		output_func: Callable = identity,
		cost_func: Callable = MSE,
		cost_der: Callable = MSE_derivative,
		seed: int = None,
	):
		"""Feed Forward Neural Network implementation.
		
		Parameters:
			dimensions: Tuple specifying number of nodes in each layer (input, hidden, output)
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
		assert len(dimensions) >= 2, "Need at least two layers (input and output)"

		n_hidden_layers = len(dimensions) - 2

		# Default to sigmoid activation if none provided
		if hidden_funcs is None:
			hidden_funcs = tuple([sigmoid] * n_hidden_layers)
		
		assert len(hidden_funcs) == n_hidden_layers, f"Expected {n_hidden_layers} hidden activation functions, got {len(hidden_funcs)}"

		self.dimensions = dimensions
		self.hidden_funcs = hidden_funcs	
		self.output_func = output_func
		self.cost_func = cost_func
		self.cost_der = cost_der

		self.classification = None

		self.reset_weights()
		self._set_classification()

	def fit(self, X, y, scheduler, epochs=100, batch_size=None, lmbda=0):
		"""
		Train the network using stochastic gradient descent.

		Paremeters:
			X: Training data
			y: Target values
			scheduler: Learning rate scheduler
			epochs: Number of epochs (Default = 100)
			batch_size: Size of mini-batches (None = full batch)
			lmbda: Regularization parameter (Default = 0)	

		"""	
		# Set batch size to full batch if None
		if batch_size is None:
			batch_size = X.shape[0]
		

		# Track scoring parameters
		training_scores = {
			"cost": [],
			"R2": [] if not self.classification else None,
			"accuracy": [] if self.classification else None
		}

		for epoch in range(epochs):
			# Create mini-batches
			indices = np.random.permutation(X.shape[0])
			for i in range(0, X.shape[0], batch_size):
				batch_indices = indices[i:i+batch_size]
				X_batch = X[batch_indices]
				y_batch = y[batch_indices]

				# Compute gradients
				gradients = self._backpropagation(X_batch, y_batch, lmbda)

				# Update weights and biases
				for layer in range(len(self.weights)):
					dW, db = gradients[layer]
					self.weights[layer] -= scheduler.update_change(dW)
					self.biases[layer] -= scheduler.update_change(db)

			# Compute training scores
			pred_train = self.predict(X)
			training_score = self.cost_func(pred_train, y) 
			
			training_scores["cost"].append(training_score) 
			
			if not self.classification:
				R2_score = R2(pred_train, y)
				training_scores["R2"].append(R2_score)	
			else:
				accuracy = self._accuracy(X, y)
				training_scores["accuracy"].append(accuracy)

		return training_scores

	def predict(self, X):
		"""Make predictions for given inputs."""
		return self._forward(X)
	
	def reset_weights(self):
		"""Reset weights and biases to random values."""
		self.weights = []
		self.biases = []
		for i in range(len(self.dimensions) - 1):
			W = np.random.randn(self.dimensions[i], self.dimensions[i + 1])
			b = np.random.randn(1, self.dimensions[i + 1])
			self.weights.append(W)
			self.biases.append(b)

	def _forward(self, inputs):
		"""
		Forward pass through the network.

		Parameters:
			inputs: Input data of shape (batch_size, input_nodes)

		Returns:
			tuple_ (activations, weighted_sums)
				- activations: List of activations for each layer
				- weighted_sums: List of weighted sums for each layer
		"""

		a = inputs
		activations = [a] # List to store activations for each layer
		weighted_sums = []

		# Loop through the hidden layers
		for (W, b, func) in zip(self.weights[:-1], self.biases[:-1], self.hidden_funcs):
			z = np.dot(a, W) + b
			a = func(z)
		
		# Handle output layer separately with output activation function
		W = self.weights[-1]
		b = self.biases[-1]
		z = np.dot(a, W) + b

		weighted_sums.append(z)
		activations.append(self.output_func(z))

		return self.output_func(z)
	
	def _forward_pass(self, inputs):
		"""Forward pass that saves intermediate values for backpropagation.
		
		Parameters:
			inputs: Input data of shape (batch_size, input_nodes)
			
		Returns:
			Output predictions from the network
		"""
		a = inputs
		activations = [a]  # Include input layer
		weighted_sums = []

		# Handle hidden layers
		for (W, b, func) in zip(self.weights[:-1], self.biases[:-1], self.hidden_funcs):
			z = np.dot(a, W) + b
			weighted_sums.append(z)
			a = func(z)
			activations.append(a)
		
		# Handle output layer separately
		W = self.weights[-1]
		b = self.biases[-1]
		z = np.dot(a, W) + b

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
		list: List of tuples (dW, db) containing gradients for each layer
		"""
		# Forward pass to get all activations and weighted sums
		activations, weighted_sums = self._forward_pass(inputs)

		gradients = []
		n_layers = len(self.weights)

		delta = self.cost_der(activations[-1], targets)	
		if self.output_func != identity:  # Only apply if non-identity output function
			delta *= derivate(self.output_func)(weighted_sums[-1])

		# Calculate gradients for output layer
		dW = np.dot(activations[-2].T, delta) + lambd * self.weights[-1]
		db = np.sum(delta, axis=0, keepdims=True)
		gradients.append((dW, db))

		# Backpropagate through hidden layers
		for l in range(n_layers -2, -1, -1): # Loop backwards through layers
			delta = np.dot(delta, self.weights[l+1].T) 
			delta *= derivate(self.hidden_funcs[l])(weighted_sums[l])
			
			# Calculate gradients for hidden layer
			dW = np.dot(activations[l].T, delta) + lambd * self.weights[l]
			db = np.sum(delta, axis=0, keepdims=True)
			gradients.append((dW, db))

		return gradients[::-1]
	
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
	
if __name__ == "__main__":
		
	dimensions = (64, 32, 16, 10)  # Input layer: 64, 2 hidden layers: 32 & 16, output: 10
	net = FFNN(
		dimensions=dimensions,
		hidden_funcs=(ReLU, ReLU),  # Activation for the two hidden layers
		cost_func=MSE,
		cost_der=MSE_derivative
	)