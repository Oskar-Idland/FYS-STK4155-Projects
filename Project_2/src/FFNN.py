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
		hidden_der: tuple[Callable] = None,
		output_func: Callable = identity,
		cost_func: Callable = MSE,
		cost_der: Callable = MSE_derivative,
		seed: int = None,
	):
		"""Feed Forward Neural Network implementation.
		
		Parameters:
			dimensions: Tuple specifying number of nodes in each layer (input, hidden, output)
			hidden_funcs: Activation functions for hidden layers
			hidden_der: Derivatives of hidden layer activation functions
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
		if hidden_der is None:
			hidden_der = tuple([sigmoid_derivative] * n_hidden_layers)
		
		assert len(hidden_funcs) == n_hidden_layers, f"Expected {n_hidden_layers} hidden activation functions, got {len(hidden_funcs)}"
		assert len(hidden_der) == n_hidden_layers, f"Expected {n_hidden_layers} hidden activation derivatives, got {len(hidden_der)}"

		self.dimensions = dimensions
		self.hidden_funcs = hidden_funcs	
		self.hidden_der = hidden_der
		self.output_func = output_func
		self.cost_func = cost_func
		self.cost_der = cost_der
		
		# Initialize weights and biases
		self.weights = []
		self.biases = []

		for i in range(len(dimensions) - 1):
			W = np.random.randn(dimensions[i], dimensions[i + 1])

			b = np.random.randn(1, dimensions[i + 1])

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
			tuple: (activations, weighted_sums)
				- activations: List of activations for each layer
				- weighted_sums: List of weighted sums for each layer
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
	
	def _backpropagation(self, inputs, targets):
		gradients = []
		activations, weighted_sums = self._forward_pass(inputs)

		dC_da = self.cost_der(activations[-1], targets)	

		for i in reversed(range(len(self.dimensions))):
			W, b = self.weights[i], self.biases[i]

			dW = np.dot(activations[i].T, dC_da)
			db = np.sum(dC_da, axis=0, keepdims=True)
			gradients.append((dW, db))

			if i > 0:
				dC_da = np.dot(dC_da, W.T) * self.hidden_der[i - 1](weighted_sums[i - 1])

		return gradients[::-1]
	
if __name__ == "__main__":
		
	dimensions = (64, 32, 16, 10)  # Input layer: 64, 2 hidden layers: 32 & 16, output: 10
	net = FFNN(
		dimensions=dimensions,
		hidden_funcs=(ReLU, ReLU),  # Activation for the two hidden layers
		hidden_der=(ReLU_derivative, ReLU_derivative),
		cost_func=MSE,
		cost_der=MSE_derivative
	)