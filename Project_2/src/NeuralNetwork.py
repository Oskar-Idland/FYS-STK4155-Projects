import autograd.numpy as np  # We need to use this numpy wrapper to make automatic differentiation work later
from autograd import grad

class NeuralNetwork:
    def __init__(self, network_input_size, layer_output_sizes, activation_funcs, activation_ders, cost_func, cost_der):
        # Initialize network structure and hyperparameters
        self.layers = self.create_layers(network_input_size, layer_output_sizes)
        self.activation_funcs = activation_funcs
        self.activation_ders = activation_ders
        self.cost_func = cost_func
        self.cost_der = cost_der

    def create_layers(self, network_input_size, layer_output_sizes):
        layers = []
        input_size = network_input_size
        for layer_output_size in layer_output_sizes:
            # Initialize weights and biases randomly
            W = np.random.randn(input_size, layer_output_size)
            b = np.random.randn(layer_output_size)
            layers.append((W, b))
            input_size = layer_output_size
        return layers

    def predict(self, inputs):
        # Simple feed forward pass
        a = inputs
        for (W, b), activation_func in zip(self.layers, self.activation_funcs):
            z = a @ W + b
            a = activation_func(z)
        return a

    def cost(self, inputs, targets):
        # Compute the cost function based on the predictions
        predictions = self.predict(inputs)
        return self.cost_func(predictions, targets)

    def _feed_forward_saver(self, inputs):
        # Save the activations and linear transformations (zs) for backpropagation
        layer_inputs = []
        zs = []
        a = inputs
        for (W, b), activation_func in zip(self.layers, self.activation_funcs):
            layer_inputs.append(a)
            z = a @ W + b
            a = activation_func(z)
            zs.append(z)
        return layer_inputs, zs, a

    def compute_gradient(self, inputs, targets):
        # Compute the gradients of the cost function w.r.t the weights and biases using backpropagation
        layer_inputs, zs, prediction = self._feed_forward_saver(inputs)
        layer_grads = [() for layer in self.layers]
        
        # Compute the cost derivative for the last layer
        dC_da = self.cost_der(prediction, targets)

        for i in reversed(range(len(self.layers))):
            # For the last layer, dC/dz(L) = dC/da(L) * da/dz(L)
            dC_dz = dC_da * self.activation_ders[i](zs[i])
            dC_dW = dC_dz.T @ layer_inputs[i]
            dC_db = dC_dz
            
            # Store gradients
            layer_grads[i] = (dC_dW.T, dC_db.T)
            
            # Propagate the error back to the previous layers
            if i > 0:
                W, _ = self.layers[i]
                dC_da = dC_dz @ W.T  # Backpropagate error for next layer

        return layer_grads

    def update_weights(self, layer_grads, learning_rate):
        # Update the weights and biases based on computed gradients
        for i, (dW, db) in enumerate(layer_grads):
            W, b = self.layers[i]
            self.layers[i] = (W - learning_rate * dW, b - learning_rate * db)

    #TODO these two are not necessary for the project
    def autograd_compliant_predict(self, layers, inputs):
        # This function seems to mimic autograd predictions for comparison
        a = inputs
        for (W, b), activation_func in zip(layers, self.activation_funcs):
            z = a @ W + b
            a = activation_func(z)
        return a

    def autograd_gradient(self, inputs, targets):
        # Wrapper function to check against autograd, could call an external grad() function
        predictions = self.autograd_compliant_predict(self.layers, inputs)
        return grad(self.cost_func)(predictions, targets)