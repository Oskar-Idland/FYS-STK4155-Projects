import autograd.numpy as np  # We need to use this numpy wrapper to make automatic differentiation work later

#TODO old class, maybe edit to work with both linear regression and classification?
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

    def compute_gradients(self, inputs, targets):
        # Compute the gradients of the cost function w.r.t the weights and biases using backpropagation
        layer_inputs, zs, predictions = self._feed_forward_saver(inputs)
        layer_grads = [() for layer in self.layers]
        
        # Compute the cost derivative for the last layer
        dC_da = self.cost_der(predictions, targets)

        for i in reversed(range(len(self.layers))):
            # For the last layer, dC/dz(L) = dC/da(L) * da/dz(L)
            dC_dz = dC_da * self.activation_ders[i](zs[i])
            dC_dW = dC_dz.T @ layer_inputs[i]
            dC_db = np.sum(dC_dz, axis = 0)
            
            # Store gradients
            layer_grads[i] = (dC_dW.T, dC_db.T)
            
            # Propagate the error back to the previous layers
            if i > 0:
                W, _ = self.layers[i]
                dC_da = dC_dz @ W.T  # Backpropagate error for next layer

        return layer_grads

        #TODO fix this
        # a_1, a_2 = forwardpropagation(x)
        # # parameter delta for the output layer, note that a_2=z_2 and its derivative wrt z_2 is just 1
        # delta_2 = a_2 - y
        # print(0.5*((a_2-y)**2))
        # # delta for  the hidden layer
        # delta_1 = np.matmul(delta_2, w_2.T) * a_1 * (1 - a_1)
        # # gradients for the output layer
        # output_weights_gradient = np.matmul(a_1.T, delta_2)
        # output_bias_gradient = np.sum(delta_2, axis=0)
        # # gradient for the hidden layer
        # hidden_weights_gradient = np.matmul(x.T, delta_1)
        # hidden_bias_gradient = np.sum(delta_1, axis=0)
        # return output_weights_gradient, output_bias_gradient, hidden_weights_gradient, hidden_bias_gradient

    def update_weights(self, layer_grads, learning_rate):
        #TODO fix with momentum?
        #TODO how to do with regression?
        # Update the weights and biases based on computed gradients
        for i, (W, b) in enumerate(self.layers):
            W_g, b_g = layer_grads[i]
            self.layers[i] = (W - learning_rate * W_g, b - learning_rate * b_g)
    
    def update_weights_sgd(self, input_data, target_data, learning_rate, momentum, epochs, batch_size):
        # Initialize velocity for each layer to zero
        velocities = [(np.zeros_like(W), np.zeros_like(b)) for W, b in self.layers]
        
        for epoch in range(epochs):
            # Shuffle the data
            indices = np.random.permutation(len(input_data))
            input_data, target_data = input_data[indices], target_data[indices]
            
            for start in range(0, len(input_data), batch_size):
                # Get a mini-batch
                end = start + batch_size
                inputs = input_data[start:end]
                targets = target_data[start:end]
                
                # Compute gradients using backpropagation
                layer_grads = self.compute_gradients(inputs, targets)
                
                # Update weights and biases with momentum
                for i, (W, b) in enumerate(self.layers):
                    W_g, b_g = layer_grads[i]  # Get gradients for layer i
                    
                    # Update velocities
                    vW, vb = velocities[i]
                    vW = momentum * vW - learning_rate * W_g  # Momentum update for weights
                    vb = momentum * vb - learning_rate * b_g  # Momentum update for biases
                    
                    # Update parameters
                    self.layers[i] = (W + vW, b + vb)
                    
                    # Store updated velocities
                    velocities[i] = (vW, vb)