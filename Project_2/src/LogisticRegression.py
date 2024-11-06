import numpy as np
from activation_funcs import sigmoid

class LogisticRegression:
    def __init__(self, scheduler, lmbda=0):
        """
        Parameters:
        -----------
        scheduler: Learning rate scheduler (from your Scheduler.py)
        lmbda: Regularization parameter (default=0)
        """
        self.scheduler = scheduler
        self.lmbda = lmbda
        self.beta = None
        
    def cost_function(self, X, y, beta):
        """Binary cross-entropy cost function with L2 regularization"""
        n_samples = X.shape[0]
        probabilities = sigmoid(X @ beta)
        
        # Add small constant to avoid log(0)
        epsilon = 1e-15
        probabilities = np.clip(probabilities, epsilon, 1 - epsilon)
        
        cost = -np.sum(y * np.log(probabilities) + 
                      (1-y) * np.log(1-probabilities)) / n_samples
        
        # Add L2 regularization term
        if self.lmbda > 0:
            cost += self.lmbda * np.sum(beta**2)
            
        return cost
    
    def gradient(self, X, y, beta):
        """Compute gradient of cost function"""
        n_samples = X.shape[0]
        error = sigmoid(X @ beta) - y
        gradient = (X.T @ error) / n_samples
        
        # Add regularization term to gradient
        if self.lmbda > 0:
            gradient += 2 * self.lmbda * beta
            
        return gradient
    
    def fit(self, X, y, epochs=100, batch_size=32):
        """
        Train using mini-batch SGD
        """
        n_samples, n_features = X.shape
        self.beta = np.zeros((n_features, 1))
        
        n_batches = n_samples // batch_size
        
        for _ in range(epochs):
            indices = np.random.permutation(n_samples)
            
            for i in range(n_batches):
                batch_start = i * batch_size
                batch_end = batch_start + batch_size
                batch_idx = indices[batch_start:batch_end]
                
                X_batch = X[batch_idx]
                y_batch = y[batch_idx]
                
                gradient = self.gradient(X_batch, y_batch, self.beta)
                update = self.scheduler.update_change(gradient)
                self.beta -= update
                
        return self
    
    def predict(self, X):
        """Predict class labels"""
        probabilities = sigmoid(X @ self.beta)
        return (probabilities >= 0.5).astype(int)
    
    def accuracy(self, X, y):
        """Calculate classification accuracy"""
        predictions = self.predict(X)
        return np.mean(predictions == y)

