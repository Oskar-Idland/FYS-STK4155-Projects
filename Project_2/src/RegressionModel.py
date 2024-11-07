import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, KFold
from sklearn.utils import resample
from autograd import grad
from jax import jit
from Schedulers import Adagrad, AdagradMomentum, RMS_prop, Adam, Constant, Momentum


class RegressionModel:
    def __init__(self, x: list | np.ndarray, y: np.ndarray, degree: int, test_size: float = 0.2, seed: int | None = None):
        """
        Class initializer.

        ## Parameters:
        x (list | np.ndarray): The independent variable(s). If there are two independent variables, x should be a list where the first (second) element is a np.ndarray representing the first (second) variable.
        y (np.ndarray): The dependent variable.
        degree (int): The degree of the polynomial features.
        test_size (float, optional): The portion of the dataset to include in the test split. Default is 0.2.
        seed(int | None, optional): The random seed for reproducibility. Default is None.
        """
        self.x, self.y, self.degree, self.test_size, self.seed = x, y, degree, test_size, seed
        
        self.X = self.create_X()
        self._split()
        self._scale()

    def _split(self):
        """
        Splits stored data into training and test sets.
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size = self.test_size, random_state = self.seed)

    def _scale(self):
        """
        Scales stored training and test data.
        """
        self.scaler_X = StandardScaler().fit(self.X_train)
        self.X_train  = self.scaler_X.transform(self.X_train)
        self.X_test   = self.scaler_X.transform(self.X_test)

        self.scaler_y = StandardScaler().fit(self.y_train)
        self.y_train  = self.scaler_y.transform(self.y_train)
        self.y_test   = self.scaler_y.transform(self.y_test)

    
    """ 
    Setters 
    """
    def set_degree(self, degree: int):
        """
        Sets the degree used of the polynomial features.
        """
        self.degree = degree

        # Update design matrix, training and test data
        self.X = self.create_X()
        self._split()
        self._scale()

    def set_test_size(self, test_size: float):
        """
        Sets the portion of the dataset to include in the test split.
        """
        self.test_size = test_size

        # Update design matrix, training and test data
        self.X = self.create_X()
        self._split()
        self._scale()

    """
    Getters
    """
    def create_X(self) -> np.ndarray:
        """
        Creates the design matrix X.

        ## Returns:
        np.ndarray: The design matrix X.
        """
        if isinstance(self.x, list):
            x1, x2 = self.x
            if len(x1.shape) > 1:
                x1 = np.ravel(x1)
            if len(x2.shape) > 1:
                x2 = np.ravel(x2)

            N = int(len(x1) * len(x2))                          # Number of rows in the design matrix
            l = int((self.degree + 1) * (self.degree + 2) / 2)  # Number of columns in the design matrix
            X = np.ones((N, l))
            
            x1, x2 = np.meshgrid(x1, x2)   # Make a meshgrid to get all possible combinations of x1 and x2 values
            x1 = x1.flatten()
            x2 = x2.flatten()

            idx = 1
            for i in range(1, self.degree + 1):
                for j in range(i + 1):
                    X[:, idx] = x1**(i - j) * x2**j
                    idx += 1
        
        else:
            x = self.x
            if len(x.shape) > 1:
                x = np.ravel(x)

            l = int(self.degree + 1)   # Number of columns in the design matrix
            X = np.ones((len(x), l))

            idx = 1
            for i in range(1, l):
                X[:, i] = x**i

        return X

    def get_MSE(self, y_pred: np.ndarray) -> float:
        """
        Calculates the Mean-Squared Error (MSE) between the true and predicted values.

        ## Parameters:
        y_pred (np.ndarray): The data values predicted by the model.

        ## Returns:
        float: The Mean-Squared Error.
        """
        y_test = self.y_test.flatten()
        y_pred = y_pred.flatten()
        n = len(y_test)
        return np.sum((y_test - y_pred)**2) / n

    def get_R2(self, y_pred: np.ndarray) -> float:
        """
        Calculates the R2 score of the model.

        ## Parameters:
        y_pred (np.ndarray): The data values predicted by the model.

        ## Returns:
        float: The R2 score.
        """
        y_test = self.y_test.flatten()
        y_pred = y_pred.flatten()
        return 1 - np.sum((y_test - y_pred)**2) / np.sum((y_test - np.mean(y_test))**2)
    

    """
    Linear regression methods (OLS and Ridge combined into one)
    """
    def linear_regression(self, lmbd: float = 0.0, return_theta: bool = False) -> tuple[np.ndarray]:
        """
        Performs linear regression using either Ordinary Least Squares (OLS) or Ridge.

        ## Parameters:
        lmbd (float, optional): The regularization parameter in Ridge regression. Passing as zero corresponds to OLS. Default is 0.0.

        ## Returns:
        tuple: A tuple containing the Mean-Squared Error (MSE) score and the R-squared (R2) score, as well as the theta values (coefficients) if return_theta is passed as True.
        """
        theta = np.linalg.pinv(self.X_train.T @ self.X_train + lmbd * np.eye(self.X_train.shape[1])) @ self.X_train.T @ self.y_train
    
        y_pred = self.X_test @ theta
    
        MSE = self.get_MSE(y_pred)
        R2  = self.get_R2(y_pred)

        quantities = [MSE, R2]
        if return_theta:
            quantities.append(theta)
        return quantities
    
    
    """ 
    Gradient descent methods 
    """
    def gradient_descent(self, n_iter: int, eta: float | None = None, tuning_method: str  = 'Const', rho_1: float | None = None, rho_2: float | None = None, lmbd: float | None = 0.0, gamma: float | None = 0.0, use_autograd: bool = False, return_theta: bool = False) -> tuple[np.ndarray]:
        """
        Perform gradient descent optimization to estimate model parameters.
        Parameters
        ----------
        n_iter : int
            Number of iterations for the gradient descent.
        eta : float, optional
            Learning rate for the gradient descent. If None, it will be set based on the largest eigenvalue of the Hessian matrix.
        tuning_method : str, optional
            Method for tuning the gradient descent. Options are 'Const', 'Momentum', 'Adagrad', 'AdagradMomentum', 'RMS_prop', and 'Adam'. Default is 'Const'.
        rho_1 : float, optional
            Decay rate for RMS_prop and Adam methods. Must be between 0.0 and 1.0.
        rho_2 : float, optional
            Second decay rate for Adam method. Must be between 0.0 and 1.0.
        lmbd : float, optional
            Regularization parameter. Default is 0.0.
        gamma : float, optional
            Momentum parameter. Must be between 0.0 and 1.0. Default is 0.0.
        use_autograd : bool, optional
            If True, use autograd for automatic differentiation. Default is False.
        return_theta : bool, optional
            If True, return the estimated theta values. Default is False.
        Returns
        -------
        tuple[np.ndarray]
            A tuple containing the Mean Squared Error (MSE) and R-squared (R2) values. If return_theta is True, the estimated theta values are also included.
        Raises
        ------
        ValueError
            If an invalid tuning method is provided.
        AssertionError
            If required parameters for RMS_prop or Adam methods are not provided or are out of bounds.
        """

        np.random.seed(self.seed)
       
        if eta < 0.0 or eta > 1.0:
            H = 2.0 * ((1.0 / len(self.y_train)) * self.X_train.T @ self.X_train + lmbd * np.eye(self.X_train.shape[1])) 
            eigvals, eigvecs = np.linalg.eig(H)
            eta = 1.0 / np.max(eigvals)
      

        if gamma < 0.0 or gamma > 1.0:
            print("The momentum parameter gamma must be between 0.0 and 1.0! Setting to 0.0.")
            gamma = 0.0

        if use_autograd:
            def cost_func(theta):
                return (1.0 / len(self.y_train)) * np.sum((self.X_train @ theta - self.y_train)**2) + lmbd * np.sum(theta**2)
        else:
            def training_gradient(theta):
                return 2.0 * ((1.0 / len(self.y_train)) * self.X_train.T @ (self.X_train @ theta - self.y_train) + lmbd * theta)
            
            
        match tuning_method.lower():
            case "const":    # Uses a decaying learning rate
                gradient_change = Constant(eta)
            case "momentum":
                gradient_change = Momentum(eta, gamma)
            case "adagrad":
                gradient_change = Adagrad(eta)
            case "adagradmomentum":
                gradient_change = AdagradMomentum(eta, gamma)
            case "rms_prop":
                success = rho_1 >= 0.0 and rho_1 <= 1.0
                assert success, f"The decay rate rho_1 must be passed for RMS_prop. Default is None, got {rho_1}."
                gradient_change = RMS_prop(eta, rho_1)
            case "adam":
                success = rho_1 >= 0.0 and rho_1 <= 1.0 and rho_2 >= 0.0 and rho_2 <= 1.0
                assert success, f"The decay rates rho_1 and rho_2 must be passed for Adam. Default is None, got {rho_1} and {rho_2}."
                gradient_change = Adam(eta, rho_1, rho_2)
            case _:
                raise ValueError(f"Invalid tuning method: {tuning_method}.")
            
        
        # Estimating theta's with gradient descent
        theta = np.random.randn(self.X.shape[1], 1) 
        for _ in range(n_iter):
            gradients = training_gradient(theta)
            change = gradient_change.update_change(gradients)
            theta -= change
    
        y_pred = self.X_test @ theta
        
        MSE = self.get_MSE(y_pred)
        R2  = self.get_R2(y_pred)

        quantities = [MSE, R2]
        if return_theta:
            quantities.append(theta)
            
        gradient_change.reset()
        return quantities

    def stochastic_gradient_descent(self, n_epochs: int, M: int, eta: float | None = None, tuning_method: str = 'Const', rho_1: float | None = None, rho_2: float | None = None, lmbd: float | None = 0.0, gamma: float | None = 0.0, use_autograd: bool = False, return_theta: bool = False) -> tuple[np.ndarray]:
        """
        Perform stochastic gradient descent to optimize the regression model parameters.
        Parameters
        ----------
        n_epochs : int
            Number of epochs to run the gradient descent.
        M : int
            Size of the mini-batches.
        eta : float, optional
            Learning rate. Default is None.
        tuning_method : str, optional
            Method for tuning the gradient descent. Options are 'Const', 'Momentum', 'Adagrad', 'AdagradMomentum', 'RMS_prop', 'Adam'. Default is 'Const'.
        rho_1 : float, optional
            Hyperparameter for RMS_prop and Adam methods. Default is None.
        rho_2 : float, optional
            Hyperparameter for Adam method. Default is None.
        lmbd : float, optional
            Regularization parameter. Default is 0.0.
        gamma : float, optional
            Momentum parameter. Must be between 0.0 and 1.0. Default is 0.0.
        use_autograd : bool, optional
            Whether to use autograd for gradient computation. Default is False.
        return_theta : bool, optional
            Whether to return the optimized theta parameters. Default is False.
        Returns
        -------
        tuple[np.ndarray]
            A tuple containing the Mean Squared Error (MSE), R-squared (R2) value, and optionally the optimized theta parameters.
        Raises
        ------
        ValueError
            If an invalid tuning method is provided.
        """
       
        np.random.seed(self.seed)
        if gamma < 0.0 or gamma > 1.0:
            print("The momentum parameter gamma must be between 0.0 and 1.0! Setting to 0.0.")
            gamma = 0.0

        if use_autograd:
            def cost_func(X, y, theta):
                return (1.0 / len(y)) * np.sum((X @ theta - y)**2) + lmbd * np.sum(theta**2) 
            training_gradient = grad(cost_func, 2)
        else:
            def training_gradient(X, y, theta):
                return 2.0 * ((1.0 / len(y)) * X.T @ (X @ theta - y) + lmbd * theta) 

        
        match tuning_method.lower():
            case "const":    # Uses a decaying learning rate
                gradient_change = Constant(eta)
            case "momentum":
                gradient_change = Momentum(eta, gamma)
            case "adagrad":
                gradient_change = Adagrad(eta)
            case "adagradmomentum":
                gradient_change = AdagradMomentum(eta, gamma)
            case "rms_prop":
                gradient_change = RMS_prop(eta, rho_1)
            case "adam":
                gradient_change = Adam(eta, rho_1, rho_2)
            case _:
                raise ValueError(f"Invalid tuning method: {tuning_method}.")
            
            
        # Estimating theta's with gradient descent
        m      = int(len(self.y_train) / M) 
        theta  = np.random.randn(self.X.shape[1], 1) 
        
        for s in range(1, n_epochs + 1):
            for i in range(m):
                k  = M*np.random.randint(m)
                Xi = self.X_train[k:k+M]
                yi = self.y_train[k:k+M]
                gradients = training_gradient(Xi, yi, theta)
                change = gradient_change.update_change(gradients)
                theta -= change
    
        y_pred = self.X_test @ theta
        
        MSE = self.get_MSE(y_pred)
        R2  = self.get_R2(y_pred)

        quantities = [MSE, R2]
        if return_theta:
            quantities.append(theta)
            
            
        gradient_change.reset()
        return quantities


    """ 
    Resampling methods 
    """
    def bootstrap(self, n: int, return_theta: bool = False) -> tuple[np.ndarray]:
        """
        Perform bootstrap resampling to estimate the mean squared error (MSE), bias, and variance of the model.
        Parameters
        ----------
        n : int
            Number of bootstrap samples.
        return_theta : bool, optional
            If True, the function also returns the mean of the estimated coefficients (theta). Default is False.
        Returns
        -------
        tuple[np.ndarray]
            A tuple containing the following elements:
            - MSE : float
                Mean squared error of the predictions.
            - bias : float
                Bias of the predictions.
            - variance : float
                Variance of the predictions.
            - theta : np.ndarray, optional
                Mean of the estimated coefficients, returned only if `return_theta` is True.
        """
        
        y_pred = np.empty((self.y_test.shape[0], n))
        theta = np.empty((self.X.shape[0], n)) 

        for i in range(n):
            X_, y_ = resample(self.X_train, self.y_train)
            
            theta[:, i] = np.linalg.pinv(X_.T @ X_) @ X_.T @ y_

            y_pred[:, i] = (self.X_test @ theta[:, i]).ravel()

        MSE  = np.mean(np.mean((self.y_test - y_pred)**2, axis = 1, keepdims = True))  
        bias = np.mean((self.y_test - np.mean(y_pred, axis = 1, keepdims = True))**2) 
                   
        variance = np.mean(np.var(y_pred, axis = 1, keepdims = True))  

        quantities = [MSE, bias, variance]
        if return_theta:
            theta = np.mean(theta, axis = 1, keepdims = True)
            quantities.append(theta)
        return quantities