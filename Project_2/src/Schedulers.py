import numpy as np

class Scheduler:
    """
    Base class for learning rate schedulers.

    Parameters
    ----------
    eta : float
        The constant learning rate.

    Methods
    -------
    update_change(gradient)
        Computes the change to be applied to the parameters based on the gradient.
    reset()
        Resets the scheduler state.
    """

    def __init__(self, eta: float):
        self.eta = eta

    def update_change(self, gradient: np.ndarray) -> np.ndarray:
        """
        Computes the change to be applied to the parameters based on the gradient.

        Parameters
        ----------
        gradient : np.ndarray
            The gradient of the loss function with respect to the parameters.

        Returns
        -------
        np.ndarray
            The change to be applied to the parameters.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def reset(self):
        """
        Resets the scheduler state.
        """
        pass


class Constant(Scheduler):
    """
    Constant learning rate scheduler.

    Parameters
    ----------
    eta : float
        The constant learning rate.

    Methods
    -------
    update_change(gradient)
        Computes the change to be applied to the parameters based on the gradient.
    reset()
        Resets the scheduler state (no-op for constant scheduler).
    """

    def __init__(self, eta: float):
        super().__init__(eta)

    def update_change(self, gradient: np.ndarray) -> np.ndarray:
        """
        Computes the change to be applied to the parameters based on the gradient.

        Parameters
        ----------
        gradient : np.ndarray
            The gradient of the loss function with respect to the parameters.

        Returns
        -------
        np.ndarray
            The change to be applied to the parameters.
        """
        return self.eta * gradient
    
    def reset(self):
        """
        Resets the scheduler state (no-op for constant scheduler).
        """
        pass


class Momentum(Scheduler):
    """
    Momentum-based learning rate scheduler.

    Parameters
    ----------
    eta : float
        The learning rate.
    momentum : float
        The momentum factor.

    Methods
    -------
    update_change(gradient)
        Computes the change to be applied to the parameters based on the gradient.
    reset()
        Resets the scheduler state.
    """

    def __init__(self, eta: float, momentum: float):
        super().__init__(eta)
        self.momentum = momentum
        self.change = 0

    def update_change(self, gradient: np.ndarray) -> np.ndarray:
        """
        Computes the change to be applied to the parameters based on the gradient.

        Parameters
        ----------
        gradient : np.ndarray
            The gradient of the loss function with respect to the parameters.

        Returns
        -------
        np.ndarray
            The change to be applied to the parameters.
        """
        self.change = self.momentum * self.change + self.eta * gradient
        return self.change

    def reset(self):
        """
        Resets the scheduler state.
        """
        self.change = 0


class Adagrad(Scheduler):
    """
    Adagrad learning rate scheduler.

    Parameters
    ----------
    eta : float
        The learning rate.

    Methods
    -------
    update_change(gradient)
        Computes the change to be applied to the parameters based on the gradient.
    reset()
        Resets the scheduler state.
    """

    def __init__(self, eta: float):
        super().__init__(eta)
        self.G_t = None

    def update_change(self, gradient: np.ndarray) -> np.ndarray:
        """
        Computes the change to be applied to the parameters based on the gradient.

        Parameters
        ----------
        gradient : np.ndarray
            The gradient of the loss function with respect to the parameters.

        Returns
        -------
        np.ndarray
            The change to be applied to the parameters.
        """
        delta = 1e-8  # avoid division by zero

        if self.G_t is None:
            self.G_t = np.zeros_like(gradient)

        self.G_t += gradient * gradient

        return self.eta * gradient / (np.sqrt(self.G_t + delta))

    def reset(self):
        """
        Resets the scheduler state.
        """
        self.G_t = None


class AdagradMomentum(Scheduler):
    """
    Adagrad with Momentum learning rate scheduler.

    Parameters
    ----------
    eta : float
        The learning rate.
    momentum : float
        The momentum factor.

    Methods
    -------
    update_change(gradient)
        Computes the change to be applied to the parameters based on the gradient.
    reset()
        Resets the scheduler state.
    """

    def __init__(self, eta: float, momentum: float):
        super().__init__(eta)
        self.G_t = None
        self.momentum = momentum
        self.change = 0

    def update_change(self, gradient: np.ndarray) -> np.ndarray:
        """
        Computes the change to be applied to the parameters based on the gradient.

        Parameters
        ----------
        gradient : np.ndarray
            The gradient of the loss function with respect to the parameters.

        Returns
        -------
        np.ndarray
            The change to be applied to the parameters.
        """
        delta = 1e-8  # avoid division ny zero

        if self.G_t is None:
            self.G_t = np.zeros((gradient.shape[0], gradient.shape[0]))

        self.G_t += gradient @ gradient.T

        G_t_inverse = 1 / (
            delta + np.sqrt(np.reshape(np.diagonal(self.G_t), (self.G_t.shape[0], 1)))
        )
        self.change = self.change * self.momentum + self.eta * gradient * G_t_inverse
        return self.change

    def reset(self):
        """
        Resets the scheduler state.
        """
        self.G_t = None


class RMS_prop(Scheduler):
    """
    RMSprop learning rate scheduler.

    Parameters
    ----------
    eta : float
        The learning rate.
    rho : float
        The decay rate for the moving average of squared gradients.

    Methods
    -------
    update_change(gradient)
        Computes the change to be applied to the parameters based on the gradient.
    reset()
        Resets the scheduler state.
    """

    def __init__(self, eta: float, rho: float):
        super().__init__(eta)
        self.rho = rho
        self.second = 0.0

    def update_change(self, gradient: np.ndarray) -> np.ndarray:
        """
        Computes the change to be applied to the parameters based on the gradient.

        Parameters
        ----------
        gradient : np.ndarray
            The gradient of the loss function with respect to the parameters.

        Returns
        -------
        np.ndarray
            The change to be applied to the parameters.
        """
        delta = 1e-8  # avoid division ny zero
        self.second = self.rho * self.second + (1 - self.rho) * gradient * gradient
        return self.eta * gradient / (np.sqrt(self.second + delta))

    def reset(self):
        """
        Resets the scheduler state.
        """
        self.second = 0.0


class Adam(Scheduler):
    """
    Adam (Adaptive Moment Estimation) learning rate scheduler.

    Parameters
    ----------
    eta : float
        The learning rate.
    rho : float
        The exponential decay rate for the first moment estimates.
    rho2 : float
        The exponential decay rate for the second moment estimates.

    Methods
    -------
    update_change(gradient)
        Computes the change to be applied to the parameters based on the gradient.
    reset()
        Resets the scheduler state.
    """

    def __init__(self, eta: float, rho: float, rho2: float):
        super().__init__(eta)
        self.rho = rho
        self.rho2 = rho2
        self.moment = 0
        self.second = 0
        self.n_epochs = 1

    def update_change(self, gradient: np.ndarray) -> np.ndarray:
        """
        Computes the change to be applied to the parameters based on the gradient.

        Parameters
        ----------
        gradient : np.ndarray
            The gradient of the loss function with respect to the parameters.

        Returns
        -------
        np.ndarray
            The change to be applied to the parameters.
        """
        delta = 1e-8  # avoid division ny zero
        self.moment = self.rho * self.moment + (1 - self.rho) * gradient
        self.second = self.rho2 * self.second + (1 - self.rho2) * gradient * gradient

        moment_hat = self.moment / (1 - self.rho ** self.n_epochs)
        second_hat = self.second / (1 - self.rho2 ** self.n_epochs)

        self.n_epochs += 1

        return self.eta * moment_hat / (np.sqrt(second_hat) + delta)

    def reset(self):
        """
        Resets the scheduler state.
        """
        self.moment = 0
        self.second = 0
        self.n_epochs = 1