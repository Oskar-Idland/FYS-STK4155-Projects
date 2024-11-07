import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import seaborn as sns
from tqdm import tqdm

def FrankeFunction(x: float | np.ndarray,y: float | np.ndarray, noise: float = 0, seed: int = 42) -> float | np.ndarray:
    """ 
    Generates a surface plot of the Franke function.
    # Parameters:
    x (float | np.ndarray): The x-value(s).
    y (float | np.ndarray): The y-value(s).
    noise (float): The standard deviation of the noise. Default is 0 (no noise).
    seed (int): The seed for the random number generator. Default is 42.
    """
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    
    np.random.seed(seed)
    noise = np.random.normal(0, noise, (x.shape[0], y.shape[0]))
    return term1 + term2 + term3 + term4 + noise

def MSE(pred: np.ndarray | float, targets: np.ndarray | float) -> float:
    """
    Calculate the Mean Squared Error (MSE) between predictions and targets.
    Parameters
    ----------
    pred : np.ndarray or float
        Predicted values.
    targets : np.ndarray or float
        Actual target values.
    Returns
    -------
    float
        The mean squared error between the predictions and targets.
    """
    return np.mean((pred - targets) ** 2)

def MSE_derivative(pred: np.ndarray | float, targets: np.ndarray | float) -> np.ndarray | float:
    """
    Compute the derivative of the Mean Squared Error (MSE) loss function.

    Parameters
    ----------
    pred : np.ndarray or float
        Predicted values.
    """
    return 2 * (pred - targets) / len(pred)

def R2(pred: np.ndarray | float, targets: np.ndarray | float) -> float:
    """
    Calculate the R-squared (coefficient of determination) regression score function.
    Parameters
    ----------
    pred : np.ndarray or float
        Predicted values.
    targets : np.ndarray or float
        True values.
    Returns
    -------
    float
        The R-squared score, which indicates the proportion of the variance in the dependent variable
        that is predictable from the independent variable(s). The best possible score is 1.0 and it can
        be negative (because the model can be arbitrarily worse).
    """
    return 1 - np.sum((targets - pred) ** 2) / np.sum((targets - np.mean(targets)) ** 2)


def create_X(x: np.ndarray, y: np.ndarray, n: int) -> np.ndarray:
    """
    Creates the design matrix X.

    ## Parameters:
    x (np.ndarray): Independent variable.
    y (np.ndarray): Independent variable.
    n (int): The degree of the polynomial features.

    ## Returns:
    np.ndarray: The design matrix X.
    """
    N = len(x)
    l = int((n + 1) * (n + 2) / 2)  # Number of elements in beta
    X = np.ones((N, l))

    for i in range(1, n + 1):
        q = int((i) * (i + 1) / 2)
        for k in range(i + 1):
            X[:, q + k] = (x ** (i - k)) * (y**k)
    return X

def grid_search(x_array: np.ndarray,  y_array: np.ndarray, search_params_name: tuple, method: str,  model,  stochastic: bool = False, n_iterations: int = 100, **kwargs):
    """
    Perform a grid search over specified parameters for a given model using gradient descent.
    Parameters
    ----------
    x_array : np.ndarray
        Array of values for the first parameter to search over.
    y_array : np.ndarray
        Array of values for the second parameter to search over.
    search_params_name : tuple
        Tuple containing the names of the parameters to search over. Example: ('eta', 'gamma').
    method : str
        The tuning method to be used in the gradient descent. Example: 'AdaGrad'.
    model : object
        The model object which contains the gradient descent methods.
    stochastic : bool, optional
        If True, use stochastic gradient descent. Otherwise, use regular gradient descent. Default is False.
    n_iterations : int, optional
        Number of iterations for the gradient descent. Default is 100.
    **kwargs
        Additional keyword arguments to pass to the gradient descent method.
    Returns
    -------
    MSE_matrix : np.ndarray
        Array of Mean Squared Error (MSE) values for the grid search.
    R2_matrix : np.ndarray
        Array of R-squared (R2) values for the grid search.
    """

    nx = len(x_array) ; ny = len(y_array)
    MSE_matrix = np.zeros((ny, nx))
    R2_matrix = np.zeros((ny, nx))
    
    if stochastic:
        descent = model.stochastic_gradient_descent
    else:
        descent = model.gradient_descent
    
    for i, y in enumerate(tqdm(y_array)):
        for j, x in enumerate(x_array):
            param_dict = {param_name: x if i == 0 else y for i, param_name in enumerate(search_params_name) if isinstance(param_name, str)}
            param_dict.update({param_name: y for _, param_name in enumerate(search_params_name) if isinstance(param_name, tuple) for param_name in param_name})
            MSE, R2 = descent(n_iterations, **param_dict , tuning_method=method, **kwargs)
            MSE_matrix[i, j] = MSE
            R2_matrix[i, j] = R2

    return  MSE_matrix, R2_matrix




def optimal_parameters(matrix: np.ndarray, x: np.ndarray, y: np.ndarray, max_or_min: str = 'min') -> tuple[np.ndarray, np.ndarray]:
    '''This function calculates optimal parameters based on a matrix and input arrays, with an option to
    minimize or maximize the result.
    
    Parameters
    ----------
    matrix : np.ndarray
        The `matrix` parameter is a NumPy array that represents the data or coefficients for a mathematical
    model.
    x : np.ndarray
        `x` the x-axis parameter values used to generate the matrix.
    y : np.ndarray
        `y` the y-axis parameter values used to generate the
    max_or_min : str, optional
        The `max_or_min` parameter is a string that determines whether to minimize or maximize the result. Options are 'min' or 'max' (default is 'min').
    
    '''
    if max_or_min == 'max':
        idx = np.where(matrix == np.nanmax(matrix))
    elif max_or_min == 'min':
        idx = np.where(matrix == np.nanmin(matrix))
    else:
        raise ValueError("max_or_min must be either 'max' or 'min'.")
    
    
    i = idx[0][0] ; j = idx[1][0]
    return x[j], y[i]


def plot_mse_contour(MSE_matrix: np.ndarray[float, float], x_array: np.ndarray, x_name: str, y_array: np.ndarray, y_name: str, log: bool = False, levels: bool = False, n_ticks: int | None = None, scatter: bool = False, show: bool = False) -> None | plt.Figure:
    """
    Plots a contour plot of the Mean Squared Error (MSE) matrix as a function of two parameters.
    Parameters
    ----------
    MSE_matrix : np.ndarray[float, float]
        A 2D array containing the MSE values.
    x_array : np.ndarray
        A 1D array containing the values of the first parameter.
    x_name : str
        The name of the first parameter.
    y_array : np.ndarray
        A 1D array containing the values of the second parameter.
    y_name : str
        The name of the second parameter.
    log : bool, optional
        If True, use a logarithmic scale for the color normalization. Default is False.
    levels : bool, optional
        If True, use specific levels for the contour plot. Default is False.
    n_ticks : int or None, optional
        Number of ticks to use on the x and y axes. If None, default ticks are used. Default is None.
    scatter : bool, optional
        If True, scatter plot the optimal parameters on the contour plot. Default is False.
    show : bool, optional
        If True, display the plot. If False, return the figure object. Default is False.
    Returns
    -------
    None or plt.Figure
        If show is False, returns the figure object. Otherwise, displays the plot and returns None.
    """
    if levels:
        MSE_max, MSE_min = np.max(MSE_matrix), np.min(MSE_matrix)
        ny, nx = MSE_matrix.shape
        levels = np.linspace(MSE_min, MSE_max, nx*ny)
    else:
        levels = None
        
    if log:
        norm = mcolors.LogNorm()
    else:
        norm = None
    
    fig = plt.figure()
    plt.contourf(x_array, y_array, MSE_matrix, levels=None, cmap='viridis', extend='both', norm=norm)
    plt.colorbar(label='MSE', format='%.0e')
        
    
    
    if n_ticks:
        x_min = np.min(x_array) ; x_max = np.max(x_array) 
        y_min = np.min(y_array) ; y_max = np.max(y_array)
        plt.xticks(np.linspace(x_min, x_max, n_ticks))
        plt.yticks(np.linspace(y_min, y_max, n_ticks))
        
    if scatter:
        x_optimal, y_optimal = optimal_parameters(MSE_matrix, x_array, y_array)
        plt.scatter(x_optimal, y_optimal, color='r', label='Optimal parameters', marker='x')
        plt.legend()
        
            
    plt.title(f'MSE as a function of {x_name} and {y_name}')
    plt.xlabel(f'{x_name}')
    plt.ylabel(f'{y_name}')
    
    if show:
        plt.show()
    else:
        return fig 
    
def gridsearch_plot(scores, ticks, cmap = "viridis", opt_search = None, opt_color = "red", fmt1 = ".2f", fmt2 = ".2f"):
    """
    Plots the results of a grid search as heatmaps.
    Parameters
    ----------
    scores : tuple of np.ndarray
        A tuple containing two 2D arrays of scores to be plotted.
    ticks : tuple of list
        A tuple containing two lists of tick labels for the x and y axes.
    cmap : str, optional
        The colormap to be used for the heatmaps (default is "viridis").
    opt_search : tuple of str, optional
        A tuple containing two strings indicating whether to highlight the 
        maximum ("max") or minimum ("min") score in each heatmap (default is None).
    opt_color : str, optional
        The color to be used for highlighting the optimal values (default is "red").
    fmt1 : str, optional
        The format string for annotations in the first heatmap (default is ".2f").
    fmt2 : str, optional
        The format string for annotations in the second heatmap (default is ".2f").
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the heatmaps.
    axs : np.ndarray of matplotlib.axes._subplots.AxesSubplot
        An array of the axes objec ts for the heatmaps.
    """

    score1, score2 = scores
    ticks1, ticks2 = ticks

    fig, axs = plt.subplots(1, 2, figsize=(13, 5))

    sns.heatmap(score1, 
                cmap=cmap,
                xticklabels=ticks1,
                yticklabels=ticks2, 
                annot=True,
                ax=axs[0],
                fmt=fmt1,
                cbar=False,
            )

    sns.heatmap(score2,
                cmap=cmap,
                xticklabels=ticks1,
                yticklabels=ticks2, 
                annot=True,
                ax=axs[1],
                fmt=fmt2,
                cbar=False,
            )

    # Highlight optimal values if requested
    if opt_search is not None:
        # First plot
        if opt_search[0] == "max":
            score1_idx = np.unravel_index(np.argmax(score1), score1.shape)
        else:
            score1_idx = np.unravel_index(np.argmin(score1), score1.shape)
        rect1 = plt.Rectangle((score1_idx[1], score1_idx[0]), 1, 1, 
                            fill=False, edgecolor=opt_color, lw=2)
        axs[0].add_patch(rect1)

        # Second plot
        if opt_search[1] == "max":
            score2_idx = np.unravel_index(np.argmax(score2), score2.shape)
        else:
            score2_idx = np.unravel_index(np.argmin(score2), score2.shape)
        rect2 = plt.Rectangle((score2_idx[1], score2_idx[0]), 1, 1, 
                            fill=False, edgecolor=opt_color, lw=2)
        axs[1].add_patch(rect2)

    return fig, axs

    
def parameter_print_plot(MSE_matrix: np.ndarray[float, float], R2_matrix: np.ndarray[float, float], x_array: np.ndarray, y_array: np.ndarray, x_label: str, y_label: str, log: bool = True, levels: bool = False, n_ticks: int | None = None, scatter: bool = False, show: bool = False):
    """
    Prints the optimal parameters and plots the MSE contour.
    Parameters
    ----------
    MSE_matrix : np.ndarray[float, float]
        Matrix of Mean Squared Errors.
    R2_matrix : np.ndarray[float, float]
        Matrix of R-squared values.
    x_array : np.ndarray
        Array of x-axis values.
    y_array : np.ndarray
        Array of y-axis values.
    x_label : str
        Label for the x-axis.
    y_label : str
        Label for the y-axis.
    log : bool, optional
        Whether to use a logarithmic scale for the plot (default is True).
    levels : bool, optional
        Whether to use levels in the contour plot (default is False).
    n_ticks : int or None, optional
        Number of ticks on the axes (default is None).
    scatter : bool, optional
        Whether to scatter plot the optimal points (default is False).
    show : bool, optional
        Whether to show the plot (default is False).
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the plot.
    """
    

    x_optimal, y_optimal = optimal_parameters(MSE_matrix, x_array, y_array)
    
    greek_dict = {r'$\eta$': 'η', r'$\gamma$': 'γ', r'$\rho$': 'ρ'}
    if x_label in greek_dict:
        x_label_greek = greek_dict[x_label]
    else:
        x_label_greek = x_label
    if y_label in greek_dict:
        y_label_greek = greek_dict[y_label]
    else:
        y_label_greek = y_label
    
    
        
    print(f'Minimum MSE: {np.nanmin(MSE_matrix):>16.3e}')
    print(f'Optimal {x_label_greek} for MSE: {x_optimal: 3.3e}')
    print(f'Optimal {y_label_greek} for MSE: {y_optimal: 3.3e}')
    
    print()
    
    x_optimal, y_optimal = optimal_parameters(R2_matrix, x_array, y_array, max_or_min='max')
    print(f'Maximum R2: {np.nanmax(R2_matrix):>16.2%}')
    print(f'Optimal {x_label_greek} for R2: {x_optimal: 3.3e}')
    print(f'Optimal {y_label_greek} for R2: {y_optimal: 3.3e}')
    
    fig = plot_mse_contour(MSE_matrix, x_array, x_label, y_array, y_label,log=log, levels=levels, n_ticks=n_ticks, scatter=scatter, show=show)
    return fig

