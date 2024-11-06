import numpy as np
import matplotlib.pyplot as plt
# TODO: Add docstrings to the functions below

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
    The function calculates the mean squared error between predicted and target values.
    
    Args:
      pred (np.ndarray | float): The `pred` parameter represents the predicted values, which can be
    either a NumPy array or a single float value.
      targets (np.ndarray | float): The `targets` parameter in the `MSE` function represents the actual
    values or ground truth values that you are trying to predict or estimate. These are the values that
    your model is attempting to approximate or match with its predictions.
    """
    return np.mean((pred - targets) ** 2)

def MSE_derivative(pred: np.ndarray | float, targets: np.ndarray | float) -> np.ndarray | float:
    return 2 * (pred - targets) / len(pred)

def R2(pred: np.ndarray | float, targets: np.ndarray | float) -> float:    
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

def optimal_parameters(matrix: np.ndarray, x: np.ndarray, y: np.ndarray, max_or_min: str = 'min') -> tuple[np.ndarray, np.ndarray]:
    """
    This function calculates optimal parameters based on a matrix and input arrays, with an option to
    minimize or maximize the result.
    
    Args:
      matrix (np.ndarray): The `matrix` parameter is a NumPy array that represents the data or
    coefficients for a mathematical model.
      x (np.ndarray): `x` is an array containing the input values for your model. It is typically used
    as the independent variable in a regression or optimization problem.
      y (np.ndarray): The `optimal_parameters` function seems to be missing some important information
    about the parameters. Could you please provide more details about the `y` parameter so that I can
    assist you better?
      max_or_min (str): The `max_or_min` parameter specifies whether we are looking to maximize or
    minimize a certain value. In this case, it can take on the values 'max' or 'min' to indicate whether
    we want to maximize or minimize the objective function. Defaults to min
    """
    
    
    
    
    '''This function calculates optimal parameters based on a matrix and input arrays, with an option to
    minimize or maximize the result.
    
    Parameters
    ----------
    matrix : np.ndarray
        The `matrix` parameter is a NumPy array that represents the data or coefficients for a mathematical
    model.
    x : np.ndarray
        `x` is an array containing the input values for your model. It is used as one of the inputs to
    calculate the optimal parameters.
    y : np.ndarray
        The `optimal_parameters` function seems to be missing some important information about the
    parameters `y` and `x`. Could you please provide more details about what these parameters represent
    or how they are used in the function? This will help me provide a more accurate explanation or
    assistance.
    max_or_min : str, optional
        The `max_or_min` parameter specifies whether you want to maximize or minimize a certain value. In
    this case, it is a string parameter that can take the values 'max' or 'min'. This parameter helps
    determine the direction of optimization when finding the optimal parameters based on the input
    matrix and arrays
    
    '''
    """
    Finds the indices of the minimum value in a matrix.

    ## Parameters:
    matrix (np.ndarray): The matrix to search.
    x (np.ndarray): The x-values.
    y (np.ndarray): The y-values.
    max_or_min (str ['min' | 'max']): Whether to find the maximum or minimum value. Default is 'min'.

    ## Returns:
    tuple[np.ndarray, np.ndarray]: The indices of the minimum value.
    """
    if max_or_min == 'max':
        idx = np.unravel_index(np.nanargmax(matrix), matrix.shape, )
    elif max_or_min == 'min':
        idx = np.unravel_index(np.nanargmin(matrix), matrix.shape)
    else:
        raise ValueError("max_or_min must be either 'max' or 'min'.")
    
    return x[idx[1]], y[idx[0]]


def plot_mse_contour(MSE_matrix: np.ndarray[float, float], x_array: np.ndarray, x_name: str, y_array: np.ndarray, y_name: str, n_ticks: int | None = None, scatter: bool = False, show: bool = False) -> None | plt.Figure:
    '''This function plots a contour plot of Mean Squared Error (MSE) values based on a given MSE matrix
    and corresponding x and y arrays, with options to customize axis labels, tick marks, and scatter
    points.
    
    Parameters
    ----------
    MSE_matrix : np.ndarray[float, float]
        An array containing Mean Squared Error (MSE) values for different combinations of x and y
    parameters.
    x_array : np.ndarray
        x_array is an array containing the values for the x-axis in the contour plot.
    x_name : str
        The `x_name` parameter is a string that represents the name of the x-axis in the plot. It is used
    to label the x-axis with a descriptive name that indicates what the values on the x-axis represent.
    y_array : np.ndarray
        `y_array` is an array containing the values for the y-axis in the contour plot. It is used to
    generate the contour plot based on the Mean Squared Error (MSE) values provided in the `MSE_matrix`.
    y_name : str
        The `y_name` parameter is a string that represents the name of the y-axis in the plot. It is used
    to label the y-axis with a descriptive name that helps users understand the data being displayed.
    n_ticks : int | None
        The `n_ticks` parameter in the `plot_mse_contour` function is used to specify the number of ticks
    on the contour plot axes. If `n_ticks` is set to `None`, the plotting function will determine the
    number of ticks automatically based on the data range. If a specific
    scatter : bool, optional
        The `scatter` parameter in the `plot_mse_contour` function is a boolean flag that determines
    whether to plot the MSE values as a scatter plot on top of the contour plot. If `scatter` is set to
    `True`, the function will overlay a scatter plot of the MSE values on
    
    '''

    nx, ny = MSE_matrix.shape
    MSE_max, MSE_min = np.max(MSE_matrix), np.min(MSE_matrix)
    levels = np.linspace(MSE_min, MSE_max, nx*ny)
    
    fig = plt.figure()
    plt.contourf(x_array, y_array, MSE_matrix, cmap='viridis', extend='both')
    plt.colorbar(label='MSE', format='%.0e')
    plt.contourf(x_array, y_array, MSE_matrix, levels=levels, cmap='viridis')
    
    x_optimal, y_optimal = optimal_parameters(MSE_matrix, x_array, y_array)
    
    if n_ticks:
        x_min = np.min(x_array) ; x_max = np.max(x_array) 
        y_min = np.min(y_array) ; y_max = np.max(y_array)
        plt.xticks(np.linspace(x_min, x_max, n_ticks))
        plt.yticks(np.linspace(y_min, y_max, n_ticks))
        
    if scatter:
        plt.scatter(x_optimal, y_optimal, color='r', label='Optimal parameters', marker='x')
        plt.legend()
        
            
    plt.title(f'MSE as a function of {x_name} and {y_name}')
    plt.xlabel(f'{x_name}')
    plt.ylabel(f'{y_name}')
    
    if show:
        plt.show()
    else:
        return fig
    
    
def parameter_print_plot(MSE_array, R2_array, x_array, y_array, x_label, y_label, n_ticks: int | None = None, scatter: bool = False, show: bool = False):
    x_optimal, y_optimal = optimal_parameters(MSE_array, x_array, y_array)
    
    greek_dict = {r'$\eta$': 'η', r'$\gamma$': 'γ', r'$\rho$': 'ρ'}
    if x_label in greek_dict:
        x_label_greek = greek_dict[x_label]
    else:
        x_label_greek = x_label
    if y_label in greek_dict:
        y_label_greek = greek_dict[y_label]
    else:
        y_label_greek = y_label
    
    
        
    x_label_greek = greek_dict[x_label]
    y_label_greek = greek_dict[y_label]
    print(f'Minimum MSE: {np.nanmin(MSE_array):>16.3e}')
    print(f'Optimal {x_label_greek} for MSE: {x_optimal: 3.3e}')
    print(f'Optimal {y_label_greek} for MSE: {y_optimal: 3.3e}')
    
    print()
    
    x_optimal, y_optimal = optimal_parameters(R2_array, x_array, y_array, max_or_min='max')
    print(f'Maximum R2: {np.nanmax(R2_array):>16.2%}')
    print(f'Optimal {x_label_greek} for R2: {x_optimal: 3.3e}')
    print(f'Optimal {y_label_greek} for R2: {y_optimal: 3.3e}')
    
    fig = plot_mse_contour(MSE_array, x_array, x_label, y_array, y_label, n_ticks=n_ticks, scatter=scatter, show=show)
    return fig