import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import seed
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import warnings
warnings.filterwarnings("ignore")

from activation_funcs import *
from cost_funcs import *
from Schedulers import *
from FFNN import FFNN

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
    if len(x.shape) > 1:
        x = np.ravel(x)
        y = np.ravel(y)

    N = int(len(x)*len(y))            # Number of rows in the design matrix
    l = int((n+1)*(n+2)/2)            # Number of columns in the design matrix
    X = np.ones((N, l))
    
    x_mesh, y_mesh = np.meshgrid(x, y)        # Make a meshgrid to get all possible combinations of x and y values
    x_mesh = x_mesh.flatten()
    y_mesh = y_mesh.flatten()

    idx = 1
    for i in range(1, n+1):
        for j in range(i+1):
            X[:, idx] = x_mesh**(i-j) * y_mesh**j
            idx += 1

    return X

def FrankeFunction(x: float | np.ndarray,y: float | np.ndarray) -> float | np.ndarray:
    """ 
    Generates a surface plot of the Franke function.
    # Parameters:
    x (float | np.ndarray): The x-value(s).
    y (float | np.ndarray): The y-value(s).
    """
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

if __name__ == "__main__":
    '''
    --------
    Example use of FFNN using Franke data
    -------- 
    '''
    N  = 1000
    x = np.arange(0, 1, 1/N)
    y = np.arange(0, 1, 1/N)
    X, Y = np.meshgrid(x, y)

    z = (FrankeFunction(X, Y) + np.random.normal(0, 0.1, (N, N))).reshape(-1, 1) 

    poly_degree = 3
    design_matrix = create_X(x, y, poly_degree)

    X_train, X_test, t_train, t_test = train_test_split(design_matrix, z, test_size=0.2)

    input_nodes = X_train.shape[1]
    output_nodes = 1

    linear_regression = FFNN(
        (input_nodes, output_nodes), 
        output_func=identity, 
        cost_func=CostOLS,
        seed=42069
        )

    linear_regression.reset_weights() # reset weitght so that previous training does not affect the new training

    scheduler = Constant(eta=1e-3)
    scores = linear_regression.fit(X_train, t_train, scheduler)

    print(f"Training score: {scores}")