import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import seed
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
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

if __name__ == "__main__":
    '''
    --------
    Example use of FFNN using Franke data
    -------- 
    '''
    N  = 10
    x = np.arange(0, 1, 1/N)
    y = np.arange(0, 1, 1/N)
    X, Y = np.meshgrid(x, y)

    z = (FrankeFunction(X, Y) + np.random.normal(0, 0.1, (N, N))).reshape(-1, 1) 

    poly_degree = 3
    design_matrix = create_X(x, y, poly_degree)

    X_train, X_test, t_train, t_test = train_test_split(design_matrix, z, test_size=0.2)

    input_nodes = X_train.shape[1]
    hidden_nodes = 2
    output_nodes = 1

    dims = [input_nodes, hidden_nodes, output_nodes]

    linear_regression = FFNN(
        dims, 
        output_func=identity, 
        cost_func=CostOLS,
        seed=42069
        )

    # linear_regression.reset_weights() # reset weitght so that previous training does not affect the new training

    # scheduler = Adam(eta=1e-4, rho=0.9, rho2=0.999)

    # scores = linear_regression.fit(X_train, t_train, scheduler, X_val=X_test, t_val=t_test, epochs=1000)

    import seaborn as sns

    sns.set_theme()

    eta_vals = np.logspace(-5, 1, 3)
    lmbd_vals = np.logspace(-5, 1, 3)
    # store the models for later use
    # DNN_numpy = np.zeros((len(eta_vals), len(lmbd_vals)), dtype=object)
    # DNN_MSE = np.zeros((len(eta_vals), len(lmbd_vals), 1000))
    # DNN_R2 = np.zeros((len(eta_vals), len(lmbd_vals), 1000))

    # # grid search
    # for i, eta in enumerate(eta_vals):
    #     for j, lmbd in enumerate(lmbd_vals):
    #         scheduler = Adagrad(eta)

    #         DNN_numpy[i][j] = linear_regression.fit(X_train, t_train, scheduler, lambd=lmbd , epochs=1000)   
    #         print(DNN_numpy[i][j]["train_errors"].shape)
    #         DNN_MSE[i][j] = DNN_numpy[i][j]["train_errors"]
    #         DNN_R2[i][j] = DNN_numpy[i][j]["R2_scores"]

    # extent = [np.min(lmbd_vals), np.max(lmbd_vals), np.min(eta_vals), np.max(eta_vals)]

    # fig, axs = plt.subplots(1, 2, figsize = (13, 5))
    # im1 = axs[0].imshow(DNN_MSE.T[::-1], cmap = "YlGn", aspect = "auto", extent = extent)
    # plt.colorbar(im1, ax = axs[0], pad = 0.02, aspect = 10)
    # im2 = axs[1].imshow(DNN_R2.T[::-1], cmap = "YlGn", aspect = "auto", extent = extent)
    # plt.colorbar(im2, ax = axs[1], pad = 0.02, aspect = 10)
    # axs[0].set_xticks(lmbd_vals[::2])
    # axs[1].set_xticks(lmbd_vals[::2])
    # axs[0].set_title("MSE")
    # axs[1].set_title("R2")
    # fig.supylabel("Eta")
    # fig.supxlabel("Lambda")
    # plt.tight_layout
    # # plt.savefig("../figs/f_kfold_vs_bootstrap.pdf")
    # plt.show()

    # DNN_scikit = np.zeros((len(eta_vals), len(lmbd_vals)), dtype=object)
    # DNN_MSE = np.zeros((len(eta_vals), len(lmbd_vals), 1000))
    # DNN_R2 = np.zeros((len(eta_vals), len(lmbd_vals), 1000))

    # for i, eta in enumerate(eta_vals):
    #     for j, lmbd in enumerate(lmbd_vals):
    #         scheduler = Adagrad(eta)
    #         dnn = MLPRegressor(hidden_layer_sizes=(2), activation='logistic',
    #                             alpha=lmbd, learning_rate_init=eta, max_iter=100)
    #         DNN_scikit[i][j] = linear_regression.fit(X_train, t_train, scheduler, lambd=lmbd , epochs=1000)   
    #         DNN_MSE[i][j] = DNN_scikit[i][j]["train_errors"]
    #         DNN_R2[i][j] = DNN_scikit[i][j]["R2_scores"]


    # extent = [np.min(lmbd_vals), np.max(lmbd_vals), np.min(eta_vals), np.max(eta_vals)]   
    # fig, axs = plt.subplots(1, 2, figsize = (13, 5))
    # im1 = axs[0].imshow(DNN_MSE.T[::-1], cmap = "YlGn", aspect = "auto", extent = extent)
    # plt.colorbar(im1, ax = axs[0], pad = 0.02, aspect = 10)
    # im2 = axs[1].imshow(DNN_R2.T[::-1], cmap = "YlGn", aspect = "auto", extent = extent)
    # plt.colorbar(im2, ax = axs[1], pad = 0.02, aspect = 10)
    # axs[0].set_xticks(lmbd_vals[::2])
    # axs[1].set_xticks(lmbd_vals[::2])
    # axs[0].set_title("MSE")
    # axs[1].set_title("R2")
    # fig.supylabel("Eta")
    # fig.supxlabel("Lambda")
    # plt.tight_layout
    # # plt.savefig("../figs/f_kfold_vs_bootstrap.pdf")
    # plt.show()
