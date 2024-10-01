import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.utils import resample
plt.rcParams.update({'text.usetex': True, 'font.size': 16, 'font.family': 'serif', 'font.serif': 'Computer Modern Sans Serif', 'font.weight': 100, 'mathtext.fontset': 'cm', 'xtick.labelsize': 14, 'ytick.labelsize': 14})






def FrankeFunction(x,y):
    """ 
    Generates a surface plot of the Franke function.
    """
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

def MSE(y, y_tilde):
    """
    Calculates the mean squared error between the predicted and true values.
    """
    n = len(y)
    return np.sum((y - y_tilde)**2) / n

def R2(y_data, y_model):
    """
    Calculates the R2 score of the model.
    """
    return 1 - np.sum((y_data - y_model)**2) / np.sum((y_data - np.mean(y_data))**2)

def create_X(x, y, n):
	"""
    Creates the design matrix X.
    """
	if len(x.shape) > 1:
		x = np.ravel(x)
		y = np.ravel(y)

	N = len(x)
	l = int((n+1)*(n+2)/2)		# Number of elements in beta
	X = np.ones((N,l))

	for i in range(1,n+1):
		q = int((i)*(i+1)/2)
		for k in range(i+1):
			X[:,q+k] = (x**(i-k))*(y**k)

	return X

def kfold_crossval(x, y, z, k, model, degrees):
    """
    Performs k-fold cross-validation.
    """
    kfold = KFold(n_splits=k)
    estimated_mse = np.zeros(len(degrees))

    for i, degree in enumerate(degrees):
        X = create_X(x, y, degree)
        estimated_mse_folds = cross_val_score(model, X, z, scoring='neg_mean_squared_error', cv=kfold)
        estimated_mse[i] = np.mean(-estimated_mse_folds)

    return estimated_mse

if __name__ == "__main__":
      # seed = np.random.randint(1,1000)
    seed = 42
    np.random.seed(seed)
    N = 100

    x = np.arange(0, 1, 1/N)
    y = np.arange(0, 1, 1/N)

    z = FrankeFunction(x, y)

    degrees = np.arange(1, 6) 
    k = 5

    """kfold OLS"""    
    OLS = LinearRegression(fit_intercept=False)
    OLS_mse_kfold = kfold_crossval(x, y, z, k, OLS, degrees)
  
    """kfold Ridge"""
    lambdas = np.logspace(-4, 4, 6)
    ridge_mse_kfold = np.zeros((len(lambdas), len(degrees)))

    for i, lmb in enumerate(lambdas):
        ridge = linear_model.Ridge(alpha=lmb, fit_intercept=False)
        ridge_mse_kfold[i] = kfold_crossval(x, y, z, k, ridge, degrees)

    print(OLS_mse_kfold.shape, ridge_mse_kfold.shape)
