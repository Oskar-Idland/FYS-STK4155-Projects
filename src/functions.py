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
	l = int((n+1)*(n+2)/2) # Number of elements in beta
	X = np.ones((N,l))

	for i in range(1,n+1):
		q = int((i)*(i+1)/2)
		for k in range(i+1):
			X[:,q+k] = (x**(i-k))*(y**k)

	return X


def OLS(x, y, z, degrees, scale=True, test_size=0.2, seed=None, intercept=False):
    """
    Performs OLS regression. Returns MSE and R2 score and beta values. 
    """

    MSE_list = []
    R2_list = []
    β_list = []

    for degree in degrees:
        X = create_X(x, y, degree)
        X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=test_size, random_state=seed)

        if scale: # Scaling the data
            scaler_X = StandardScaler().fit(X_train)
            X_train = scaler_X.transform(X_train)
            X_test = scaler_X.transform(X_test)

            scaler_z = StandardScaler().fit(z_train)
            z_train = scaler_z.transform(z_train)
            z_test= scaler_z.transform(z_test)


        β = np.linalg.pinv(X_train.T @ X_train) @ X_train.T @ z_train

        if intercept:
            β[0] = 0

        β_list.append(β)
        
        z_tilde = X_train @ β 
        z_pred = X_test @ β
        
        MSE_list.append(MSE(z_train, z_tilde))
        R2_list.append(R2(z_test, z_pred))

    return MSE_list, R2_list, β_list


def Ridge(x, y, z, degrees, λ, scale=True, test_size=0.2, seed=None, intercept=False):
    """Performs Ridge regression. Returns MSE and R2 score and beta values"""
    MSE_list = np.zeros(len(degrees))
    R2_list = np.zeros(len(degrees))
    β_list = []

    for i, degree in enumerate(degrees):
        X = create_X(x, y, degree)
        X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=test_size, random_state=seed)

        if scale: # Scaling the data
            scaler_X = StandardScaler().fit(X_train)
            X_train = scaler_X.transform(X_train)
            X_test = scaler_X.transform(X_test)

            scaler_z = StandardScaler().fit(z_train)
            z_train = scaler_z.transform(z_train)
            z_test= scaler_z.transform(z_test)

        β = np.linalg.pinv(X_train.T @ X_train + λ*np.eye(X_train.shape[1])) @ X_train.T @ z_train

        if intercept:
            β[0] = 0

        β_list.append(β)

        z_tilde = X_train @ β
        z_pred = X_test @ β

        MSE_list[i] = MSE(z_train, z_tilde)
        R2_list[i] = R2(z_test, z_pred)

def Lasso(x, y, z, degrees, λ, scale=True, test_size=0.2, seed=None, intercept=False):
    """Performs Lasso regression. Returns MSE and R2 score and beta values"""
    MSE_list = np.zeros(len(degrees))
    R2_list = np.zeros(len(degrees))
    β_list = []

    for i, degree in enumerate(degrees):
        X = create_X(x, y, degree)
        X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=test_size, random_state=seed)

        if scale: # Scaling the data
            scaler_X = StandardScaler().fit(X_train)
            X_train = scaler_X.transform(X_train)
            X_test = scaler_X.transform(X_test)

            scaler_z = StandardScaler().fit(z_train)
            z_train = scaler_z.transform(z_train)
            z_test= scaler_z.transform(z_test)

        RegLasso = linear_model.Lasso(λ, fit_intercept=True)
        RegLasso.fit(X_train, z_train)

        z_tilde = RegLasso.predict(X_train)
        z_pred = RegLasso.predict(X_test)

        MSE_list[i] = MSE(z_train, z_tilde)
        R2_list[i] = R2(z_test, z_pred)


    # print(MSE_list.shape, R2_list.shape, β.shape)

    return MSE_list, R2_list, β_list

def Kfold_crossval(x, y, z, k, model, degrees):
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
    X, Y = np.meshgrid(x, y)

    z = FrankeFunction(X, Y)

    """"Task a"""
    degrees = np.arange(1, 6)
    MSE_list, R2_list, β_list = OLS(x, y, z, degrees, seed=seed)

    """Task b"""
    degrees = np.arange(1, 6)
    lambdas = np.logspace(-4, 4, 6)

    MSE_list = np.zeros((len(lambdas), len(degrees)))
    R2_list = np.zeros((len(lambdas), len(degrees)))
    β_list = [[] for i in range(len(lambdas))]


    for i, lmb in enumerate(lambdas):
        MSE_list[i], R2_list[i], β_list[i] = Ridge(x, y, z, degrees, lmb, seed=seed)

    """Task f"""
    degrees = np.arange(1, 6) 
    k = 5

    # kfold OLS  
    OLS = LinearRegression(fit_intercept=False)
    OLS_mse_kfold = Kfold_crossval(x, y, z, k, OLS, degrees)
  
    # kfold Ridge
    lambdas = np.logspace(-4, 4, 6)
    ridge_mse_kfold = np.zeros((len(lambdas), len(degrees)))

    for i, lmb in enumerate(lambdas):
        ridge = linear_model.Ridge(alpha=lmb, fit_intercept=False)
        ridge_mse_kfold[i] = Kfold_crossval(x, y, z, k, ridge, degrees)

    # print(OLS_mse_kfold.shape, ridge_mse_kfold.shape)
