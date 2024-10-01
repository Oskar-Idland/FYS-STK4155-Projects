import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
import sklearn
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.utils import resample
import warnings
warnings.filterwarnings("ignore")


def FrankeFunction(x,y):
    """ 
    Generates a surface plot of the Franke function.
    """
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

def MSE(y: np.ndarray, y_tilde: np.ndarray) -> float:
    """
    Calculates the Mean Squared Error (MSE) between the true and predicted values.

    Parameters:
    y (np.ndarray): The actual data values.
    y_pred (np.ndarray): The predicted data values from the model.

    Returns:
    float: The Mean Squared Error.
    """
    n = len(y)
    return np.sum((y - y_tilde)**2) / n

def R2(y_data: np.ndarray, y_model: np.ndarray) -> float:
    """
    Calculates the R2 score of the model.

    Parameters:
    y_data (np.ndarray): The actual data values.
    y_model (np.ndarray): The predicted data values from the model.

    Returns:
    float: The R2 score.
    """
    return 1 - np.sum((y_data - y_model)**2) / np.sum((y_data - np.mean(y_data))**2)


def create_X(x: np.ndarray, y: np.ndarray, n: int) -> np.ndarray:
    """
    Creates the design matrix X.

    Parameters:
    x (np.ndarray): The independent variable(s).
    y (np.ndarray): The independent variable(s).
    n (int): The degree of the polynomial features.

    Returns:
    np.ndarray: The design matrix X.
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


def OLS(x: np.ndarray, y: np.ndarray, z: np.ndarray[np.ndarray, np.ndarray], degree: int, scale: bool =True, test_size: float =0.2, seed: int =None, intercept: bool =False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Performs Ordinary Least Squares (OLS) regression.

    Parameters:
    x (array-like): The independent variable(s).
    y (array-like): The independent variable(s).
    z (array-like): The dependent variable.
    degree (int): The degree of the polynomial features.
    scale (bool, optional): Whether to scale the data. Default is True.
    test_size (float, optional): The proportion of the dataset to include in the test split. Default is 0.2.
    seed (int, optional): The random seed for reproducibility. Default is None.
    intercept (bool, optional): Whether to include an intercept term. Default is False.

    Returns:
    tuple: A tuple containing the Mean Squared Error (MSE) score, R-squared (R2) score, and the beta values (coefficients).
    '''
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

    if not intercept:
        β[0] = 0
    
    z_tilde = X_train @ β 
    z_pred = X_test @ β
    
    MSE_score = MSE(z_train, z_tilde)
    R2_score = R2(z_test, z_pred)

    return MSE_score, R2_score, β


def Ridge(x: np.ndarray, y: np.ndarray, z: np.ndarray, degree: int, λ: float, scale: bool = True, test_size: float = 0.2, seed: int = None, intercept: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Performs Ridge regression.

    Parameters:
    x (np.ndarray): The independent variable(s).
    y (np.ndarray): The independent variable(s).
    z (np.ndarray): The dependent variable.
    degree (int): The degree of the polynomial features.
    λ (float): The regularization parameter.
    scale (bool, optional): Whether to scale the data. Default is True.
    test_size (float, optional): The proportion of the dataset to include in the test split. Default is 0.2.
    seed (int, optional): The random seed for reproducibility. Default is None.
    intercept (bool, optional): Whether to include an intercept term. Default is False.

    Returns:
    tuple: A tuple containing the Mean Squared Error (MSE) score, R-squared (R2) score, and the beta values (coefficients).
    """

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

    if not intercept:
        β[0] = 0

    z_tilde = X_train @ β
    z_pred = X_test @ β

    MSE_score = MSE(z_train, z_tilde)
    R2_score = R2(z_test, z_pred)

    return MSE_score, R2_score, β


def Lasso(x: np.ndarray, y: np.ndarray, z: np.ndarray, degree: int, λ: float, scale: bool = True, test_size: float = 0.2, seed: int = None, intercept: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Performs Lasso regression.

    Parameters:
    x (np.ndarray): The independent variable(s).
    y (np.ndarray): The independent variable(s).
    z (np.ndarray): The dependent variable.
    degree (int): The degree of the polynomial features.
    λ (float): The regularization parameter.
    scale (bool, optional): Whether to scale the data. Default is True.
    test_size (float, optional): The proportion of the dataset to include in the test split. Default is 0.2.
    seed (int, optional): The random seed for reproducibility. Default is None.
    intercept (bool, optional): Whether to include an intercept term. Default is False.

    Returns:
    tuple: A tuple containing the Mean Squared Error (MSE) score, R-squared (R2) score, and the beta values (coefficients).
    """

    X = create_X(x, y, degree)
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=test_size, random_state=seed)

    if not scale: # Scaling the data
        scaler_X = StandardScaler().fit(X_train)
        X_train = scaler_X.transform(X_train)
        X_test = scaler_X.transform(X_test)

        scaler_z = StandardScaler().fit(z_train)
        z_train = scaler_z.transform(z_train)
        z_test= scaler_z.transform(z_test)

    lasso = linear_model.Lasso(λ, fit_intercept=intercept)
    lasso.fit(X_train, z_train)

    β = lasso.coef_

    if intercept:
      β = [lasso.intercept_, *lasso.coef_]

    z_tilde = lasso.predict(X_train)
    z_pred = lasso.predict(X_test)

    MSE_score = MSE(z_train, z_tilde)
    R2_score = R2(z_test, z_pred)

    return MSE_score, R2_score, β

def Kfold_crossval(x: np.ndarray, y: np.ndarray, z: np.ndarray, k: int, model: sklearn.linear_model.LinearRegression | sklearn.linear_model.Ridge | sklearn.linear_model.Lasso, degree: int) -> float:
    """
    Performs k-fold cross-validation.

    Parameters:
    x (np.ndarray): The independent variable(s).
    y (np.ndarray): The independent variable(s).
    z (np.ndarray): The dependent variable.
    k (int): The number of folds in the k-fold cross-validation.
    model (sklearn.linear_model.LinearRegression | sklearn.linear_model.Ridge | sklearn.linear_model.Lasso): The regression model to be used.
    degree (int): The degree of the polynomial features.

    Returns:
    float: The estimated Mean Squared Error (MSE) from the k-fold cross-validation.
    """

    kfold = KFold(n_splits=k)

    X = create_X(x, y, degree)
    estimated_mse_folds = cross_val_score(model, X, z, scoring='neg_mean_squared_error', cv=kfold)
    estimated_mse = np.mean(-estimated_mse_folds)

    return estimated_mse

if __name__ == "__main__":
      # seed = np.random.randint(1,1000)

    """Initial setup"""
    seed = 42
    np.random.seed(seed)
    N = 100

    x = np.arange(0, 1, 1/N)
    y = np.arange(0, 1, 1/N)
    X, Y = np.meshgrid(x, y)

    z = FrankeFunction(X, Y)

    """"OLS example"""
    degrees = np.arange(1, 6)

    MSE_list = []
    R2_list = []
    β_list = []

    for degree in degrees:
        MSE_i, R2_i, β_i = OLS(x, y, z, degree, seed=seed)
        MSE_list.append(MSE_i) ; R2_list.append(R2_i) ; β_list.append(β_i)


    """Ridge example"""
    degrees = np.arange(1, 6)
    lambdas = np.logspace(-4, 4, 6)

    MSE_list = []
    R2_list = []
    β_list = []

    for degree in degrees:
        for lmb in lambdas:
            MSE_i, R2_i, β_i = Ridge(x, y, z, degree, lmb, seed=seed)
            MSE_list.append(MSE_i) ; R2_list.append(R2_i) ; β_list.append(β_i)
      

    """Lasso example"""
    degrees = np.arange(1, 6)
    lambdas = np.logspace(-4, 4, 6)

    MSE_list = []
    R2_list = []
    β_list = []

    for degree in degrees:
        for lmb in lambdas:
            MSE_i, R2_i, β_i = Lasso(x, y, z, degree, lmb, seed=seed)
            MSE_list.append(MSE_i) ; R2_list.append(R2_i) ; β_list.append(β_i)
      
    """Kfold example"""
    degrees = np.arange(1, 6) 
    k = [5, 10]

    # kfold OLS  
    OLS_mse_kfold = []
    OLS = LinearRegression(fit_intercept=False)
    for k_i in k:
        for degree in degrees:
            OLS_mse_kfold.append(Kfold_crossval(x, y, z, k_i, OLS, degree))
  
    # kfold Ridge
    lambdas = np.logspace(-4, 4, 6)
    ridge_mse_kfold = []

    for k_i in k:
        for degree in degrees:
            for lmb in lambdas:
                ridge = linear_model.Ridge(alpha=lmb, fit_intercept=False)
                ridge_mse_kfold.append(Kfold_crossval(x, y, z, k_i, ridge, degree))

