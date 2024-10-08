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

def MSE(z: np.ndarray, z_pred: np.ndarray) -> float:
    """
    Calculates the Mean Squared Error (MSE) between the true and predicted values.

    ## Parameters:
    z (np.ndarray): The actual data values.
    z_pred (np.ndarray): The predicted data values from the model.

    ## Returns:
    float: The Mean Squared Error.
    """
    z = z.flatten()
    z_pred = z_pred.flatten()
    n = len(z)
    return np.sum((z - z_pred)**2) / n

def R2(z: np.ndarray, z_pred: np.ndarray) -> float:
    """
    Calculates the R2 score of the model.

    ## Parameters:
    z (np.ndarray): The actual data values.
    z_pred (np.ndarray): The predicted data values from the model.

    ## Returns:
    float: The R2 score.
    """
    z = z.flatten()
    z_pred = z_pred.flatten()
    return 1 - np.sum((z - z_pred)**2) / np.sum((z - np.mean(z))**2)


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
    
    xx, yy = np.meshgrid(x, y)        # Make a meshgrid to get all possible combinations of x and y values
    xx = xx.flatten()
    yy = yy.flatten()

    idx = 1
    for i in range(1, n+1):
        for j in range(i+1):
            X[:, idx] = xx**(i-j)*yy**j
            idx += 1

    return X


def OLS(x: np.ndarray, y: np.ndarray, z: np.ndarray[np.ndarray, np.ndarray], degree: int, scale: bool = True, test_size: float = 0.2, seed: int | None = None, return_beta: bool = False, return_X: bool = False, return_scalers: bool = False, return_train_test: bool = False) -> tuple[np.ndarray]:
    '''
    Performs Ordinary Least Squares (OLS) regression.

    ## Parameters:
    x (array-like): Independent variable.
    y (array-like): Independent variable.
    z (array-like): The dependent variable.
    degree (int): The degree of the polynomial features.
    scale (bool, optional): Whether to scale the data. Default is True.
    test_size (float, optional): The proportion of the dataset to include in the test split. Default is 0.2.
    seed(int | None, optional): The random seed for reproducibility. Default is None.
    return beta (bool, optional): Whether to return the features β. Default is False.
    return_X (bool, optional): Whether to return X. Default is False.
    return_scalers (bool, optional): Whether to return the scalers used to scale the data. Default is False.
    return_train_test (bool, optional): Whether to return X_train, X_test, z_train and z_test. Default is False.

    ## Returns:
    tuple: A tuple containing the Mean Squared Error (MSE) score and the R-squared (R2) score, as well as the beta values (coefficients), the design matrix X, the scalers for X and z, and/or the training and test sets for X and z, depending on the passed arguments.
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
    
    z_pred = X_test @ β
    
    MSE_score = MSE(z_test, z_pred)
    R2_score = R2(z_test, z_pred)

    quantities = [MSE_score, R2_score]
    if return_beta:
        quantities.append(β)
    if return_X:
        quantities.append(X)
    if return_scalers:
        quantities.extend([scaler_X, scaler_z])
    if return_train_test:
        quantities.extend([X_train, X_test, z_train, z_test])
    return tuple(quantities)


def Ridge(x: np.ndarray, y: np.ndarray, z: np.ndarray, degree: int, λ: float, scale: bool = True, test_size: float = 0.2, seed: int | None = None, return_beta: bool = False, return_X: bool = False, return_scalers: bool = False, return_train_test: bool = False) -> tuple[np.ndarray]:
    """
    Performs Ridge regression.

    ## Parameters:
    x (np.ndarray): Independent variable.
    y (np.ndarray): Independent variable.
    z (np.ndarray): The dependent variable.
    degree (int): The degree of the polynomial features.
    λ (float): The regularization parameter.
    scale (bool, optional): Whether to scale the data. Default is True.
    test_size (float, optional): The proportion of the dataset to include in the test split. Default is 0.2.
    seed(int | None, optional): The random seed for reproducibility. Default is None.
    return beta (bool, optional): Whether to return the features β. Default is False.
    return_X (bool, optional): Whether to return X. Default is False.
    return_scalers (bool, optional): Whether to return the scalers used to scale the data. Default is False.
    return_train_test (bool, optional): Whether to return X_train, X_test, z_train and z_test. Default is False.

    ## Returns:
    tuple: A tuple containing of length 2-10, containing the Mean Squared Error (MSE) score and the R-squared (R2) score, as well as the beta values (coefficients), the design matrix X, the scalers for X and z, and/or the training and test sets for X and z, depending on the passed arguments.
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

    z_pred = X_test @ β

    MSE_score = MSE(z_test, z_pred)
    R2_score = R2(z_test, z_pred)

    quantities = [MSE_score, R2_score]
    if return_beta:
        quantities.append(β)
    if return_X:
        quantities.append(X)
    if return_scalers:
        quantities.extend([scaler_X, scaler_z])
    if return_train_test:
        quantities.extend([X_train, X_test, z_train, z_test])
    return tuple(quantities)


def Lasso(x: np.ndarray, y: np.ndarray, z: np.ndarray, degree: int, λ: float, scale: bool = True, test_size: float = 0.2, seed: int | None = None, intercept: bool = False, return_beta: bool = False, return_X: bool = False, return_scalers: bool = False, return_train_test: bool = False) -> tuple[np.ndarray]:
    """
    Performs Lasso regression.

    ## Parameters:
    x (np.ndarray): Independent variable.
    y (np.ndarray): Independent variable.
    z (np.ndarray): The dependent variable.
    degree (int): The degree of the polynomial features.
    λ (float): The regularization parameter.
    scale (bool, optional): Whether to scale the data. Default is True.
    test_size (float, optional): The proportion of the dataset to include in the test split. Default is 0.2.
    seed(int | None, optional): The random seed for reproducibility. Default is None.
    intercept (bool, optional): Whether to include an intercept term. Default is False.
    return beta (bool, optional): Whether to return the features β. Default is False.
    return_X (bool, optional): Whether to return X. Default is False.
    return_scalers (bool, optional): Whether to return the scalers used to scale the data. Default is False.
    return_train_test (bool, optional): Whether to return X_train, X_test, z_train and z_test. Default is False.

    ## Returns:
    tuple: A tuple of length 2-10, containing the Mean Squared Error (MSE) score and the R-squared (R2) score, as well as the beta values (coefficients), the design matrix X, the scalers for X and z, and/or the training and test sets for X and z, depending on the passed arguments.
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

    lasso = linear_model.Lasso(λ, fit_intercept=intercept)
    lasso.fit(X_train, z_train)

    β = lasso.coef_
    if intercept:
        β[0] = lasso.intercept_[0]

    z_pred = lasso.predict(X_test)

    MSE_score = MSE(z_test, z_pred)
    R2_score = R2(z_test, z_pred)

    quantities = [MSE_score, R2_score]
    if return_beta:
        quantities.append(β)
    if return_X:
        quantities.append(X)
    if return_scalers:
        quantities.extend([scaler_X, scaler_z])
    if return_train_test:
        quantities.extend([X_train, X_test, z_train, z_test])
    return tuple(quantities)


def Bootstrap(x: np.ndarray, y: np.ndarray, z: np.ndarray, degree: int, n_bootstraps: int, scale: bool = True, test_size: float = 0.2, seed: int | None = None) -> tuple[float, float, float]:
    """
    Performs bootstrapping.

    ## Parameters:
    x (np.ndarray): Independent variable.
    y (np.ndarray): Independent variable.
    z (np.ndarray): The dependent variable.
    degree (int): The degree of the polynomial features.
    n_bootstraps (int): The number of bootstraps.
    scale (bool, optional): Whether to scale the data. Default is True.
    test_size (float, optional): The proportion of the dataset to include in the test split. Default is 0.2.
    seed(int | None, optional): The random seed for reproducibility. Default is None.   

    ## Returns:
    tuple: A tuple of length 3 containing the Mean Squared Error (MSE) score, bias, and variance
    """

    X = create_X(x, y, degree)
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=test_size, random_state=seed)
    
    if scale:
        scaler_X = StandardScaler().fit(X_train)
        X_train = scaler_X.transform(X_train)
        X_test = scaler_X.transform(X_test)

        scaler_y = StandardScaler().fit(z_train)
        z_train = scaler_y.transform(z_train)
        z_test = scaler_y.transform(z_test)

    z_pred = np.empty((z_test.shape[0], n_bootstraps))

    for j in range(n_bootstraps):
        X_, z_ = resample(X_train, z_train)
        
        β = np.linalg.pinv(X_.T @ X_) @ X_.T @ z_

        z_pred[:, j] = (X_test @ β).ravel()

    error = np.mean(np.mean((z_test - z_pred)**2, axis=1, keepdims=True))  
    bias = np.mean((z_test - np.mean(z_pred, axis=1, keepdims=True))**2)            
    variance = np.mean(np.var(z_pred, axis=1, keepdims=True))  

    return error, bias, variance


def kfold_crossval(x: np.ndarray, y: np.ndarray, z: np.ndarray, k: int, model, degree: int, scale: bool = True, predict: bool = False, seed: int | None = None) -> float | np.ndarray:
    """
    Performs k-fold cross-validation.

    ## Parameters:
    x (np.ndarray): Independent variable.
    y (np.ndarray): Independent variable.
    z (np.ndarray): The dependent variable.
    k (int): The number of folds in the k-fold cross-validation.
    model (sklearn.linear_model.LinearRegression | sklearn.linear_model.Ridge | sklearn.linear_model.Lasso): The regression model to be used.
    degree (int): The degree of the polynomial features.
    seed(int | None, optional): The random seed for reproducibility. Default is None.
    scale (bool, optional): Whether to scale the data. Default is True.
    predict (bool, optional): Whether to return the predicted values instead of the score. Default is False.

    ## Returns:
    float | np.ndarray: The estimated Mean Squared Error (MSE) from the k-fold cross-validation if predict is passed as False, and the predicted values if predict is passed as True.
    """

    kfold = KFold(n_splits = k, shuffle = True, random_state = seed) 

    X = create_X(x, y, degree)
    if scale:
        scaler_X = StandardScaler().fit(X)
        scaler_z = StandardScaler().fit(z)
        X = scaler_X.transform(X)
        z = scaler_z.transform(z)

    if predict:
        z_pred = cross_val_predict(model, X, z, cv = kfold).reshape(-1, 1)
        if scale:
            z_pred = scaler_z.inverse_transform(z_pred)
        return z_pred
    else:
        estimated_mse_folds = cross_val_score(model, X, z, scoring = "neg_mean_squared_error", cv = kfold)
        estimated_mse = np.mean(-estimated_mse_folds)
        return estimated_mse


if __name__ == "__main__":
      # seed = np.random.randint(1,1000)

    """Initial setup"""
    seed = 43
    np.random.seed(seed)
    N = 50

    x = np.arange(0, 1, 1/N)
    y = np.arange(0, 1, 1/N)
    X, Y = np.meshgrid(x, y)

    z = (FrankeFunction(X, Y))

    """Franke plot"""
    fig = plt.figure(figsize = (8, 6))
    ax = fig.add_subplot(projection="3d")
    surf = ax.plot_surface(X, Y, z, cmap = cm.coolwarm, linewidth = 0, antialiased = False)
    fig.colorbar(surf)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_title("Franke function")
    plt.show()
 

    """"OLS example"""
    degrees = np.arange(1, 6)
    z = (z + np.random.normal(0, 0.1, (N, N))).reshape(-1, 1) 

    MSE_list = []
    R2_list = []
    β_list = []

    for degree in degrees:
        MSE_i, R2_i, β_i = OLS(x, y, z, degree, seed=seed, return_beta=True)
        MSE_list.append(MSE_i) ; R2_list.append(R2_i) ; β_list.append(β_i)

    values = [MSE_list, R2_list, β_list]
    titles = ["MSE", r"$R^2$", r"$\beta$"]

    fig, axs = plt.subplots(1, 3, figsize = (15, 5), constrained_layout = True)
    
    for i in range(len(axs)-1): # Plotting MSE and R2
        axs[i].plot(degrees, values[i], '--o')
        axs[i].set_title(titles[i]) 

    for i, deg in enumerate(degrees): # Plotting β
        m = len(β_list[i][:, 0])             
        plt.scatter([deg]*m, β_list[i][:, 0], c = "blue")
        plt.title(titles[2])
        
    fig.supxlabel('Polynomial degree')
    fig.supylabel('Values')
    fig.suptitle("OLS")
    plt.show()

    """Ridge example"""
    degrees = np.arange(1, 6)
    lambdas = np.logspace(-4, 4, 6)

    MSE_list = []
    R2_list = []

    for i, degree in enumerate(degrees):
        MSE_list.append([]) ; R2_list.append([])
        for lmb in lambdas:
            MSE_i, R2_i = Ridge(x, y, z, degree, lmb, seed=seed)
            MSE_list[i].append(MSE_i) ; R2_list[i].append(R2_i)

    values = ([MSE_list, R2_list])
    titles = ["MSE", r"$R^2$"]
    im = []

    fig, axs = plt.subplots(1, 2, figsize = (15, 5), constrained_layout = True)
    extent = [np.min(degrees), np.max(degrees), np.min(lambdas), np.max(lambdas)]

    for i in range(len(axs)): # Plotting MSE and R2
        im.append(axs[i].imshow(values[i], cmap = "Reds", extent = extent, aspect = "auto"))
        axs[i].set_title(titles[i])
        plt.colorbar(im[i], ax = axs[i])

    fig.supxlabel("Polynomial degree")
    fig.supylabel(r"Hyperparameter $\lambda$")
    fig.suptitle("Ridge")
    plt.show()

    """Lasso example"""
    degrees = np.arange(1, 7)
    lambdas = np.logspace(-7, 2, 30)

    MSE_list = []
    R2_list = []

    for i, degree in enumerate(degrees):
        MSE_list.append([]) ; R2_list.append([])
        for lmb in lambdas:
            MSE_i, R2_i = Lasso(x, y, z, degree, lmb, seed=seed)
            MSE_list[i].append(MSE_i) ; R2_list[i].append(R2_i)

    fig, axs = plt.subplots(3, 2, figsize = (12, 10), constrained_layout = True)
    idx = [[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1]]
    
    for i in range(len(degrees)):   
        ax_1 = axs[idx[i][0], idx[i][1]]
        ax_2 = ax_1.twinx()
        ax_1.plot(lambdas, MSE_list[i],  'r--o')
        ax_2.plot(lambdas, R2_list[i],  'b--o')

        ax_1.set_xscale("log")
        ax_1.set_xscale("log")

        ax_1.set_title(f'Polynomial degree {degrees[i]}')
        ax_1.yaxis.set_major_formatter(FormatStrFormatter("%.2e"))
        ax_1.yaxis.set_major_formatter(FormatStrFormatter("%.1e"))

        ax_1.set_ylabel('MSE')
        ax_2.set_ylabel(r'$R^2$')

        ax_1.tick_params("y", colors = "r")
        ax_2.tick_params("y", colors = "b")

        if i < len(degrees) - 2:
            ax_1.set_xticks([])
            ax_1.set_xticks([])

    fig.suptitle("Lasso")
    plt.show()

    """Bootstrap example"""

    degrees = range(1, 41, 4)

    n_bootstraps = 100

    error = np.zeros((len(degrees), 1))
    bias = np.zeros((len(degrees), 1))
    variance = np.zeros((len(degrees), 1))

    for i, degree in enumerate(degrees):
        error[i], bias[i], variance[i] = Bootstrap(x, y, z, degree, n_bootstraps, seed=seed)
      
    plt.figure(figsize = (10, 6))
    plt.plot(degrees, error, label = "MSE on test set", linestyle = "--", marker = "o", color = "#ff8d00")
    plt.plot(degrees, bias, label = "Bias", linestyle = "--", marker = "o", color = "slateblue")
    plt.plot(degrees, variance, label = "Variance", linestyle = "--", marker = "o", color = "#ff77bc")
    plt.legend(loc = 2)
    plt.xticks(degrees[::2])
    plt.xlabel("Polynomial degree")
    plt.ylabel("Error")
    plt.title("Bootstrap")
    plt.tight_layout()
    plt.show()


    """Kfold example"""
    degrees = np.arange(1, 6) 
    k = [5, 10]

    # kfold OLS  
    OLS_mse_kfold = np.zeros((len(degrees), len(k)))
                             
    for i in range(len(degrees)):
        for j in range(len(k)):
            OLS_mse_kfold[i, j] = kfold_crossval(x, y, z, k[j], LinearRegression(fit_intercept = False), degrees[i], seed = seed)
    
    plt.imshow(OLS_mse_kfold.T[::-1], cmap = "PuRd", aspect = "auto", extent = [np.min(degrees), np.max(degrees), np.min(k), np.max(k)])
    plt.colorbar(label = "MSE")
    plt.title("OLS k-fold cross-validation")
    plt.xlabel("Polynomial degree")
    plt.ylabel(r"Number of folds $k$")
    plt.show()


    # kfold Ridge
    lambdas = [0.001, 0.5]
    ridge_mse_kfold = np.zeros((len(degrees), len(k), len(lambdas)))

    for i in range(len(degrees)):
        for j in range(len(k)):
            for l in range(len(lambdas)):
                ridge_mse_kfold[i, j, l] = kfold_crossval(x, y, z, k[j], linear_model.Ridge(lambdas[l], fit_intercept = False), degrees[i], seed = seed)

    im = []

    fig, axs = plt.subplots(1, 2, figsize = (15, 5), constrained_layout = True)

    for i in range(len(axs)): # Plotting MSE
        im.append(axs[i].imshow(ridge_mse_kfold[:, :, i].T[::-1], cmap = "PuBu", aspect = "auto", extent = [np.min(degrees), np.max(degrees), np.min(k), np.max(k)]))
        axs[i].set_title(f"λ = {lambdas[i]}")
        plt.colorbar(im[-1], ax = axs[i], pad = 0.02, aspect = 10, label = "MSE" if j == 1 else None)

    fig.supxlabel("Polynomial degree")
    fig.supylabel(r"Number of folds $k$")
    fig.suptitle("Ridge k-fold cross-validation")
    plt.show()