def profile_gradient_descent(model):
    model.gradient_descent(50, eta=param2_array[np.argmin(MSE_array_plain)])
    
profile_gradient_descent(model)
