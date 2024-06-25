import numpy as np

np.set_printoptions(precision=4)

def linear_regression(y:np.ndarray, x:np.ndarray) -> np.ndarray:
    one_padded_x = np.ones((4, 6))
    one_padded_x[1:3, :] = x
    one_padded_x = one_padded_x.T

    inverse = np.linalg.pinv(one_padded_x)  # pinv(x) -> inv(x.T @ x) x.T
    weights = inverse @ labels

    # solution: a
    error = np.linalg.norm(y - (one_padded_x @ weights))
    print(f"Linear regression error: {error}")

    # solution: b
    predicted_labels = (one_padded_x @ weights)
    print(f"Predicted labels before clamping: {predicted_labels.squeeze(-1)}")
    clamped_labels = np.where(predicted_labels > 0.5, 1, 0)
    print(f"Predicted labels after clamping:{clamped_labels.squeeze(-1)}")

    return weights.squeeze(-1)

def _sigmoid(x):
    return 1 / (1 + np.exp(-x))

def logistic_regression(y:np.ndarray, x:np.ndarray) -> np.ndarray:
    one_padded_x = np.ones((4, 6))
    one_padded_x[1:3, :] = x
    one_padded_x = one_padded_x.T

    weights = np.random.rand(one_padded_x.shape[-1])
    first_predictions = _sigmoid(one_padded_x @ weights)
    print(f"Logistic Regression before optmization:{first_predictions}")

    for _ in range(100):
        z = one_padded_x @ weights
        predictions = _sigmoid(z)
        
        # Compute the gradient of the log-likelihood
        gradient = (one_padded_x.T @ (np.expand_dims(predictions, -1) - y)) / y.size   # dL/dW
        
        # Compute the Hessian matrix
        R = np.diag(predictions * (1 - predictions))
        H = (one_padded_x.T @ R @ one_padded_x) / y.size                               # ddL/ddW
        
        # Update the weights using the Newton-Raphson update rule
        weights -= (np.linalg.pinv(H) @ gradient).squeeze(-1)
        
    optimized_predictions = _sigmoid(one_padded_x @ weights)
    print(f"Logistic Regression after optmization:{optimized_predictions}")

    return weights


if __name__ == "__main__":

    labels = np.array([[1],
                       [0],
                       [0],
                       [1],
                       [1],
                       [1]])
    
    data = np.array([[1, -2, 0.3, 5, 3, 7],
                     [3, 2, 1, -1, 4, 3]])
    

    linear_reg_coeff = linear_regression(labels, data)
    print(f"Linear Regression coeffcients: {linear_reg_coeff}\n")

    logistic_reg_coeff = logistic_regression(labels, data)
    print(f"Logistic Regression coefficient: {logistic_reg_coeff}")