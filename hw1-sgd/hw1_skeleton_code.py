import pandas as pd
import logging
import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


### Assignment Owner: Tian Wang

#######################################
####Q2.1: Normalization


def feature_normalization(train, test):
    """
    One common approach to feature normalization is to linearly transform
    (i.e. shift and rescale) each feature so that all feature values in the
    training set are in [0, 1]. Each feature gets its own transformation.
    We then apply the same transformations to each feature on the test set.
    It’s important that the transformation is “learned” on the training set,
    and then applied to the test set. It is possible that some transformed test
    set values will lie outside the [0, 1] interval.
    Args:
        train - training set, a 2D numpy array of size (num_instances, num_features)
        test  - test set, a 2D numpy array of size (num_instances, num_features)
    Returns:
        train_normalized - training set after normalization
        test_normalized  - test set after normalization

    """
    # TODO
    max = train.max(axis=0)
    min = train.min(axis=0)
    scope = max - min
    train_normalized = (train - min) / scope  # shift by min and rescale by scope
    test_normalized = (test - min) / scope

    return train_normalized, test_normalized


########################################
####Q2.2a: The square loss function

def compute_square_loss(X, y, theta):
    """
    Given a set of X, y, theta, compute the square loss for predicting y with X*theta
    
    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        theta - the parameter vector, 1D array of size (num_features)
    
    Returns:
        loss - the square loss, scalar
    """
    loss = 0  # initialize the square_loss
    # TODO
    A = X.dot(theta)
    m = y.shape[0]
    loss = np.dot(A, A) - 2 * np.dot(A, y) + np.dot(y, y)
    loss = loss / (2 * m)
    return loss


########################################
###Q2.2b: compute the gradient of square loss function
def compute_square_loss_gradient(X, y, theta):
    """
    Compute gradient of the square loss (as defined in compute_square_loss), at the point theta.
    
    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        theta - the parameter vector, 1D numpy array of size (num_features)
    
    Returns:
        grad - gradient vector, 1D numpy array of size (num_features)
    """
    # TODO
    m = y.shape[0]
    grad = theta.T @ (X.T @ X) - y.T @ X
    grad = grad / m
    return grad


###########################################
###Q2.3a: Gradient Checker
# Getting the gradient calculation correct is often the trickiest part
# of any gradient-based optimization algorithm.  Fortunately, it's very
# easy to check that the gradient calculation is correct using the
# definition of gradient.
# See http://ufldl.stanford.edu/wiki/index.php/Gradient_checking_and_advanced_optimization
def grad_checker(X, y, theta, epsilon=0.01, tolerance=1e-4):
    """Implement Gradient Checker
    Check that the function compute_square_loss_gradient returns the
    correct gradient for the given X, y, and theta.

    Let d be the number of features. Here we numerically estimate the
    gradient by approximating the directional derivative in each of
    the d coordinate directions: 
    (e_1 = (1,0,0,...,0), e_2 = (0,1,0,...,0), ..., e_d = (0,...,0,1) 

    The approximation for the directional derivative of J at the point
    theta in the direction e_i is given by: 
    ( J(theta + epsilon * e_i) - J(theta - epsilon * e_i) ) / (2*epsilon).

    We then look at the Euclidean distance between the gradient
    computed using this approximation and the gradient computed by
    compute_square_loss_gradient(X, y, theta).  If the Euclidean
    distance exceeds tolerance, we say the gradient is incorrect.

    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        theta - the parameter vector, 1D numpy array of size (num_features)
        epsilon - the epsilon used in approximation
        tolerance - the tolerance error
    
    Return:
        A boolean value indicate whether the gradient is correct or not

    """
    true_gradient = compute_square_loss_gradient(X, y, theta)  # the true gradient
    num_features = theta.shape[0]
    approx_grad = np.zeros(num_features)  # Initialize the gradient we approximate
    # TODO
    for i in range(num_features):
        e_ = np.zeros(num_features)
        e_[i] = 1
        approx_grad += ((compute_square_loss(X, y, theta + epsilon * e_) -
                        compute_square_loss(X, y, theta - epsilon * e_)) / (2 * epsilon)) * e_

    if np.dot(true_gradient-approx_grad, true_gradient-approx_grad) > tolerance**2:
        return False
    return True


#################################################
###Q2.3b: Generic Gradient Checker
def generic_gradient_checker(X, y, theta, objective_func, gradient_func, epsilon=0.01, tolerance=1e-4):
    """
    The functions takes objective_func and gradient_func as parameters. And check whether gradient_func(X, y, theta) returned
    the true gradient for objective_func(X, y, theta).
    Eg: In LSR, the objective_func = compute_square_loss, and gradient_func = compute_square_loss_gradient
    """
    # TODO
    true_gradient = gradient_func(X, y, theta)  # the true gradient
    num_features = theta.shape[0]
    approx_grad = np.zeros(num_features)  # Initialize the gradient we approximate
    # TODO
    for i in range(num_features):
        e_ = np.zeros(num_features)
        e_[i] = 1
        approx_grad += ((objective_func(X, y, theta + epsilon * e_) -
                        objective_func(X, y, theta - epsilon * e_)) / (2 * epsilon)) * e_

    if np.dot(true_gradient-approx_grad, true_gradient-approx_grad) > tolerance**2:
        return False
    return True

####################################
####Q2.4a: Batch Gradient Descent
def batch_grad_descent(X, y, alpha=0.1, num_iter=1000, check_gradient=False):
    """
    In this question you will implement batch gradient descent to
    minimize the square loss objective

    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        alpha - step size in gradient descent
        num_iter - number of iterations to run
        check_gradient - a boolean value indicating whether checking the gradient when updating

    Returns:
        theta_hist - store the the history of parameter vector in iteration, 2D numpy array of size (num_iter+1, num_features)
                    for instance, theta in iteration 0 should be theta_hist[0], theta in ieration (num_iter) is theta_hist[-1]
        loss_hist - the history of objective function vector, 1D numpy array of size (num_iter+1)
    """
    num_instances, num_features = X.shape[0], X.shape[1]
    theta_hist = np.zeros((num_iter + 1, num_features))  # Initialize theta_hist
    loss_hist = np.zeros(num_iter + 1)  # initialize loss_hist
    theta = np.ones(num_features)  # initialize theta
    theta_hist[0] = theta
    loss_hist[0] = compute_square_loss(X, y, theta)
    for i in range(num_iter):
        if check_gradient and not grad_checker(X, y, theta):
            return False
        alpha = backtracking_line_search(X, y, theta, alpha)
        theta = theta - alpha * compute_square_loss_gradient(X, y, theta)
        theta_hist[i+1] = theta
        loss_hist[i+1] = compute_square_loss(X, y, theta)
    return theta_hist, loss_hist


####################################
###Q2.4b: Implement backtracking line search in batch_gradient_descent
###Check http://en.wikipedia.org/wiki/Backtracking_line_search for details
def backtracking_line_search(X, y, theta, alpha, lambda_reg=0):
    while True:
        loss = compute_square_loss(X, y, theta)
        old_theta = theta
        theta = theta - alpha * compute_regularized_square_loss_gradient(X, y, theta, lambda_reg)
        if compute_square_loss(X, y, theta) > loss: #we need to backtrack and find a better loss
            alpha = alpha / 2
            theta = old_theta
        else: #current alpha would suffice
            break
    return alpha

###################################################
###Q2.5a: Compute the gradient of Regularized Batch Gradient Descent
def compute_regularized_square_loss_gradient(X, y, theta, lambda_reg):
    """
    Compute the gradient of L2-regularized square loss function given X, y and theta
    
    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        theta - the parameter vector, 1D numpy array of size (num_features)
        lambda_reg - the regularization coefficient
    
    Returns:
        grad - gradient vector, 1D numpy array of size (num_features)
    """
    # TODO
    grad = compute_square_loss_gradient(X, y, theta) + 2*lambda_reg*theta.T
    return grad
###################################################
###Q2.5b: Batch Gradient Descent with regularization term
def regularized_grad_descent(X, y, alpha=0.1, lambda_reg=1, num_iter=1000):
    """
    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        alpha - step size in gradient descent
        lambda_reg - the regularization coefficient
        numIter - number of iterations to run 
        
    Returns:
        theta_hist - the history of parameter vector, 2D numpy array of size (num_iter+1, num_features) 
        loss_hist - the history of regularized loss value, 1D numpy array
    """
    (num_instances, num_features) = X.shape
    theta = np.ones(num_features)  # Initialize theta
    theta_hist = np.zeros((num_iter + 1, num_features))  # Initialize theta_hist
    loss_hist = np.zeros(num_iter + 1)  # Initialize loss_hist
    theta_hist[0] = theta
    loss_hist[0] = compute_square_loss(X, y, theta)
    for i in range(num_iter):
        # alpha = backtracking_line_search(X, y, theta, alpha, lambda_reg)
        theta = theta - alpha * compute_regularized_square_loss_gradient(X, y, theta, lambda_reg)
        theta_hist[i+1] = theta
        loss_hist[i+1] = compute_square_loss(X, y, theta)
    return theta_hist, loss_hist

#############################################
##Q2.5c: Visualization of Regularized Batch Gradient Descent
##X-axis: log(lambda_reg)
##Y-axis: square_loss
def visualize_batch_grad(X, y, alpha=0.01, lambda_reg=0.01, num_iter=1000):
    theta_hist, loss_hist = regularized_grad_descent(X, y, alpha, lambda_reg, num_iter)
    # for i in range(num_iter):
    #     print(loss_hist[i])
    x_axis = np.linspace(0, 10, 1001)
    y_axis = loss_hist
    plt.plot(x_axis, y_axis, '-')
    plt.show()

#############################################
def shuffle_rows(arr,rows):
    np.random.shuffle(arr[rows[0]:rows[1]+1])
###Q2.6a: Stochastic Gradient Descent
def stochastic_grad_descent(X, y, alpha=0.1, lambda_reg=1, num_iter=1000):
    """
    In this question you will implement stochastic gradient descent with a regularization term
    
    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        alpha - string or float. step size in gradient descent
                NOTE: In SGD, it's not always a good idea to use a fixed step size. Usually it's set to 1/sqrt(t) or 1/t
                if alpha is a float, then the step size in every iteration is alpha.
                if alpha == "1/sqrt(t)", alpha = 1/sqrt(t)
                if alpha == "1/t", alpha = 1/t
        lambda_reg - the regularization coefficient
        num_iter - number of epochs (i.e number of times) to go through the whole training set
    
    Returns:
        theta_hist - the history of parameter vector, 3D numpy array of size (num_iter, num_instances, num_features) 
        loss hist - the history of regularized loss function vector, 2D numpy array of size(num_iter, num_instances)
    """
    num_instances, num_features = X.shape[0], X.shape[1]
    theta = np.ones(num_features)  # Initialize theta

    theta_hist = np.zeros((num_iter, num_instances, num_features))  # Initialize theta_hist
    loss_hist = np.zeros((num_iter, num_instances))  # Initialize loss_hist
    # TODO
    shuffle_rows(X, [0, num_instances-1])
    for i in range(num_iter):
        for j in range(num_instances):
            grad = (np.dot(theta, X[j]) - y[j]) * X[j].T + 2*lambda_reg*theta.T
            theta = theta - alpha * grad
            theta_hist[i][j] = theta
            loss_hist[i][j] = 1/2 * ((np.dot(theta, X[j]) - y[j])**2)
################################################
###Q2.6b Visualization that compares the convergence speed of batch
###and stochastic gradient descent for various approaches to step_size
##X-axis: Step number (for gradient descent) or Epoch (for SGD)
##Y-axis: log(objective_function_value)

def main():
    # Loading the dataset
    print('loading the dataset')

    df = pd.read_csv('hw1-data.csv', delimiter=',')
    X = df.values[:, :-1]
    y = df.values[:, -1]

    print('Split into Train and Test')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=100, random_state=10)

    print("Scaling all to [0, 1]")
    X_train, X_test = feature_normalization(X_train, X_test)
    X_train = np.hstack((X_train, np.ones((X_train.shape[0], 1))))  # Add bias term
    X_test = np.hstack((X_test, np.ones((X_test.shape[0], 1))))  # Add bias term


    # TODO
    visualize_batch_grad(X_train, y_train, )

if __name__ == "__main__":
    main()
