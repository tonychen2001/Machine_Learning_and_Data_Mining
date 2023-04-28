"""
CSCC11 - Introduction to Machine Learning, Fall 2021, Assignment 2
M. Ataei
"""

import numpy as np
from numpy.core.shape_base import hstack

from logistic_regression import LogisticRegression
from utils import load_pickle_dataset


def train(train_X,
          train_y,
          test_X=None,
          test_y=None,
          data_preprocessing=lambda X: X,
          factor=1,
          bias=0,
          alpha_inverse=0,
          beta_inverse=0,
          num_epochs=1000,
          step_size=1e-3,
          check_grad=False,
          verbose=False):
    """ This function trains a logistic regression model given the data.

    Args:
    - train_X (ndarray (shape: (N, D))): A NxD matrix consisting N D-dimensional training inputs.
    - train_y (ndarray (shape: (N, 1))): A N-column vector consisting N scalar training outputs (labels).
    - test_X (ndarray (shape: (M, D))): A NxD matrix consisting M D-dimensional test inputs.
    - test_y (ndarray (shape: (M, 1))): A N-column vector consisting M scalar test outputs (labels).
    - data_preprocessing (ndarray -> ndarray): A data-preprocessing function that is applied on both the
                                               training and test inputs.

    Initialization Args:
    - factor (float): A constant factor of the randomly initialized weights.
    - bias (float): The bias value

    Learning Args:
    - num_epochs (int): Number of gradient descent steps
                        NOTE: 1 <= num_epochs
    - step_size (float): Gradient descent step size
    - check_grad (bool): Whether or not to check gradient using finite difference.
    - verbose (bool): Whether or not to print gradient information for every step.
    """
    train_accuracy = 0
    # ====================================================
    # TODO: Implement your solution within the box
    # Step 0: Apply data-preprocessing (i.e. feature map) on the input data
    train_X = data_preprocessing(train_X)
    # Step 1: Initialize model and initialize weights
    model = LogisticRegression(train_X.shape[1], np.amax(train_y) + 1)
    model.init_weights(factor, bias)

    # Step 2: Train the model
    model.learn(train_X, train_y, num_epochs, step_size,
                check_grad, verbose, alpha_inverse, beta_inverse)

    # Step 3: Evaluate training performance
    train_probs = model.predict(train_X)

    # ====================================================
    train_preds = np.argmax(train_probs, axis=1)
    train_accuracy = 100 * np.mean(train_preds == train_y.flatten())
    print("Training Accuracy: {}%".format(train_accuracy))

    if test_X is not None and test_y is not None:
        test_accuracy = 0
        # ====================================================
        # TODO: Implement your solution within the box
        # Evaluate test performance
        test_X = data_preprocessing(test_X)
        test_probs = model.predict(test_X)

        # ====================================================
        test_preds = np.argmax(test_probs, axis=1)
        test_accuracy = 100 * np.mean(test_preds == test_y.flatten())
        print("Test Accuracy: {}%".format(test_accuracy))


def feature_map(X):
    """ This function perform applies a feature map on the given input.

        Given any 2D input vector x, the output of the feature map psi is a 3D vector, defined as:
        psi(x) = (x_1, x_2, x_1 * x_2)^T

        Args:
        - X (ndarray (shape: (N, 2))): A Nx2 matrix consisting N 2-dimensional inputs.

        Output:
        - X_mapped (ndarray (shape: (N, 3))): A Nx3 matrix consisting N 3-dimensional vectors corresponding 
                                              to the outputs of the feature map applied on the inputs X.
    """
    assert X.shape[1] == 2, f"This feature map only applies to 2D inputs. Got: {X.shape[1]}"
    # ====================================================
    # TODO: Implement your non-linear-map here

    X_mapped = np.zeros(shape=(X.shape[0], 3), dtype=np.float)

    for i in range(X.shape[0]):
        X_mapped[i] = hstack((X[i], np.array([X[i, 0] * X[i, 1]])))

    # ====================================================

    return X_mapped


if __name__ == "__main__":
    seed = 0
    np.random.seed(seed)

    # Support generic_1, generic_2, generic_3, wine
    dataset = "generic_2"

    assert dataset in ("generic_1", "generic_2", "generic_3",
                       "wine"), f"Invalid dataset: {dataset}"

    dataset_path = f"./datasets/{dataset}.pkl"
    data = load_pickle_dataset(dataset_path)

    train_X = data['train_X']
    train_y = data['train_y']
    test_X = test_y = None
    test_X = test_y = None

    if 'test_X' in data and 'test_y' in data:
        test_X = data['test_X']
        test_y = data['test_y']

    # ====================================================
    # Hyperparameters
    # NOTE: This is definitely not the best way to pass all your hyperparameters.
    #       We can usually use a configuration file to specify these.
    # ====================================================
    factor = 1
    bias = 0
    alpha_inverse = 1
    beta_inverse = 10
    num_epochs = 1000
    step_size = 1e-3
    apply_data_preprocessing = True
    check_grad = True
    verbose = False

    def data_preprocessing(X): return X
    if apply_data_preprocessing:
        data_preprocessing = feature_map

    train(train_X=train_X,
          train_y=train_y,
          test_X=test_X,
          test_y=test_y,
          data_preprocessing=data_preprocessing,
          factor=factor,
          bias=bias,
          alpha_inverse=alpha_inverse,
          beta_inverse=beta_inverse,
          num_epochs=num_epochs,
          step_size=step_size,
          check_grad=check_grad,
          verbose=verbose)
