"""
CSCC11 - Introduction to Machine Learning, Fall 2021, Assignment 2
M. Ataei
"""

import numpy as np

from utils import softmax


class LogisticRegression:
    def __init__(self,
                 num_features,
                 num_classes,
                 rng=np.random):
        """ This class represents a multinomial logistic regression model.
        NOTE: We assume lables are 0 to K - 1, where K is number of classes.

        self.parameters contains the model weights.
        NOTE: Bias term is the first term

        TODO: You will need to implement the methods of this class:
        - _compute_loss_and_gradient: ndarray, ndarray -> float, ndarray

        Implementation description will be provided under each method.

        For the following:
        - N: Number of samples.
        - D: Dimension of input features.
        - K: Number of classes.

        Args:
        - num_features (int): The number of features in the input data.
        - num_classes (int): The number of classes in the task.
        - rng (RandomState): The random number generator to initialize weights.
        """
        self.num_features = num_features
        self.num_classes = num_classes
        self.rng = rng

        # Initialize parameters
        self.parameters = np.zeros(shape=(num_classes, self.num_features + 1))

    def init_weights(self, factor=1, bias=0):
        """ This randomly initialize the model weights.

        Args:
        - factor (float): A constant factor of the randomly initialized weights.
        - bias (float): The bias value
        """
        self.parameters[:, 1:] = factor * \
            self.rng.rand(self.num_classes, self.num_features)
        self.parameters[:, 0] = bias

    def _compute_loss_and_gradient(self, X, y, alpha_inverse=0, beta_inverse=0):
        """ This computes the negative log likelihood (NLL) or negative log posterior and its gradient.

        NOTE: When we have alpha_inverse != 0 or beta_inverse != 0, we have negative log posterior (NLP) instead.
        NOTE: For the L2 term, drop all the log constant terms and cosntant factor.
              For the NLL term, divide by the number of data points (i.e. we are taking the mean).
              The new loss should take the form:
                  E_new(w) = (NLL_term / N) + L2_term
        NOTE: Compute the gradient based on the modified loss E_new(w)

        Args:
        - X (ndarray (shape: (N, D))): A NxD matrix consisting N D-dimensional inputs.
        - y (ndarray (shape: (N, 1))): A N-column vector consisting N scalar outputs (labels).
        - alpha_inverse (float): 1 / variance for an optional isotropic Gaussian  prior (for the weights) on NLP.
                                NOTE: 0 <= alpha_inverse. Setting alpha_inverse to 0 means no prior on weights.
        - beta_inverse (float): 1 / variance for an optional Gaussian prior (for the bias term) on NLP.
                                NOTE: 0 <= beta_inverse. Setting beta_inverse to 0 means no prior on the bias term.

        Output:
        - nll (float): The NLL (or NLP) of the given inputs and outputs.
        - grad (ndarray (shape: (K, D + 1))): A Kx(D + 1) weight matrix (including bias) consisting the gradient of NLL (or NLP)
                                              (i.e. partial derivatives of NLL (or NLP) w.r.t. self.parameters).
        """
        (N, D) = X.shape
        # ====================================================
        # TODO: Implement your solution within the box

        probabilities = self.predict(X)
        grad = np.zeros(shape=(self.num_classes, D+1))

        # compute NLL
        nll = 0
        for i in range(N):
            nll -= np.log(probabilities[i, y[i]])
        nll = nll / N

        # compute gradient of NLL WRT w_k
        for k in range(self.num_classes):
            for i in range(N):
                if y[i] == k:
                    grad[k] -= (np.hstack((np.array([1], dtype=np.float),
                                           X[i])) * (1 - probabilities[i, k]))
                else:
                    grad[k] += (np.hstack((np.array([1], dtype=np.float), X[i]))
                                * probabilities[i, k])
            grad[k] = grad[k] / N

        # compute L2 term and gradient of L2 term for NLP
        if not alpha_inverse == 0 and not beta_inverse == 0:
            covInvDiag = np.hstack(
                (np.array([beta_inverse]), np.full(D, alpha_inverse)))
            covInv = np.diag(covInvDiag)
            l2 = np.trace(self.parameters @ covInv @ self.parameters.T)
            nll = nll + l2

            for k in range(self.num_classes):
                grad[k] += np.reshape((covInv @ np.reshape(self.parameters[k],
                                                           newshape=(D+1, 1))), newshape=D+1)

        # ====================================================

        return nll, grad

    def learn(self,
              train_X,
              train_y,
              num_epochs=1000,
              step_size=1e-3,
              check_grad=False,
              verbose=False,
              alpha_inverse=0,
              beta_inverse=0,
              eps=np.finfo(np.float).eps):
        """ This performs gradient descent to learn the parameters given the training data.

        NOTE: This method mutates self.parameters

        Args:
        - train_X (ndarray (shape: (N, D))): A NxD matrix consisting N D-dimensional training inputs.
        - train_y (ndarray (shape: (N, 1))): A N-column vector consisting N scalar training outputs (labels).
        - num_epochs (int): Number of gradient descent steps
                        NOTE: 1 <= num_epochs
        - step_size (float): Gradient descent step size
        - check_grad (bool): Whether or not to check gradient using finite difference.
        - verbose (bool): Whether or not to print gradient information for every step.
        - alpha_inverse (float): 1 / variance for an optional isotropic Gaussian  prior (for the weights) on NLL.
                                NOTE: 0 <= alpha_inverse. Setting alpha_inverse to 0 means no prior on weights.
        - beta_inverse (float): 1 / variance for an optional Gaussian prior (for the bias term) on NLL.
                                NOTE: 0 <= beta_inverse. Setting beta_inverse to 0 means no prior on the bias term.
        - eps (float): Machine epsilon

        ASIDE: The design for applying gradient descent to find local minimum is usually different from this.
               You should think about a better way to do this! Scipy is a good reference for such design.
        """
        assert len(train_X.shape) == len(
            train_y.shape) == 2, f"Input/output pairs must be 2D-arrays. train_X: {train_X.shape}, train_y: {train_y.shape}"
        (N, D) = train_X.shape
        assert N == train_y.shape[
            0], f"Number of samples must match for input/output pairs. train_X: {N}, train_y: {train_y.shape[0]}"
        assert D == self.num_features, f"Expected {self.num_features} features. Got: {D}"
        assert train_y.shape[1] == 1, f"train_Y must be a column vector. Got: {train_y.shape}"
        assert 1 <= num_epochs, f"Must take at least 1 gradient step. Got: {num_epochs}"

        nll, grad = self._compute_loss_and_gradient(
            train_X, train_y, alpha_inverse, beta_inverse)

        # Check gradient using finite difference
        if check_grad:
            original_parameters = np.copy(self.parameters)
            grad_approx = np.zeros(
                shape=(self.num_classes, self.num_features + 1))
            h = 1e-8

            # Compute finite difference w.r.t. each weight vector component
            for ii in range(self.num_classes):
                for jj in range(self.num_features + 1):
                    self.parameters = np.copy(original_parameters)
                    self.parameters[ii][jj] += h
                    grad_approx[ii][jj] = (self._compute_loss_and_gradient(
                        train_X, train_y, alpha_inverse, beta_inverse)[0] - nll) / h

            # Reset parameters back to original
            self.parameters = np.copy(original_parameters)

            print(f"Negative Log Probability: {nll}")
            print(f"Analytic Gradient: {grad.T}")
            print(f"Numerical Gradient: {grad_approx.T}")
            print("The gradients should be nearly identical.")

        # Perform gradient descent
        for epoch_i in range(num_epochs):
            original_parameters = np.copy(self.parameters)
            # Check gradient flow
            if np.linalg.norm(grad) < eps:
                print(
                    f"Gradient is close to 0: {eps}. Terminating gradient descent.")
                break

            # Determine the suitable step size.
            step_size *= 2
            self.parameters = original_parameters - step_size * grad
            E_new, grad_new = self._compute_loss_and_gradient(
                train_X, train_y, alpha_inverse, beta_inverse)
            assert np.isfinite(E_new), f"Error is NaN/Inf"

            while E_new >= nll and step_size > 0:
                step_size /= 2
                self.parameters = original_parameters - step_size * grad
                E_new, grad_new = self._compute_loss_and_gradient(
                    train_X, train_y, alpha_inverse, beta_inverse)
                assert np.isfinite(E_new), f"Error is NaN/Inf"

            if step_size <= eps:
                print(
                    f"Infinitesimal step: {step_size}. Terminating gradient descent.")
                break

            if verbose:
                print(
                    f"Epoch: {epoch_i}, Step size: {step_size}, Gradient Norm: {np.linalg.norm(grad)}, NLL: {nll}")

            # Update next loss and next gradient
            grad = grad_new
            nll = E_new

    def predict(self, X):
        """ This computes the probability of labels given X.

        Args:
        - X (ndarray (shape: (N, D))): A NxD matrix consisting N D-dimensional inputs.

        Output:
        - probs (ndarray (shape: (N, K))): A NxK matrix consisting N K-probabilities for each input.
        """
        (N, D) = X.shape
        assert D == self.num_features, f"Expected {self.num_features} features. Got: {D}"

        # Pad 1's for bias term
        X = np.hstack((np.ones(shape=(N, 1), dtype=np.float), X))

        # This receives the probabilities of class 1 given inputs
        probs = softmax(X @ self.parameters.T)
        return probs
