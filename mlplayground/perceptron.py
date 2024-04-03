"""A pure numpy implementation of perceptron learning algorithm."""

from typing import Optional

import numpy as np


class Perceptron:
    """Perceptron binary classifier."""

    def __init__(self, learning_rate: float = 1.0, max_iter: int = 100):
        """Initialize the perceptron object.

        Parameters
        ----------
        learning_rate : float, default=1.0
            The learning rate (between 0.0 and 1.0).
        max_iter : int, default=100
            The maximum number of iterations over the training dataset.
        """
        if learning_rate <= 0 or learning_rate > 1:
            raise ValueError("learning_rate must be in (0, 1]")
        self.learning_rate = learning_rate

        if not isinstance(max_iter, int) or max_iter <= 0:
            raise ValueError("n_iter must be a positive integer")
        self.max_iter = max_iter

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Perceptron":
        """Fit the model to the training data.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training data.
        y : array-like, shape = [n_samples]
            Target values. This should be a binary class {-1, 1}.
        """

        self._validate_data(X, y)
        X = np.c_[np.ones(X.shape[0]), X]  # Add bias term to the input data
        self.weights = np.zeros(X.shape[1])  # Initialize weights to zeros
        self.weights_history = [self.weights]
        self.errors = []

        for _ in range(self.max_iter):
            errors = 0  # Number of misclassifications in this iteration
            for xi, yi in zip(X, y):
                if yi * np.dot(self.weights.T, xi) <= 0:
                    #  If the prediction is wrong, update the weights
                    self.weights += self.learning_rate * yi * xi
                    errors += 1
            self.errors.append(errors)
            self.weights_history.append(self.weights)
            # If there are no errors after a full iteration, stop training
            if errors == 0:
                break

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for samples in X.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Samples.

        Returns
        -------
        array, shape = [n_samples]
            Predicted class label per sample {-1, 1}.
        """

        predicted = np.dot(X, self.weights[1:]) + self.weights[0]
        return np.where(predicted > 0, 1, -1)

    def _validate_data(
        self, X: Optional[np.ndarray], y: Optional[np.ndarray]
    ) -> None:
        """Validate input data."""
        if X is not None:
            if not isinstance(X, np.ndarray):
                raise ValueError("X must be a numpy array")
            if X.ndim != 2:
                raise ValueError("X must be a 2D array")

        if y is not None:
            if not isinstance(y, np.ndarray):
                raise ValueError("y must be a numpy array")
            if y.ndim != 1:
                raise ValueError("y must be a 1D array")

        if (X is not None) and (y is not None):
            if X.shape[0] != y.shape[0]:
                raise ValueError(
                    "X and y must have the same number of samples"
                )
