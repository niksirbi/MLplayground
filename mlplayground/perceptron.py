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
        if not isinstance(learning_rate, (int, float)):
            raise ValueError("learning_rate must be a number")
        learning_rate = float(learning_rate)
        if learning_rate <= 0 or learning_rate > 1:
            raise ValueError("learning_rate must be in (0, 1]")
        self.learning_rate = learning_rate

        if not isinstance(max_iter, int) or max_iter <= 0:
            raise ValueError("max_iter must be a positive integer")
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
        X = self._add_bias(X)
        self.weights = np.zeros(X.shape[1])  # Initialize weights to zeros
        print("Initial weights: ", self.weights)
        self.errors = []

        for i in range(self.max_iter):
            print(f"Iteration {i+1}:")
            errors = 0  # Number of misclassifications in this iteration

            for xi, yi in zip(X, y):
                # Update weights if there is a misclassification
                if yi * np.dot(self.weights.T, xi) <= 0:
                    self.weights += self.learning_rate * yi * xi
                    print("\tUpdated weights: ", self.weights)
                    errors += 1
            self.errors.append(errors)

            # If there are no errors after a full iteration, stop training
            if errors == 0:
                print("\tNo errors. Stopping training.")
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
        self._validate_data(X, None)
        X = self._add_bias(X)
        return np.sign(np.dot(X, self.weights))

    @property
    def hyperplane(self):
        """Return the parameters (slope, intercept) of the hyperplane
        separating the two classes.
        """
        if not hasattr(self, "weights"):
            raise AttributeError("Model has not been trained yet.")
        w1, w2, bias = self.weights
        intercept = -bias / w2
        slope = -w1 / w2
        return slope, intercept

    def _validate_data(
        self, X: Optional[np.ndarray], y: Optional[np.ndarray] = None
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
            if not np.array_equal(np.unique(y), np.array([-1, 1])):
                raise ValueError("y must contain only -1 and 1")

        if (X is not None) and (y is not None):
            if X.shape[0] != y.shape[0]:
                raise ValueError(
                    "X and y must have the same number of samples"
                )

    def _add_bias(self, X: np.ndarray) -> np.ndarray:
        """Add bias term to the input data.

        The bias is a column of ones appended to the input data."""
        return np.hstack([X, np.ones((X.shape[0], 1))])
