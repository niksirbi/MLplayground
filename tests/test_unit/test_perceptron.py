"""Tests for the Perceptron class"""

import numpy as np
import pytest

from mlplayground import Perceptron


class TestPerceptron:

    @pytest.fixture
    def X_train(self):
        return np.array([[10, -2], [12, 2]])

    @pytest.fixture
    def y_train(self):
        return np.array([1, -1])

    @pytest.fixture
    def X_test(self):
        return np.array([[11, -2], [11, 2]])

    @pytest.fixture
    def y_test(self):
        return np.array([1, -1])

    @pytest.mark.parametrize("learning_rate", [-1, 2.0, "invalid"])
    @pytest.mark.parametrize("max_iter", ["invalid", -10])
    def test_init_invalid(self, learning_rate, max_iter):
        """Test the Perceptron class initialization with invalid inputs"""
        with pytest.raises(ValueError):
            Perceptron(learning_rate=learning_rate, max_iter=max_iter)

    @pytest.mark.parametrize("learning_rate", [0.1, 1.0])
    @pytest.mark.parametrize("max_iter", [10, 1000])
    def test_init_valid(self, learning_rate, max_iter):
        """Test the Perceptron class initialization with valid inputs"""
        parceptron = Perceptron(learning_rate=learning_rate, max_iter=max_iter)
        assert parceptron.learning_rate == learning_rate
        assert parceptron.max_iter == max_iter

    def test_fit(self, X_train, y_train):
        """Test the Perceptron class fit method"""
        perceptron = Perceptron()
        perceptron.fit(X_train, y_train)
        assert len(perceptron.errors) == 6
        assert np.allclose(perceptron.weights, np.array([2.0, -18.0, 1.0]))

    def test_predict(self, X_test, y_test):
        """Test the Perceptron class predict method"""
        perceptron = Perceptron()
        perceptron.weights = np.array([2.0, -18.0, 1.0])
        y_pred = perceptron.predict(X_test)
        assert np.allclose(y_pred, y_test)

    def test_hyperplane(self, X_train, y_train):
        """Test the Perceptron class hyperplane property"""
        perceptron = Perceptron()
        with pytest.raises(AttributeError):
            perceptron.hyperplane

        perceptron.fit(X_train, y_train)
        assert np.allclose(perceptron.hyperplane, (0.1111111, 0.05555555))

    def test_add_bias(self):
        """Test the Perceptron class _add_bias method"""
        perceptron = Perceptron()
        X = np.array([[10, -2], [12, 2]])
        X_bias = perceptron._add_bias(X)
        assert np.allclose(X_bias, np.array([[10, -2, 1], [12, 2, 1]]))

    @pytest.mark.parametrize("X_train", ["invalid", np.array([10, -2])])
    @pytest.mark.parametrize(
        "y_train", ["invalid", np.array([[1], [-1]]), np.array([1, 2])]
    )
    def test_validate_data(self, X_train, y_train):
        """Test that the Perceptron class _validate_data method raises errors
        when the input data is invalid"""
        perceptron = Perceptron()
        with pytest.raises(ValueError):
            perceptron._validate_data(X_train, y_train)
