"""
Perceptron Binary Classifier
============================

Performs binary classification using the Perceptron algorithm.
"""

# %%
# Imports
# -------
from typing import Optional

import numpy as np
from matplotlib import pyplot as plt

from mlplayground import Perceptron

# %%
# Generate some example training data
# ---------------------------------------

X_train = np.array([[10, -2], [12, 2]])
y_train = np.array([1, -1])

print("Training data:")
print(X_train)
print("\nLabels:")
print(y_train)

# %%
# Define a function to plot the data
# ----------------------------------


def plot_data(X, y, title, model: Optional[Perceptron]) -> None:
    fig, ax = plt.subplots()
    ax.scatter(X[y == 1, 0], X[y == 1, 1], label="Class 1")
    ax.scatter(X[y == -1, 0], X[y == -1, 1], label="Class -1")
    ax.set_title(title)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")

    if model is not None and hasattr(model, "weights"):
        min_x = X_train[:, 0].min()
        max_x = X_train[:, 0].max()
        # Draw the hyperplane (decision boundary)
        x = np.linspace(min_x, max_x, 100)
        slope, intercept = model.hyperplane
        y = slope * x + intercept
        ax.plot(x, y, color="red", label="Hyperplane")

    ax.legend()
    fig.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.95)


# %%
# Plot the training data
# ----------------------

plot_data(
    X_train,
    y_train,
    model=None,
    title="Training data",
)

# %%
# Train the Perceptron
# --------------------

perceptron = Perceptron(learning_rate=1.0, max_iter=100)
perceptron.fit(X_train, y_train)

# %%
# Plot the decision boundary
# --------------------------

plot_data(
    X_train,
    y_train,
    model=perceptron,
    title="Decision boundary after training",
)
