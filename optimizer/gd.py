import numpy as np


def compute_loss(a, b, X, y):
    m = len(y)
    predictions = a * X + b
    return (1 / (2 * m)) * np.sum((predictions - y) ** 2)


def compute_gradients(a, b, X, y):
    m = len(y)
    da = (1 / m) * np.sum((a * X + b - y) * X)
    db = (1 / m) * np.sum(a * X + b - y)
    return da, db


def gradient_descent(X, y, start_a, start_b, learning_rate, n_iterations):
    a, b = start_a, start_b
    path = [(a, b, compute_loss(a, b, X, y))]

    for i in range(n_iterations):
        da, db = compute_gradients(a, b, X, y)
        a -= learning_rate * da
        b -= learning_rate * db
        path.append((a, b, compute_loss(a, b, X, y)))

    return a, b, np.array(path)
