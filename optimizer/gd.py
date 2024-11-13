import numpy as np


def compute_loss(a, b, X, y):
    theta = np.array([[a], [b]])
    m = len(y)  # X.shape[0]

    one_vector = np.ones((m, 1))
    X_prime = np.hstack((X, one_vector))

    return (1 / m) * np.sum((y - X_prime.dot(theta)) ** 2)


def compute_gradients(a, b, X, y):
    """
    Gradient
    `gradient = -(1 / m) * np.dot(X_prime.T, y - X_prime.dot(theta))`

    Gradient Descent
    `theta_new = theta - learning_rate * gradient`
    """

    theta = np.array([[a], [b]])
    m = len(y)  # X.shape[0]

    one_vector = np.ones((m, 1))
    X_prime = np.hstack((X, one_vector))

    gradient = -(1 / m) * np.dot(X_prime.T, y - X_prime.dot(theta))

    da, db = gradient[0, 0], gradient[1, 0]

    return da, db


def gradient_descent(X, y, start_a, start_b, learning_rate, n_iterations=50):
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("X와 y는 numpy array여야 합니다.")

    if X.shape[0] != y.shape[0] or y.shape[1] != 1:
        raise ValueError(
            f"입력 X의 shape은 (m, n), 입력 y의 shape은 (m, 1)이어야 합니다.\n실제 입력 X의 shape: {X.shape}, 입력 y의 shape: {y.shape}")

    a, b = start_a, start_b
    path = [(a, b, compute_loss(a, b, X, y))]

    for i in range(n_iterations):
        da, db = compute_gradients(a, b, X, y)
        a -= learning_rate * da
        b -= learning_rate * db
        path.append((a, b, compute_loss(a, b, X, y)))

    return a, b, np.array(path)
