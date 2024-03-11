import autograd.numpy as np
from autograd import grad
from matplotlib import pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import time


def plot_q_3D(x_vals, y_vals, alpha, n=10):
    x = np.linspace(-100, 100, 10)
    X, Y = np.meshgrid(x, x, indexing="ij", sparse=True)
    Z = np.array([[q(x0, alpha, n) for x0 in row] for row in X])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.5)

    ax.scatter(x_vals, list(range(len(x_vals))), y_vals, color="red")

    # ax.set_ylim(0, 100)

    ax.set_xlabel("X")
    ax.set_ylabel("Iteration")
    ax.set_zlabel("Function value")
    plt.title("3D plot of the function q(x) and the trajectory of convergence")
    plt.show()


def q(x, alpha=1, n=10):
    i = np.arange(1, n + 1)
    sum = np.sum((alpha ** ((i - 1) / (n - 1))) * (x**2))
    return sum


def plot_q(x_vals, y_vals, alpha=1, n=10):
    x = np.linspace(-100, 100, 100)
    x = np.array([x for _ in range(n)])
    y = np.array([[q(x0, alpha, n) for x0 in row] for row in x])

    for i in range(n):
        plt.plot(x[i], y[i])

    plt.scatter(x_vals[-1], y_vals[-1], color="red")
    plt.xlabel("x")
    plt.ylabel("q(x)")
    plt.title("Plot of the function q(x)")
    plt.grid(True)
    plt.show()


def test_gradient_descent(learning_rates, x0, q, max_iterations, alpha=1, n=10):
    for learning_rate in learning_rates:
        start_time = time.time()
        x_vals, y_vals = gradient_descent(
            q, x0, learning_rate, max_iterations, alpha, n
        )
        end_time = time.time()
        print(x_vals[-1], y_vals[-1])
        plot_q(x_vals, y_vals, alpha, n)
    return end_time - start_time


def plot_learning_rate(learning_rates, x0, q, num_iterations, alpha=1, n=10):
    for learning_rate in learning_rates:
        _, y_vals = gradient_descent(q, x0, learning_rate, num_iterations, alpha, n)
        plt.plot(y_vals, label=f"learning rate = {learning_rate}")
    plt.xlabel("Iteration")
    plt.ylabel("Function value")
    plt.title("Plot of the function q(x) for different learning rates")
    # plt.yscale("log")
    plt.legend(
        loc="upper right",
        fontsize="x-small",
        title="Learning rate",
        title_fontsize="small",
        shadow=True,
        fancybox=True,
        frameon=True,
    )
    plt.show()
    return


def gradient_descent(
    func, x0, learning_rate=0.001, max_iterations=1000, tolerance=1e-7, alpha=1, n=10
):
    x = x0
    gradient_func = grad(func)
    x_values = []
    func_values = []

    for _ in range(max_iterations):
        gradient = gradient_func(x)
        x = x - learning_rate * gradient

        x_values.append(x)
        func_value = func(x, alpha, n)
        func_values.append(func_value)
        if gradient < tolerance:
            break
    return x_values, func_values


def main():
    x0 = 90.0
    alphas = [1, 10, 100]
    n = 10

    for alpha in alphas:
        x, y = gradient_descent(
            q,
            x0,
            learning_rate=0.01,
            max_iterations=100,
            alpha=alpha,
            n=n,
            tolerance=1e-6,
        )
        print(x[-1], y[-1])

    learning_rates = [0.00001, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
    for alpha in alphas:
        plot_q_3D(*gradient_descent(q, x0, 0.001, 100, alpha, n), alpha=alpha)
    return


if __name__ == "__main__":
    main()
