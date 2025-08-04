import numpy as np
import matplotlib.pyplot as plt

def plot_data(X, Y):
    """
    Scatter plot for 2D data with binary labels.
    """
    Y_flat = Y.ravel()  # Ensure 1D shape for indexing
    plt.figure(figsize=(8, 6))
    plt.scatter(X[Y_flat == 0, 0], X[Y_flat == 0, 1], c='red', edgecolor='k', label='Class 0')
    plt.scatter(X[Y_flat == 1, 0], X[Y_flat == 1, 1], c='blue', edgecolor='k', label='Class 1')
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.grid(True)
    plt.title("Data Points")
    plt.show()


def plot_decision_boundary(model, X, Y, resolution=0.01, use_prob=False):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, resolution),
        np.arange(y_min, y_max, resolution)
    )

    grid_points = np.c_[xx.ravel(), yy.ravel()]
    try:
        Z = model(grid_points)
    except Exception as e:
        print("Error in model prediction:", e)
        return

    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    
    if use_prob:
        # Probability heatmap
        contour = plt.contourf(xx, yy, Z, levels=50, cmap='coolwarm', alpha=0.8)
        plt.colorbar(contour, label="Predicted Probability")
    else:
        # Binary decision regions with dark colors
        plt.contourf(xx, yy, Z, levels=[0, 0.5, 1], colors=['#FF5555', '#5555FF'], alpha=0.6)
        # Optional: draw crisp black boundary line
        plt.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)

    # Data points
    Y_flat = Y.ravel()
    plt.scatter(X[Y_flat == 0, 0], X[Y_flat == 0, 1], color='red', edgecolor='k', label='Class 0')
    plt.scatter(X[Y_flat == 1, 0], X[Y_flat == 1, 1], color='blue', edgecolor='k', label='Class 1')

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Decision Boundary")
    plt.legend()
    plt.grid(True)
    plt.show()
