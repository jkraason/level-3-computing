import numpy as np
import matplotlib.pyplot as plt
def simplecubic(dim):
    #dim = float(input("enter number of unit cells across:"))
    x_coords = np.arange(0, dim+1, 1) 
    y_coords = np.arange(0, dim+1, 1) 

    # Create the 2D coordinate arrays using meshgrid
    X, Y = np.meshgrid(x_coords, y_coords)

    # You can now use X and Y directly, for example, to plot the lattice
    print("X coordinates:\n", X)
    print("\nY coordinates:\n", Y)

    # For a simple list of (x, y) coordinate pairs, you can flatten the arrays
    points = np.vstack([X.ravel(), Y.ravel()]).T

    print("\nFlattened list of points (x, y):")
    print(points)

    # Visualize the lattice using Matplotlib
    plt.figure(figsize=(6, 6))
    plt.scatter(X, Y, marker='o', color='blue')
    plt.title("2D Simple Cubic Lattice")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.grid(True, linestyle='--')
    plt.show()
simplecubic(2)