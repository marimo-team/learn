import marimo

__generated_with = "0.15.5"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Linear Algebra Foundations for Machine Learning

    Welcome! This notebook introduces the essential concepts of linear algebra needed for machine learning, with a focus on visualization and geometric intuition.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Topics Covered
    - Vectors and vector operations
    - Matrices and matrix operations
    - Visualizing vectors and matrices
    - Why these concepts matter in ML
    """
    )
    return


@app.cell
def _():
    # Import required libraries
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    return np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Vectors: Definition and Visualization
    A vector is an ordered list of numbers, representing a point or direction in space.
    """
    )
    return


@app.cell
def _(np, plt):
    # Define two vectors in 2D
    v1 = np.array([2, 3])
    v2 = np.array([4, 1])

    # Plot the vectors
    plt.figure(figsize=(6,6))
    plt.quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy', scale=1, color='r', label='v1')
    plt.quiver(0, 0, v2[0], v2[1], angles='xy', scale_units='xy', scale=1, color='b', label='v2')
    plt.xlim(-1, 5)
    plt.ylim(-1, 5)
    plt.grid(True)
    plt.legend()
    plt.title('2D Vectors')
    plt.show()
    return v1, v2


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Vector Operations
    Let's add two vectors and visualize the result.
    """
    )
    return


@app.cell
def _(plt, v1, v2):
    # Vector addition
    v_sum = v1 + v2

    plt.figure(figsize=(6,6))
    plt.quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy', scale=1, color='r', label='v1')
    plt.quiver(0, 0, v2[0], v2[1], angles='xy', scale_units='xy', scale=1, color='b', label='v2')
    plt.quiver(0, 0, v_sum[0], v_sum[1], angles='xy', scale_units='xy', scale=1, color='g', label='v1 + v2')
    plt.xlim(-1, 7)
    plt.ylim(-1, 7)
    plt.grid(True)
    plt.legend()
    plt.title('Vector Addition')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Matrices: Definition and Visualization
    A matrix is a rectangular array of numbers. In ML, matrices often represent datasets or transformations.
    """
    )
    return


@app.cell
def _(np):
    # Define a matrix
    A = np.array([[1, 2], [3, 4]])
    print('Matrix A:')
    print(A)
    return (A,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Matrix-Vector Multiplication
    Matrix multiplication can be seen as a transformation of a vector.
    """
    )
    return


@app.cell
def _(A, v1):
    # Apply matrix A to vector v1
    v1_transformed = A @ v1
    print('A @ v1 =', v1_transformed)
    return (v1_transformed,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Visualizing Matrix Transformation
    Let's see how matrix A transforms vector v1.
    """
    )
    return


@app.cell
def _(plt, v1, v1_transformed):
    plt.figure(figsize=(6,6))
    plt.quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy', scale=1, color='r', label='Original v1')
    plt.quiver(0, 0, v1_transformed[0], v1_transformed[1], angles='xy', scale_units='xy', scale=1, color='m', label='Transformed v1')
    plt.xlim(-1, 10)
    plt.ylim(-1, 10)
    plt.grid(True)
    plt.legend()
    plt.title('Matrix Transformation')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Why Linear Algebra Matters in ML
    - Data is often represented as vectors/matrices
    - Transformations (e.g., PCA, neural networks) use matrix operations
    - Understanding these basics helps in grasping ML algorithms
    """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
