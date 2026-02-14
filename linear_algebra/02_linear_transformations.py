import marimo

__generated_with = "0.15.5"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # 2. Linear Transformations: Intuition & Examples

        Linear transformations are operations that move, rotate, scale, or shear vectors and shapes in space. They are fundamental in machine learning for manipulating data and features.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Example 1: Rotating a Vector 0001F504
        - **Original vector:** $v = [2, 1]$
        - **Rotation:** 45° counterclockwise
        - **Transformation matrix:**
          $$ R = egin{bmatrix} os	heta & -in	heta \ in	heta & os	heta nd{bmatrix} $$
        - **Result:** Vector is rotated in space
        """
    )
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt

    v = np.array([2, 1])
    theta = np.pi / 4
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    v_rot = R @ v

    plt.figure(figsize=(6,6))
    plt.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, color='r', label='Original')
    plt.quiver(0, 0, v_rot[0], v_rot[1], angles='xy', scale_units='xy', scale=1, color='b', label='Rotated')
    plt.xlim(-1, 3)
    plt.ylim(-1, 3)
    plt.grid(True)
    plt.legend()
    plt.title('Rotation of a Vector')
    plt.show()
    return np, plt, v


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Result:** The vector is rotated by 45°.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Example 2: Scaling a Vector 0001F4A1
        - **Original vector:** $v = [2, 1]$
        - **Scaling matrix:**
          $$ S = egin{bmatrix} 2 & 0 \ 0 & 0.5 nd{bmatrix} $$
        - **Result:** Vector is stretched in $x$ and compressed in $y$
        """
    )
    return


@app.cell
def _(np, plt, v):
    S = np.array([[2, 0], [0, 0.5]])
    v_scale = S @ v

    plt.figure(figsize=(6,6))
    plt.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, color='r', label='Original')
    plt.quiver(0, 0, v_scale[0], v_scale[1], angles='xy', scale_units='xy', scale=1, color='g', label='Scaled')
    plt.xlim(-1, 5)
    plt.ylim(-1, 2)
    plt.grid(True)
    plt.legend()
    plt.title('Scaling a Vector')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Result:** The vector is stretched horizontally and compressed vertically.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Example 3: Shearing a Vector 0001F4A5
        - **Original vector:** $v = [2, 1]$
        - **Shearing matrix:**
          $$ H = egin{bmatrix} 1 & 1.2 \ 0 & 1 nd{bmatrix} $$
        - **Result:** Vector is slanted horizontally
        """
    )
    return


@app.cell
def _(np, plt, v):
    H = np.array([[1, 1.2], [0, 1]])
    v_shear = H @ v

    plt.figure(figsize=(6,6))
    plt.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, color='r', label='Original')
    plt.quiver(0, 0, v_shear[0], v_shear[1], angles='xy', scale_units='xy', scale=1, color='m', label='Sheared')
    plt.xlim(-1, 5)
    plt.ylim(-1, 3)
    plt.grid(True)
    plt.legend()
    plt.title('Shearing a Vector')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Result:** The vector is slanted horizontally.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Why are Linear Transformations Important in ML?
        - They help manipulate and preprocess data
        - Used in feature engineering, PCA, neural networks, and more
        - Understanding them builds intuition for how ML algorithms work
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
