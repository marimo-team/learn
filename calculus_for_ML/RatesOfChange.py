import marimo

__generated_with = "0.13.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from sympy import symbols
    from sympy import diff
    return diff, mo, np, plt, symbols


@app.cell
def _(mo):
    mo.md(
        r"""
        In this notebook, we'll learn about rates of change in Machine Learning. 

        In simple terms, the rate of change measures how one quantity changes as we adjust its inputs or parameters. The rate of change can represent how a model's output changes as we adjust the input or parameters. 

        In calculus, the rate of change of a function is given by calculating the derivative at a certain point. So, if f(x) is a function, then f'(x) is the rate of change. 

        In the context of Machine Learning, rates of change are especially useful when working with optimization problems - for example, minimizing a loss function, or using gradient descent to try and find optimal parameters and minimize error.

        Let's get into rates of change!
        """
    )
    return


@app.cell
def _(np, plt):
    # First, let's start by definiting a simple function. 

    def f(x):
        return x ** 2

    # Now, we can plot the function.abs

    x = np.linspace(-10, 10, 500)
    y = f(x)

    plt.plot(x, y, label=r'$f(x) = x^2$')
    plt.title('Graph of f(x) = x^2')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.axhline(0, color='black',linewidth=0.5)
    plt.axvline(0, color='black',linewidth=0.5)
    plt.legend()
    plt.grid(True)
    plt.show()
    return f, x, y


@app.cell
def _(mo):
    mo.md(r"""Our graph is a parabola. Now, let's find the derivative.""")
    return


@app.cell
def _(plt, x):
    def deriv_f(x):
        return 2 * x

    deriv_y = deriv_f(x)

    plt.plot(x, deriv_y, label=r"$f'(x) = 2x$", color='red')
    plt.title('Derivative of f(x) = x^2')
    plt.xlabel('x')
    plt.ylabel("f'(x)")
    plt.axhline(0, color='black',linewidth=0.5)
    plt.axvline(0, color='black',linewidth=0.5)
    plt.legend()
    plt.grid(True)
    plt.show()
    return deriv_f, deriv_y


@app.cell
def _(mo):
    mo.md(r"""We can see that the rate of change, or the slope, is positive when x is also positive. This should all be review from high school math! :)""")
    return


@app.cell
def _(diff, symbols):
    # Now, we'll use SymPy, a library for symbolic math functions in Python.
    # Here we've created a function and then found the derivative of it.

    x_sym = symbols('x')
    f_sym = x_sym ** 3 + 4 * x_sym + 5

    f_deriv_sym = diff(f_sym, x_sym)
    f_deriv_sym
    return f_deriv_sym, f_sym, x_sym


@app.cell
def _(mo):
    mo.md(r"""Now, let's get into gradient descent. Below is an example on a quadratic function.""")
    return


@app.cell
def _(deriv_f, f, np, plt):
    # Our parameters below

    learning_rate = 0.1
    initial = 8
    iterations = 20

    current = initial
    history = [current]

    for i in range(iterations):
        grad = deriv_f(current)
        current = current - learning_rate * grad
        history.append(current)

    x_vals = np.linspace(-10, 10, 400)
    y_vals = f(x_vals)

    plt.plot(x_vals, y_vals, label=r'$f(x) = x^2$')
    plt.scatter(history, f(np.array(history)), color='red', label='Gradient Descent')
    plt.title('Gradient Descent on f(x) = x^2')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.grid(True)
    plt.show()
    return (
        current,
        grad,
        history,
        i,
        initial,
        iterations,
        learning_rate,
        x_vals,
        y_vals,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
        What's happening here?

        At a high level, gradient descent is essentially taking the partial derivative at each point in the function, until it converges to a minimum. Because our f(x) is a 'nice' function (i.e., there are no annoying curves, saddle points, or other minima/maxima), we are guaranteed that our gradient descent can find the minimum. 

        What is the minimum in this case? It's the **point on the function where the rate of change is closest to 0**.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        Now, we can get into some applications of rates of change to fully understand the concept.

        First, let's look at kinematics. We know that position, velocity, and acceleration are related, but how? Interestingly, **the derivative of position is velocity, and the derivative of velocity is acceleration!**
        """
    )
    return


@app.cell
def _(np, plt):
    # Let's say we have a position function of an object that is 5t^2, where t is time.

    time = np.linspace(0, 10, 100)

    position = 5 * time**2

    velocity = 10 * time

    acceleration = np.full_like(time, 10)

    plt.plot(time, position, label='Position (s(t) = 5t^2)')
    plt.plot(time, velocity, label='Velocity (v(t) = 10t)', linestyle='--')
    plt.plot(time, acceleration, label='Acceleration (a(t) = 10)', linestyle=':')
    plt.title('Position, Velocity, and Acceleration')
    plt.xlabel('Time (t)')
    plt.ylabel('Position / Velocity / Acceleration')
    plt.legend()
    plt.grid(True)
    plt.show()
    return acceleration, position, time, velocity


@app.cell
def _(mo):
    mo.md(r"""Here, we can see that the orange line, velocity, represents the rate of change of position at the exact time t = 2 seconds. When we look at the green line, it tell us the rate of change of velocity  at that exact same point in time!""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        We can use another example in economics. In economics, there is a term known as 'Price Sensitivity,' wjich essentially measures the rate of change in the quantity demanded or supplied of a good. The price elasticity is given by the formula:

        Elasticity = % change in qty. demanded / % change in price

        Using derivatives, we would say:

        Elasticity = the rate of change of demand with respect to price * (price / quantity)
        """
    )
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
