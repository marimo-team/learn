import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Derivatives and Critical Values

    _By [ghimiresunil](https://github.com/ghimiresunil)

    Welcome to this tutorial on **Derivatives and Critical Values**—a key turning point in our exploration of calculus. In the previous module, we learned how to describe the *rate of change* of a function. In this notebook, we take that understanding further to analyze how functions behave, where they reach their highest or lowest values, and how we can use calculus to find those points.

    ---

    ## Quick Recap: Rates of change

    In the **Rates of Change** notebook, we explored two foundational ideas:

    - **Average Rate of Change**:  
      This is the change in a function’s output relative to the change in input over an interval. It gives us a *global* view of how a function behaves between two points.

    \[ \text{Average Rate of Change} = \frac{f(b) - f(a)}{b - a} \]

    - **Instantaneous Rate of Change**:  
      As the interval becomes infinitesimally small, the average rate of change approaches the **derivative**—a *local* measure of how the function is changing at a specific point.


    \[f'(x) = \lim_{h \to 0} \frac{f(x + h) - f(x)}{h} \]

    This transition from global to local change is where derivatives begin to reveal the fine structure of functions, helping us uncover turning points, slopes, and trends.

    ---

    ## What You'll Learn in This Notebook

    In this notebook, we will focus on:

    - What *critical values* are and how to find them  
    - How to use the **first derivative** to identify increasing/decreasing behavior  
    - How to apply the **second derivative** to determine concavity and locate maxima or minima  
    - How these tools help us analyze and sketch functions  
    - How this foundation prepares us for solving real-world **optimization problems**

    By the end of this module, you’ll be able to use derivatives not just to describe change, but to *predict, optimize, and explain* key behaviors in a variety of systems—from economics to physics to machine learning.

    Let’s get started...
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Rules of Differentiation

    Learn the essential rules for differentiating mathematical functions, from basic polynomials to complex compositions. Master the power, product, quotient, and chain rules, and understand how to compute higher-order derivatives used in analysis and modeling.

    | **Rule** | **What It Is** | **Where It’s Used** | **How To Use** | **Practice Questions** |
    |----------|----------------|---------------------|----------------|-------------------------|
    | **Addition & Subtraction Rule** | The derivative of a sum or difference is the sum or difference of the derivatives | When multiple terms are combined in expressions | \( \frac{d}{dx} [f(x) \pm g(x)] = f'(x) \pm g'(x) \) | 1. \( \frac{d}{dx}(x^3 + \sin x) \) <br> 2. \( \frac{d}{dx}(e^x - x^2) \) |
    | **Power Rule** | A rule for differentiating functions of the form \( x^n \) | Polynomial functions, roots, rational powers | \( \frac{d}{dx} x^n = n x^{n-1} \) | 1. \( \frac{d}{dx}(x^4) \)  <br> 2. \( \frac{d}{dx}(x^{-2}) \) <br> 3. \( \frac{d}{dx}(\sqrt{x}) \) |
    | **Product Rule** | Differentiates the product of two functions | Used in physics and economics: force × distance, price × quantity | \( \frac{d}{dx}(u \cdot v) = u'v + uv' \) | 1. \( \frac{d}{dx}(x^2 \sin x) \) <br> 2. \( \frac{d}{dx}(e^x \ln x) \) |
    | **Quotient Rule** | For differentiating one function divided by another | Rational expressions, modeling change ratios | \( \frac{d}{dx} \left( \frac{u}{v} \right) = \frac{u'v - uv'}{v^2} \) | 1. \( \frac{d}{dx} \left( \frac{x^2}{x+1} \right) \) <br> 2. \( \frac{d}{dx} \left( \frac{\sin x}{x^2} \right) \) |
    | **Chain Rule** | Used for differentiating composite (nested) functions | Growth/decay models, trig and log functions | \( \frac{d}{dx}(g(h(x))) = g'(h(x)) \cdot h'(x) \) | 1. \( \frac{d}{dx}(\sin(x^2)) \) <br> 2. \( \frac{d}{dx}(e^{3x^2 + 2}) \) <br> 3. \( \frac{d}{dx}(\ln(\sqrt{x})) \) |
    | **Higher-Order Derivatives** | Derivatives of derivatives, e.g., \( f''(x), f^{(3)}(x) \) | Physics: velocity, acceleration, concavity in graphs | Compute derivatives multiple times: \( f''(x) = \frac{d^2f}{dx^2} \) | 1. \( f(x) = x^3 - 3x + 2 \Rightarrow f''(x) \)? <br> 2. \( \frac{d^3}{dx^3}(\sin x) \)? |

    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Import Necessary Libraries""")
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import sympy as sp
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import matplotlib.pyplot as plt
    return go, mo, np, sp


@app.cell
def _(go, np, sp):
    def best_differentiation_plot():
        x = sp.Symbol('x')

        expressions = {
            "Addition Rule": sp.sympify("x**2 + sin(x)"),
            "Subtraction Rule": sp.sympify("x**3 - cos(x)"),
            "Power Rule": sp.sympify("x**4"),
            "Product Rule": sp.sympify("x**2 * sin(x)"),
            "Quotient Rule": sp.sympify("x**2 / (x + 1)"),
            "Chain Rule": sp.sympify("sin(x**2)")
        }

        x_vals = np.linspace(-10, 10, 1000)
        traces = []
        visibilities = []
        annotations = []

        for i, (rule_name, expr) in enumerate(expressions.items()):
            derivative = sp.diff(expr, x)

            # Prepare numerical functions
            f = sp.lambdify(x, expr, 'numpy')
            f_prime = sp.lambdify(x, derivative, 'numpy')

            try:
                y_vals = f(x_vals)
                y_prime_vals = f_prime(x_vals)
            except Exception as e:
                print(f"Error with {rule_name}: {e}")
                continue

            # Plot traces (function and derivative)
            func_trace = go.Scatter(
                x=x_vals, y=y_vals, mode='lines',
                line=dict(color='blue'),
                name='f(x)',
                hovertemplate="x: %{x:.2f}<br>f(x): %{y:.2f}",
                visible=(i == 0)
            )
            deriv_trace = go.Scatter(
                x=x_vals, y=y_prime_vals, mode='lines',
                line=dict(color='red', dash='dash'),
                name="f'(x)",
                hovertemplate="x: %{x:.2f}<br>f'(x): %{y:.2f}",
                visible=(i == 0)
            )

            traces.extend([func_trace, deriv_trace])

            # Visibility map
            vis = [False] * len(expressions) * 2
            vis[2 * i] = True
            vis[2 * i + 1] = True
            visibilities.append(vis)

            # Annotations in the lower right corner
            annotations.append([
                dict(
                    text=f" $f(x) = {sp.latex(expr)}$",
                    xref="paper", yref="paper",
                    x=0.95, y=0.15,
                    showarrow=False,
                    font=dict(size=14),
                    align='left',
                    bgcolor='rgba(255,255,255,0.7)'
                ),
                dict(
                    text=f"f'(x) = $f'(x) = {sp.latex(derivative)}$",
                    xref="paper", yref="paper",
                    x=0.95, y=0.05,
                    showarrow=False,
                    font=dict(size=14),
                    align='left',
                    bgcolor='rgba(255,255,255,0.7)'
                )
            ])

        # Create figure and add all traces
        fig = go.Figure(data=traces)

        # Dropdown menu for rule selection
        dropdown_buttons = []
        for i, (rule_name, vis) in enumerate(zip(expressions.keys(), visibilities)):
            dropdown_buttons.append(dict(
                label=rule_name,
                method="update",
                args=[
                    {"visible": vis},
                    {
                        "title": f"📘 {rule_name}: Function & Derivative",
                        "annotations": annotations[i]
                    }
                ]
            ))

        # Layout
        fig.update_layout(
            title="📘 Addition Rule: Function & Derivative",
            updatemenus=[
                dict(
                    buttons=dropdown_buttons,
                    active=0,
                    x=0.5,
                    y=1.18,
                    xanchor="center",
                    yanchor="top",
                    direction="down",
                    showactive=True
                )
            ],
            annotations=annotations[0],
            width=1100,
            height=600,
            template="plotly_white",
            hovermode="closest",
            margin=dict(t=100, r=100),  # Reduced right margin since annotations are now inside
            legend=dict(
                x=0.11,
                y=0.99,
                xanchor="right",
                yanchor="top",
                bgcolor="rgba(255,255,255,0.6)",  # Optional: background for clarity
                bordercolor="gray",               # Optional: border for legend box
                borderwidth=1
            ),
        )

        fig.show()

    # Run it
    best_differentiation_plot()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
