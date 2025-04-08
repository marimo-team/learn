# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "matplotlib==3.10.1",
#     "numpy==2.2.4",
#     "drawdata==0.3.7",
#     "scikit-learn==1.6.1",
#     "polars==1.26.0",
# ]
# ///

import marimo

__generated_with = "0.12.5"
app = marimo.App(width="medium", app_title="Logistic Regression")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Logistic Regression

        _This notebook is a computational companion to ["Probability for Computer Scientists"](https://chrispiech.github.io/probabilityForComputerScientists/en/part5/log_regression/), by Stanford professor Chris Piech._

        Logistic regression learns a function approximating $P(y|x)$, and can be used to make a classifier. It makes the central assumption that $P(y|x)$ can be approximated as a sigmoid function applied to a linear combination of input features. It is particularly important to learn because logistic regression is the basic building block of artificial neural networks.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## The Binary Classification Problem

        Imagine situations where we would like to know:

        - The eligibility of getting a bank loan given the value of credit score ($x_{credit\_score}$) and monthly income ($x_{income}$)
        - Identifying a tumor as benign or malignant given its size ($x_{tumor\_size}$)
        - Classifying an email as promotional given the number of occurrences for some keywords like {'win', 'gift', 'discount'} ($x_{n\_win}$, $x_{n\_gift}$, $x_{n\_discount}$)
        - Finding a monetary transaction as fraudulent given the time of occurrence ($x_{time\_stamp}$) and amount ($x_{amount}$)

        These problems occur frequently in real life & can be dealt with machine learning. All such problems come under the umbrella of what is known as Classification. In each scenario, only one of the two possible outcomes can occur, hence these are specifically known as Binary Classification problems.

        ### How Does A Machine Perform Classification?

        During the inference, the goal is to have the ML model predict the class label for a given set of feature values.

        Specifically, a binary classification model estimates two probabilities $p_0$ & $p_1$ for 'class-0' and 'class-1' respectively where $p_0 + p_1 = 1$.

        The predicted label depends on $\max(p_0, p_1)$ i.e., it's the one which is most probable based on the given features.

        In logistic regression, $p_1$ (i.e., success probability) is compared with a predefined threshold $p$ to predict the class label like below:

        $$\text{predicted class} = 
        \begin{cases}
        1, & \text{if } p_1 \geq p \\
        0, & \text{otherwise}
        \end{cases}$$

        To keep the notation simple and consistent, we will denote the success probability as $p$, and failure probability as $(1-p)$ instead of $p_1$ and $p_0$ respectively.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Why NOT Linear Regression?

        Can't we really use linear regression to address classification? The answer is NO! The key issue is that probabilities must be between 0 and 1 and linear regression can output any real number.

        If we tried using linear regression directly:
        $$p = \beta_0 + \beta_1 \cdot x_{feature}$$

        This creates a problem: the right side can produce any value in $\mathbb{R}$ (all real numbers), but a probability $p$ must be confined to the range $(0,1)$.

        Can we convert $(\beta_0 + \beta_1 \cdot x_{tumor\_size})$ to something belonging to $(0,1)$? That may work as an estimate of a probability! The answer is YES!

        We need a converter (a function), say, $g()$ that will connect $p \in (0,1)$ to $(\beta_0 + \beta_1 \cdot x_{tumor\_size}) \in \mathbb{R}$.

        The solution is to use a "link function" that maps from any real number to a valid probability range. This is where the sigmoid function comes in.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo, np, plt):
    # plot sigmoid to evidentiate above statements
    _fig, ax = plt.subplots(figsize=(10, 6))

    # x values
    x = np.linspace(-10, 10, 1000)

    # sigmoid formula
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    y = sigmoid(x)

    # plot
    ax.plot(x, y, 'b-', linewidth=2)

    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.axhline(y=1, color='k', linestyle='-', alpha=0.3)
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)

    # vertical line at x=0
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)

    # annotations
    ax.text(1, 0.85, r'$\sigma(z) = \frac{1}{1 + e^{-z}}$', fontsize=14)
    ax.text(-9, 0.1, 'As z → -∞, σ(z) → 0', fontsize=12)
    ax.text(3, 0.9, 'As z → ∞, σ(z) → 1', fontsize=12)
    ax.text(0.5, 0.4, 'σ(0) = 0.5', fontsize=12)

    # labels and title
    ax.set_xlabel('z', fontsize=14)
    ax.set_ylabel('σ(z)', fontsize=14)
    ax.set_title('Sigmoid Function', fontsize=16)

    # axis limits set
    ax.set_xlim(-10, 10)
    ax.set_ylim(-0.1, 1.1)

    # grid
    ax.grid(True, alpha=0.3)

    mo.mpl.interactive(_fig)
    return ax, sigmoid, x, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Figure**: The sigmoid function maps any real number to a value between 0 and 1, making it perfect for representing probabilities.

        /// note
        For more information about the sigmoid function, head over to [this detailed notebook](http://marimo.app/https://github.com/marimo-team/deepml-notebooks/blob/main/problems/problem-22/notebook.py) for more insights.
        ///
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## The Core Concept (math)

        Logistic regression models the probability of class 1 using the sigmoid function:

        $$P(Y=1|X=x) = \sigma(z) \text{ where } z = \theta_0 + \sum_{i=1}^m \theta_i x_i$$

        The sigmoid function $\sigma(z)$ transforms any real number into a probability between 0 and 1:

        $$\sigma(z) = \frac{1}{1+ e^{-z}}$$

        This can be written more compactly using vector notation:

        $$P(Y=1|\mathbf{X}=\mathbf{x}) =\sigma(\mathbf{\theta}^T\mathbf{x}) \quad \text{ where we always set $x_0$ to be 1}$$

        $$P(Y=0|\mathbf{X}=\mathbf{x}) =1-\sigma(\mathbf{\theta}^T\mathbf{x}) \quad \text{ by total law of probability}$$

        Where $\theta$ represents the model parameters that need to be learned from data, and $x$ is the feature vector (with $x_0=1$ to account for the intercept term).

        > **Note:** For the detailed mathematical derivation of how these parameters are learned through Maximum Likelihood Estimation (MLE) and Gradient Descent (GD), please refer to [Chris Piech's original material](https://chrispiech.github.io/probabilityForComputerScientists/en/part5/log_regression/). The mathematical details are elegant but beyond the scope of this notebook topic (which is confined to Logistic Regression).
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Linear Decision Boundary

        A key characteristic of logistic regression is that it creates a linear decision boundary. When the model predicts, it's effectively dividing the feature space with a straight line (in 2D) or hyperplane (in higher dimensions). It is actually a straight line (of the form $y = mx + c$).

        Recall the prediction rule:
        $$\text{predicted class} = 
        \begin{cases}
        1, & \text{if } p \geq \theta_0 + \theta_1 \cdot x_{tumor\_size} \Rightarrow \log\frac{p}{1-p} \\
        0, & \text{otherwise}
        \end{cases}$$

        For a two-feature model, the decision boundary where $P(Y=1|X=x) = 0.5$ occurs at:
        $$\theta_0 + \theta_1 x_1 + \theta_2 x_2 = 0$$

        A simple logistic regression predicts the class label by identifying the regions on either side of a straight line (or hyperplane in general), hence it's a _linear_ classifier.

        This linear nature makes logistic regression effective for linearly separable classes but limited when dealing with more complex patterns.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""### Visual: Linear Separability and Classification""")
    return


@app.cell(hide_code=True)
def _(mo, np, plt):
    # show relevant comparison to the above concepts/statements

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Linear separable data
    np.random.seed(42)
    X1 = np.random.randn(100, 2) - 2
    X2 = np.random.randn(100, 2) + 2

    ax1.scatter(X1[:, 0], X1[:, 1], color='blue', alpha=0.5)
    ax1.scatter(X2[:, 0], X2[:, 1], color='red', alpha=0.5)

    # Decision boundary (line)
    ax1.plot([-5, 5], [5, -5], 'k--', linewidth=2)
    ax1.set_xlim(-5, 5)
    ax1.set_ylim(-5, 5)
    ax1.set_title('Linearly Separable Classes')

    # non-linear separable data
    radius = 2
    theta = np.linspace(0, 2*np.pi, 100)

    # Outer circle points (class 1)
    outer_x = 3 * np.cos(theta)
    outer_y = 3 * np.sin(theta)
    # Inner circle points (class 2)
    inner_x = 1.5 * np.cos(theta) + np.random.randn(100) * 0.2
    inner_y = 1.5 * np.sin(theta) + np.random.randn(100) * 0.2

    ax2.scatter(outer_x, outer_y, color='blue', alpha=0.5)
    ax2.scatter(inner_x, inner_y, color='red', alpha=0.5)

    # Attempt to draw a linear boundary (which won't work well) proving the point
    ax2.plot([-5, 5], [2, 2], 'k--', linewidth=2)

    ax2.set_xlim(-5, 5)
    ax2.set_ylim(-5, 5)
    ax2.set_title('Non-Linearly Separable Classes')

    fig.tight_layout()
    mo.mpl.interactive(fig)
    return (
        X1,
        X2,
        ax1,
        ax2,
        fig,
        inner_x,
        inner_y,
        outer_x,
        outer_y,
        radius,
        theta,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""**Figure**: On the left, the classes are linearly separable as the boundary is a straight line. However, they are not linearly separable on the right, where no straight line can properly separate the two classes.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Logistic regression is typically trained using MLE - finding the parameters $\theta$ that make our observed data most probable.

        The optimization process generally uses GD (or its variants) to iteratively improve the parameters. The gradient has a surprisingly elegant form:

        $$\frac{\partial LL(\theta)}{\partial \theta_j} = \sum_{i=1}^n \left[
        y^{(i)} - \sigma(\theta^T x^{(i)})
        \right] x_j^{(i)}$$

        This shows that the update to each parameter depends on the prediction error (actual - predicted) multiplied by the feature value.

        For those interested in the complete mathematical derivation, including log likelihood calculation and the detailed steps of GD (and relevant pseudocode followed for training), please see the [original lecture notes](https://chrispiech.github.io/probabilityForComputerScientists/en/part5/log_regression/).
        """
    )
    return


@app.cell(hide_code=True)
def _(controls, mo, widget):
    # create the layout
    mo.vstack([
        mo.md("## Interactive drawing demo\nDraw points of two different classes and see how logistic regression separates them. _The interactive demo was adapted and improvised from [Vincent Warmerdam's](https://github.com/koaning) code [here](https://github.com/probabl-ai/youtube-appendix/blob/main/04-drawing-data/notebook.ipynb)_."),
        controls,
        widget
    ])
    return


@app.cell(hide_code=True)
def _(LogisticRegression, mo, np, plt, run_button, widget):
    warning_msg = mo.md(""" /// warning
    Need more data, please draw points of at least two different colors in the scatter widget
    """)

    # mo.stop if button isn't clicked yet
    mo.stop(
        not run_button.value,
        mo.md(""" /// tip 
        click 'Run Logistic Regression' to see the model
        """)
    )

    # get data from widget (can also use as_pandas)
    df = widget.data_as_polars

    # display appropriate warning
    mo.stop(
        df.is_empty() or df['color'].n_unique() < 2,
        warning_msg
    )

    # extract features and labels
    X = df[['x', 'y']].to_numpy()
    y_colors = df['color'].to_numpy()

    # fit logistic regression model
    model = LogisticRegression()
    model.fit(X, y_colors)

    # create grid for the viz
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 100),
        np.linspace(y_min, y_max, 100)
    )

    # get probability predictions
    Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)

    # create figure
    _fig, ax_fig = plt.subplots(figsize=(12, 8))

    # plot decision boundary (probability contours)
    contour = ax_fig.contourf(
        xx, yy, Z, 
        levels=np.linspace(0, 1, 11),
        alpha=0.7,
        cmap="RdBu_r"
    )

    # plot decision boundary line (probability = 0.5)
    ax_fig.contour(
        xx, yy, Z,
        levels=[0.5],
        colors='k',
        linewidths=2
    )

    # plot the data points (use same colors as in the widget)
    ax_fig.scatter(X[:, 0], X[:, 1], c=y_colors, edgecolor='k', s=80)

    # colorbar
    plt.colorbar(contour, ax=ax_fig)

    # labels and title
    ax_fig.set_xlabel('x')
    ax_fig.set_ylabel('y')
    ax_fig.set_title('Logistic Regression')

    # model params
    coef = model.coef_[0]
    intercept = model.intercept_[0]
    equation = f"log(p/(1-p)) = {intercept:.2f} + {coef[0]:.3f}x₁ + {coef[1]:.3f}x₂"

    # relevant info in regards to regression
    model_info = mo.md(f"""
    ### Logistic regression model

    **Equation**: {equation}

    **Decision boundary**: probability = 0.5

    **Accuracy**: {model.score(X, y_colors):.2f}
    """)

    # show results vertically stacked
    mo.vstack([
        mo.mpl.interactive(_fig),
        model_info
    ])
    return (
        X,
        Z,
        ax_fig,
        coef,
        contour,
        df,
        equation,
        intercept,
        model,
        model_info,
        warning_msg,
        x_max,
        x_min,
        xx,
        y_colors,
        y_max,
        y_min,
        yy,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 🤔 Key Takeaways

        Click on the statements below that you think are correct to verify your understanding:

        /// details | Logistic regression tries to find parameters (θ) that minimize the error between predicted and actual values using ordinary least squares.
        ❌ **Incorrect.** Logistic regression uses maximum likelihood estimation (MLE), not ordinary least squares. It finds parameters that maximize the probability of observing the training data, which is different from minimizing squared errors as in linear regression.
        ///

        /// details | The sigmoid function maps any real number to a value between 0 and 1, which allows logistic regression to output probabilities.
        ✅ **Correct!** The sigmoid function σ(z) = 1/(1+e^(-z)) takes any real number as input and outputs a value between 0 and 1. This is perfect for representing probabilities and is a key component of logistic regression.
        ///

        /// details | The decision boundary in logistic regression is always a straight line, regardless of the data's complexity.
        ✅ **Correct!** Standard logistic regression produces a linear decision boundary (a straight line in 2D or a hyperplane in higher dimensions). This is why it works well for linearly separable data but struggles with more complex patterns, like concentric circles (as you might've noticed from the interactive demo).
        ///

        /// details | The logistic regression model params are typically initialized to random values and refined through gradient descent.
        ✅ **Correct!** Parameters are often initialized to zeros or small random values, then updated iteratively using gradient descent (or ascent for maximizing likelihood) until convergence.
        ///

        /// details | Logistic regression can naturally handle multi-class classification problems without any modifications.
        ❌ **Incorrect.** Standard logistic regression is inherently a binary classifier. To handle multi-class classification, techniques like one-vs-rest or softmax regression are typically used.
        ///
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Summary

        So we've just explored logistic regression. Despite its name (seriously though, why not call it "logistic classification"?), it's actually quite elegant in how it transforms a simple linear model into a powerful decision _boundary_ maker.

        The training process boils down to finding the values of θ that maximize the likelihood of seeing our training data. What's super cool is that even though the math looks _scary_ at first, the gradient has this surprisingly simple form: just the error (y - predicted) multiplied by the feature values.

        Two key insights to remember:

        - Logistic regression creates a _linear_ decision boundary, so it works great for linearly separable classes but struggles with more _complex_ patterns
        - It directly gives you probabilities, not just classifications, which is incredibly useful when you need confidence measures
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Additional resources referred to:

        - [Logistic Regression Tutorial by _Koushik Khan_](https://koushikkhan.github.io/resources/pdf/tutorials/logistic_regression_tutorial.pdf)
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Appendix (helper code)""")
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def init_imports():
    # imports for our notebook
    import numpy as np
    import matplotlib.pyplot as plt
    from drawdata import ScatterWidget
    from sklearn.linear_model import LogisticRegression


    # for consistent results
    np.random.seed(42)

    # nicer plots
    plt.style.use('seaborn-v0_8-darkgrid')
    return LogisticRegression, ScatterWidget, np, plt


@app.cell(hide_code=True)
def _(ScatterWidget, mo):
    # drawing widget
    widget = mo.ui.anywidget(ScatterWidget())

    # run_button to run model
    run_button = mo.ui.run_button(label="Run Logistic Regression", kind="success")

    # stack controls
    controls = mo.hstack([run_button])
    return controls, run_button, widget


if __name__ == "__main__":
    app.run()
