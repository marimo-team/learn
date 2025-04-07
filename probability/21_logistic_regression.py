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

__generated_with = "0.12.4"
app = marimo.App(width="medium", app_title="Logistic Regression")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Logistic Regression

        _This notebook is a computational companion to ["Probability for Computer Scientists"](https://chrispiech.github.io/probabilityForComputerScientists/en/part5/log_regression/), by Stanford professor Chris Piech._

        Logistic Regression is a classification algorithm that learns a function approximating $P(y|x)$, and can be used to make a classifier. It makes the central assumption that $P(y|x)$ can be approximated as a sigmoid function applied to a linear combination of input features. It is particularly important to learn because logistic regression is the basic building block of artificial neural networks.
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

        Can't we really use linear regression to address classification? The answer is NO!

        Let's try to understand why:

        To estimate $p$ using linear regression, we would need:
        $$p = \beta_0 + \beta_1 \cdot x_{tumor\_size}$$

        This doesn't seem to be feasible as the right side, in principle, belongs to $\mathbb{R}$ (any real number) & the left side belongs to $(0,1)$ (a probability).

        Can we convert $(\beta_0 + \beta_1 \cdot x_{tumor\_size})$ to something belonging to $(0,1)$? That may work as an estimate of a probability! The answer is YES!

        We need a converter (a function), say, $g()$ that will connect $p \in (0,1)$ to $(\beta_0 + \beta_1 \cdot x_{tumor\_size}) \in \mathbb{R}$.

        Fortunately, such functions do exist and they are often referred to as link functions in this context.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## The Mathematical Foundation

        Mathematically, for a single training datapoint $(\mathbf{x}, y)$ Logistic Regression assumes:

        $$P(Y=1|\mathbf{X}=\mathbf{x}) = \sigma(z) \text{ where } z = \theta_0 + \sum_{i=1}^m \theta_i x_i$$

        This assumption is often written in the equivalent forms:

        $$P(Y=1|\mathbf{X}=\mathbf{x}) =\sigma(\mathbf{\theta}^T\mathbf{x}) \quad \text{ where we always set $x_0$ to be 1}$$

        $$P(Y=0|\mathbf{X}=\mathbf{x}) =1-\sigma(\mathbf{\theta}^T\mathbf{x}) \quad \text{ by total law of probability}$$

        Using these equations for probability of $Y|X$ we can create an algorithm that selects values of $\theta$ that maximize that probability for all data. I am first going to state the log probability function and partial derivatives with respect to $\theta$. Then later we will (a) show an algorithm that can chose optimal values of $\theta$ and (b) show how the equations were derived.

        An important thing to realize is that: given the best values for the parameters ($\theta$), logistic regression often can do a great job of estimating the probability of different class labels. However, given bad, or even random, values of $\theta$ it does a poor job. The amount of "intelligence" that your logistic regression machine learning algorithm has is dependent on having good values of $\theta$.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Logistic Regression Is A Linear Classifier

        The logistic regression equation is actually a straight line (of the form $y = mx + c$).

        Recall the prediction rule:
        $$\text{predicted class} = 
        \begin{cases}
        1, & \text{if } p \geq \theta_0 + \theta_1 \cdot x_{tumor\_size} \Rightarrow \log\frac{p}{1-p} \\
        0, & \text{otherwise}
        \end{cases}$$

        A simple logistic regression (the one we discussed) predicts the class label by identifying the regions on either side of a straight line (or hyperplane in general), hence it's a linear classifier.

        Logistic regression works well for linearly separable classes.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""### Visual: Linear Separability and Classification""")
    return


@app.cell(hide_code=True)
def _(mo, np, plt):
    # show relevant comaparison to the above last sentence/statement

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
        ## Notation

        Before we get started I want to make sure that we are all on the same page with respect to notation. In logistic regression, $\theta$ is a vector of parameters of length $m$ and we are going to learn the values of those parameters based off of $n$ training examples. The number of parameters should be equal to the number of features of each datapoint.

        Two pieces of notation that we use often in logistic regression that you may not be familiar with are:

        $$\mathbf{\theta}^T\mathbf{x} = \sum_{i=1}^m \theta_i x_i = \theta_1 x_1 + \theta_2 x_2 + \dots + \theta_m x_m \quad \text{dot product, aka weighted sum}$$

        $$\sigma(z) = \frac{1}{1+ e^{-z}} \quad \text{sigmoid function}$$

        The sigmoid function is a special function that maps any real number to a probability between 0 and 1. It has an S-shaped curve and is particularly useful for binary classification problems.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo, np, plt):
    # Plot the sigmoid function

    _fig, ax = plt.subplots(figsize=(10, 6))

    # Generate x values
    x = np.linspace(-10, 10, 1000)

    # Compute sigmoid
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    y = sigmoid(x)

    # Plot sigmoid function
    ax.plot(x, y, 'b-', linewidth=2)

    # Add horizontal lines at y=0 and y=1
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.axhline(y=1, color='k', linestyle='-', alpha=0.3)
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)

    # Add vertical line at x=0
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)

    # Add annotations
    ax.text(1, 0.85, r'$\sigma(z) = \frac{1}{1 + e^{-z}}$', fontsize=14)
    ax.text(-9, 0.1, 'As z → -∞, σ(z) → 0', fontsize=12)
    ax.text(3, 0.9, 'As z → ∞, σ(z) → 1', fontsize=12)
    ax.text(0.5, 0.4, 'σ(0) = 0.5', fontsize=12)

    # Set labels and title
    ax.set_xlabel('z', fontsize=14)
    ax.set_ylabel('σ(z)', fontsize=14)
    ax.set_title('Sigmoid Function', fontsize=16)

    # Set axis limits
    ax.set_xlim(-10, 10)
    ax.set_ylim(-0.1, 1.1)

    # Add grid
    ax.grid(True, alpha=0.3)

    mo.mpl.interactive(_fig)

    mo.md(r"""
    **Figure**: The sigmoid function maps any real number to a value between 0 and 1, making it perfect for representing probabilities.

    /// note
    For more information about the sigmoid function and its applications in deep learning, head over to [this detailed notebook](http://marimo.app/https://github.com/marimo-team/deepml-notebooks/blob/main/problems/problem-22/notebook.py) for more insights.
    ///
    """)
    return ax, sigmoid, x, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Log Likelihood

        In order to choose values for the parameters of logistic regression we use Maximum Likelihood Estimation (MLE). As such we are going to have two steps: (1) write the log-likelihood function and (2) find the values of $\theta$ that maximize the log-likelihood function.

        The labels that we are predicting are binary, and the output of our logistic regression function is supposed to be the probability that the label is one. This means that we can (and should) interpret each label as a Bernoulli random variable: $Y \sim \text{Bern}(p)$ where $p = \sigma(\theta^T \textbf{x})$.

        To start, here is a super slick way of writing the probability of one datapoint (recall this is the equation form of the probability mass function of a Bernoulli):

        $$P(Y=y | X = \mathbf{x}) = \sigma({\mathbf{\theta}^T\mathbf{x}})^y \cdot \left[1 - \sigma({\mathbf{\theta}^T\mathbf{x}})\right]^{(1-y)}$$

        Now that we know the probability mass function, we can write the likelihood of all the data:

        $$L(\theta) = \prod_{i=1}^n P(Y=y^{(i)} | X = \mathbf{x}^{(i)}) \quad \text{The likelihood of independent training labels}$$

        $$= \prod_{i=1}^n \sigma({\mathbf{\theta}^T\mathbf{x}^{(i)}})^{y^{(i)}} \cdot \left[1 - \sigma({\mathbf{\theta}^T\mathbf{x}^{(i)}})\right]^{(1-y^{(i)})} \quad \text{Substituting the likelihood of a Bernoulli}$$

        And if you take the log of this function, you get the reported Log Likelihood for Logistic Regression. The log likelihood equation is:

        $$LL(\theta) = \sum_{i=1}^n y^{(i)} \log \sigma(\mathbf{\theta}^T\mathbf{x}^{(i)}) + (1-y^{(i)}) \log [1 - \sigma(\mathbf{\theta}^T\mathbf{x}^{(i)})]$$

        Recall that in MLE the only remaining step is to choose parameters ($\theta$) that maximize log likelihood.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Gradient of Log Likelihood

        Now that we have a function for log-likelihood, we simply need to choose the values of $\theta$ that maximize it. We can find the best values of $\theta$ by using an optimization algorithm. However, in order to use an optimization algorithm, we first need to know the partial derivative of log likelihood with respect to each parameter. First I am going to give you the partial derivative (so you can see how it is used). Then I am going to show you how to derive it:

        $$\frac{\partial LL(\theta)}{\partial \theta_j} = \sum_{i=1}^n \left[
        y^{(i)} - \sigma(\mathbf{\theta}^T\mathbf{x}^{(i)})
        \right] x_j^{(i)}$$

        This is a beautifully simple formula. Notice that the gradient is the sum of the error terms $(y^{(i)} - \sigma(\mathbf{\theta}^T\mathbf{x}^{(i)}))$ multiplied by the feature value $x_j^{(i)}$. The _error term_ represents the _difference_ between the true label and our predicted probability.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Gradient Descent Optimization

        Our goal is to choose parameters ($\theta$) that maximize likelihood, and we know the partial derivative of log likelihood with respect to each parameter. We are ready for our optimization algorithm.

        In the case of logistic regression, we can't solve for $\theta$ mathematically. Instead, we use a computer to chose $\theta$. To do so we employ an algorithm called gradient descent (a classic in optimization theory). The idea behind gradient descent is that if you continuously take small steps downhill (in the direction of your negative gradient), you will eventually make it to a local minima. In our case we want to maximize our likelihood. As you can imagine, minimizing a negative of our likelihood will be equivalent to maximizing our likelihood.

        The update to our parameters that results in each small step can be calculated as:

        $$\theta_j^{\text{ new}} = \theta_j^{\text{ old}} + \eta \cdot \frac{\partial LL(\theta^{\text{ old}})}{\partial \theta_j^{\text{ old}}}$$

        $$= \theta_j^{\text{ old}} + \eta \cdot \sum_{i=1}^n \left[
        y^{(i)} - \sigma(\mathbf{\theta}^T\mathbf{x}^{(i)})
        \right] x_j^{(i)}$$

        Where $\eta$ is the magnitude of the step size that we take. If you keep updating $\theta$ using the equation above you will converge on the best values of $\theta$. You now have an intelligent model. Here is the gradient ascent algorithm for logistic regression in pseudo-code:
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    # Create a stylized pseudocode display
    mo.md(r"""
    ```
    Initialize: θⱼ = 0 for all 0 ≤ j ≤ m

    Repeat many times:
        gradient[j] = 0 for all 0 ≤ j ≤ m

        For each training example (x, y):
            For each parameter j:
                gradient[j] += xⱼ(y - 1/(1+e^(-θᵀx)))

        θⱼ += η * gradient[j] for all 0 ≤ j ≤ m
    ```

    **Pro-tip:** Don't forget that in order to learn the value of θ₀ you can simply define x₀ to always be 1.
    """)
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
        ## Derivations

        In this section we provide the mathematical derivations for the gradient of log-likelihood. The derivations are worth knowing because these ideas are heavily used in Artificial Neural Networks.

        Our goal is to calculate the derivative of the log likelihood with respect to each theta. To start, here is the definition for the derivative of a sigmoid function with respect to its inputs:

        $$\frac{\partial}{\partial z} \sigma(z) = \sigma(z)[1 - \sigma(z)] \quad \text{to get the derivative with respect to $\theta$, use the chain rule}$$

        Take a moment and appreciate the beauty of the derivative of the sigmoid function. The reason that sigmoid has such a simple derivative stems from the natural exponent in the sigmoid denominator.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Detailed Derivation

        Since the likelihood function is a sum over all of the data, and in calculus the derivative of a sum is the sum of derivatives, we can focus on computing the derivative of one example. The gradient of theta is simply the sum of this term for each training datapoint.

        First I am going to show you how to compute the derivative the hard way. Then we are going to look at an easier method. The derivative of gradient for one datapoint $(\mathbf{x}, y)$:

        $$\begin{align}
        \frac{\partial LL(\theta)}{\partial \theta_j} &= \frac{\partial }{\partial \theta_j} y \log \sigma(\mathbf{\theta}^T\mathbf{x}) + \frac{\partial }{\partial \theta_j} (1-y) \log [1 - \sigma(\mathbf{\theta}^T\mathbf{x})] \quad \text{derivative of sum of terms}\\
        &=\left[\frac{y}{\sigma(\theta^T\mathbf{x})} - \frac{1-y}{1-\sigma(\theta^T\mathbf{x})} \right] \frac{\partial}{\partial \theta_j} \sigma(\theta^T \mathbf{x}) \quad \text{derivative of log $f(x)$}\\
        &=\left[\frac{y}{\sigma(\theta^T\mathbf{x})} - \frac{1-y}{1-\sigma(\theta^T\mathbf{x})} \right] \sigma(\theta^T \mathbf{x}) [1 - \sigma(\theta^T \mathbf{x})]\mathbf{x}_j \quad \text{chain rule + derivative of sigma}\\
        &=\left[
        \frac{y - \sigma(\theta^T\mathbf{x})}{\sigma(\theta^T \mathbf{x}) [1 - \sigma(\theta^T \mathbf{x})]}
        \right] \sigma(\theta^T \mathbf{x}) [1 - \sigma(\theta^T \mathbf{x})]\mathbf{x}_j \quad \text{algebraic manipulation}\\
        &= \left[y - \sigma(\theta^T\mathbf{x}) \right] \mathbf{x}_j \quad \text{cancelling terms}
        \end{align}$$
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Derivatives Without Tears

        That was the hard way. Logistic regression is the building block of [Artificial Neural Networks](https://en.wikipedia.org/wiki/Neural_network_(machine_learning)). If we want to scale up, we are going to have to get used to an easier way of calculating derivatives. For that we are going to have to welcome back our old friend the chain rule. By the chain rule:

        $$\begin{align}
        \frac{\partial LL(\theta)}{\partial \theta_j} &= 
        \frac{\partial LL(\theta)}{\partial p} 
        \cdot  \frac{\partial p}{\partial \theta_j}
        \quad \text{Where } p = \sigma(\theta^T\textbf{x})\\
        &= 
        \frac{\partial LL(\theta)}{\partial p} 
        \cdot  \frac{\partial p}{\partial z} 
        \cdot  \frac{\partial z}{\partial \theta_j}
        \quad \text{Where } z = \theta^T\textbf{x}
        \end{align}$$

        Chain rule is the decomposition mechanism of calculus. It allows us to calculate a complicated partial derivative $\frac{\partial LL(\theta)}{\partial \theta_j}$ by breaking it down into smaller pieces.

        $$\begin{align}
        LL(\theta) &= y \log p + (1-y) \log (1 - p) \quad \text{Where } p = \sigma(\theta^T\textbf{x}) \\
        \frac{\partial LL(\theta)}{\partial p} &= \frac{y}{p} - \frac{1-y}{1-p} \quad \text{By taking the derivative}
        \end{align}$$

        $$\begin{align}
        p &= \sigma(z) \quad \text{Where }z = \theta^T\textbf{x}\\
        \frac{\partial p}{\partial z} &= \sigma(z)[1- \sigma(z)] \quad \text{By taking the derivative of the sigmoid}
        \end{align}$$

        $$\begin{align}
        z &= \theta^T\textbf{x} \quad \text{As previously defined}\\
        \frac{\partial z}{\partial \theta_j} &= \textbf{x}_j \quad \text{ Only $\textbf{x}_j$ interacts with $\theta_j$}
        \end{align}$$

        Each of those derivatives was much easier to calculate. Now we simply multiply them together.

        $$\begin{align}
        \frac{\partial LL(\theta)}{\partial \theta_j} &=
        \frac{\partial LL(\theta)}{\partial p} 
        \cdot  \frac{\partial p}{\partial z} 
        \cdot  \frac{\partial z}{\partial \theta_j} \\
        &=
        \Big[\frac{y}{p} - \frac{1-y}{1-p}\Big]
        \cdot  \sigma(z)[1- \sigma(z)]
        \cdot \textbf{x}_j \quad \text{By substituting in for each term} \\
        &=
        \Big[\frac{y}{p} - \frac{1-y}{1-p}\Big]
        \cdot p[1- p]
        \cdot \textbf{x}_j \quad \text{Since }p = \sigma(z)\\
        &=
        [y(1-p) - p(1-y)]
        \cdot \textbf{x}_j \quad \text{Multiplying in} \\
        &= [y - p]\textbf{x}_j \quad \text{Expanding} \\
        &= [y - \sigma(\theta^T\textbf{x})]\textbf{x}_j \quad \text{Since } p = \sigma(\theta^T\textbf{x})
        \end{align}$$
        """
    )
    return


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
