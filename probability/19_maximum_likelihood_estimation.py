# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "matplotlib==3.10.1",
#     "scipy==1.15.2",
#     "numpy==2.2.4",
#     "polars==0.20.2",
#     "plotly==5.18.0",
# ]
# ///

import marimo

__generated_with = "0.12.0"
app = marimo.App(width="medium", app_title="Maximum Likelihood Estimation")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Maximum Likelihood Estimation

        _This notebook is a computational companion to ["Probability for Computer Scientists"](https://chrispiech.github.io/probabilityForComputerScientists/en/part5/mle/), by Stanford professor Chris Piech._

        Maximum Likelihood Estimation (MLE) is a fundamental method in statistics for estimating parameters of a probability distribution. The central idea is elegantly simple: **choose the parameters that make the observed data most likely**.

        In this notebook, we'll try to understand MLE, starting with the core concept of likelihood and how it differs from probability. We'll explore how to formulate MLE problems mathematically and then solve them for various common distributions. Along the way, I've included some interactive visualizations to help build your intuition about these concepts. You'll see how MLE applies to real-world scenarios like linear regression, and hopefully gain a deeper appreciation for why this technique is so widely used in statistics and machine learning. Think of MLE as detective work - we have some evidence (our data) and we're trying to figure out the most plausible explanation (our parameters) for what we've observed.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Likelihood: The Core Concept

        Before diving into MLE, we need to understand what "likelihood" means in a statistical context.

        ### Data and Parameters

        Suppose we have collected some data $X_1, X_2, \ldots, X_n$ that are independent and identically distributed (IID). We assume these data points come from a specific type of distribution (like Normal, Bernoulli, etc.) with unknown parameters $\theta$.

        ### What is Likelihood?

        Likelihood measures how probable our observed data is, given specific values of the parameters $\theta$.

        - For **discrete** distributions: likelihood is the probability mass function (PMF) of our data
        - For **continuous** distributions: likelihood is the probability density function (PDF) of our data

        /// note
        **Probability vs. Likelihood**

        - **Probability**: Given parameters $\theta$, what's the chance of observing data $X$?
        - **Likelihood**: Given observed data $X$, how likely are different parameter values $\theta$?

        They use the same formula but different perspectives!
        ///

        To simplify notation, we'll use $f(X=x|\Theta=\theta)$ to represent either the PMF or PDF of our data, conditioned on the parameters.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### The Likelihood Function

        Since we assume our data points are independent, the likelihood of all our data is the product of the likelihoods of each individual data point:

        $$L(\theta) = \prod_{i=1}^n f(X_i = x_i|\Theta = \theta)$$

        This function $L(\theta)$ gives us the likelihood of observing our entire dataset for different parameter values $\theta$.

        /// tip
        **Key Insight**: Different parameter values produce different likelihoods for the same data. Better parameter values will make the observed data more likely.
        ///
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Maximum Likelihood Estimation

        The core idea of MLE is to find the parameter values $\hat{\theta}$ that maximize the likelihood function:

        $$\hat{\theta} = \underset{\theta}{\operatorname{argmax}} \, L(\theta)$$

        The notation $\hat{\theta}$ represents our best estimate of the true parameters based on the observed data.

        ### Working with Log-Likelihood

        In practice, we usually work with the **log-likelihood** instead of the likelihood directly. Since logarithm is a monotonically increasing function, the maximum of $L(\theta)$ occurs at the same value of $\theta$ as the maximum of $\log L(\theta)$.

        Taking the logarithm transforms our product into a sum, which is much easier to work with:

        $$LL(\theta) = \log L(\theta) = \log \prod_{i=1}^n f(X_i=x_i|\Theta = \theta) = \sum_{i=1}^n \log f(X_i = x_i|\Theta = \theta)$$

        /// warning
        Working with products of many small probabilities can lead to numerical underflow. Taking the logarithm converts these products to sums, which is numerically more stable.
        ///
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Finding the Maximum

        To find the values of $\theta$ that maximize the log-likelihood, we typically:

        1. Take the derivative of $LL(\theta)$ with respect to each parameter
        2. Set each derivative equal to zero
        3. Solve for the parameters

        Let's see this approach in action with some common distributions.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## MLE for Bernoulli Distribution

        Let's start with a simple example: estimating the parameter $p$ of a Bernoulli distribution.

        ### The Model

        A Bernoulli distribution has a single parameter $p$ which represents the probability of success (getting a value of 1). Its probability mass function (PMF) can be written as:

        $$f(x|p) = p^x(1-p)^{1-x}, \quad x \in \{0, 1\}$$

        This elegant formula works because:

        - When $x = 1$: $f(1|p) = p^1(1-p)^0 = p$
        - When $x = 0$: $f(0|p) = p^0(1-p)^1 = 1-p$

        ### Deriving the MLE

        Given $n$ independent Bernoulli trials $X_1, X_2, \ldots, X_n$, we want to find the value of $p$ that maximizes the likelihood of our observed data.

        Step 1: Write the likelihood function
        $$L(p) = \prod_{i=1}^n p^{x_i}(1-p)^{1-x_i}$$

        Step 2: Take the logarithm to get the log-likelihood
        $$\begin{align*}
        LL(p) &= \sum_{i=1}^n \log(p^{x_i}(1-p)^{1-x_i}) \\
        &= \sum_{i=1}^n \left[x_i \log(p) + (1-x_i)\log(1-p)\right] \\
        &= \left(\sum_{i=1}^n x_i\right) \log(p) + \left(n - \sum_{i=1}^n x_i\right) \log(1-p) \\
        &= Y\log(p) + (n-Y)\log(1-p)
        \end{align*}$$

        where $Y = \sum_{i=1}^n x_i$ is the total number of successes.

        Step 3: Find the value of $p$ that maximizes $LL(p)$ by setting the derivative to zero
        $$\begin{align*}
        \frac{d\,LL(p)}{dp} &= \frac{Y}{p} - \frac{n-Y}{1-p} = 0 \\
        \frac{Y}{p} &= \frac{n-Y}{1-p} \\
        Y(1-p) &= p(n-Y) \\
        Y - Yp &= pn - pY \\
        Y &= pn \\
        \hat{p} &= \frac{Y}{n} = \frac{\sum_{i=1}^n x_i}{n}
        \end{align*}$$

        /// tip
        The MLE for the parameter $p$ in a Bernoulli distribution is simply the **sample mean** - the proportion of successes in our data!
        ///
        """
    )
    return


@app.cell(hide_code=True)
def _(controls):
    controls.center()
    return


@app.cell(hide_code=True)
def _(generate_button, mo, np, plt, sample_size_slider, true_p_slider):
    # generate bernoulli samples when button is clicked
    bernoulli_button_value = generate_button.value

    # get parameter values
    bernoulli_true_p = true_p_slider.value
    bernoulli_n = sample_size_slider.value

    # generate data
    bernoulli_data = np.random.binomial(1, bernoulli_true_p, size=bernoulli_n)
    bernoulli_Y = np.sum(bernoulli_data)
    bernoulli_p_hat = bernoulli_Y / bernoulli_n

    # create visualization
    bernoulli_fig, (bernoulli_ax1, bernoulli_ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # plot data histogram
    bernoulli_ax1.hist(bernoulli_data, bins=[-0.5, 0.5, 1.5], rwidth=0.8, color='lightblue')
    bernoulli_ax1.set_xticks([0, 1])
    bernoulli_ax1.set_xticklabels(['Failure (0)', 'Success (1)'])
    bernoulli_ax1.set_title(f'Bernoulli Data: {bernoulli_n} samples')
    bernoulli_ax1.set_ylabel('Count')
    bernoulli_y_counts = [bernoulli_n - bernoulli_Y, bernoulli_Y]
    for bernoulli_idx, bernoulli_count in enumerate(bernoulli_y_counts):
        bernoulli_ax1.text(bernoulli_idx, bernoulli_count/2, f"{bernoulli_count}", 
                 ha='center', va='center', 
                 color='white' if bernoulli_idx == 0 else 'black', 
                 fontweight='bold')

    # calculate log-likelihood function
    bernoulli_p_values = np.linspace(0.01, 0.99, 100)
    bernoulli_ll_values = np.zeros_like(bernoulli_p_values)

    for bernoulli_i, bernoulli_p in enumerate(bernoulli_p_values):
        bernoulli_ll_values[bernoulli_i] = bernoulli_Y * np.log(bernoulli_p) + (bernoulli_n - bernoulli_Y) * np.log(1 - bernoulli_p)

    # plot log-likelihood
    bernoulli_ax2.plot(bernoulli_p_values, bernoulli_ll_values, 'b-', linewidth=2)
    bernoulli_ax2.axvline(x=bernoulli_p_hat, color='r', linestyle='--', label=f'MLE: $\\hat{{p}} = {bernoulli_p_hat:.3f}$')
    bernoulli_ax2.axvline(x=bernoulli_true_p, color='g', linestyle='--', label=f'True: $p = {bernoulli_true_p:.3f}$')
    bernoulli_ax2.set_xlabel('$p$ (probability of success)')
    bernoulli_ax2.set_ylabel('Log-Likelihood')
    bernoulli_ax2.set_title('Log-Likelihood Function')
    bernoulli_ax2.legend()

    plt.tight_layout()
    plt.gca()

    # Create markdown to explain the results
    bernoulli_explanation = mo.md(
        f"""
        ### Bernoulli MLE Results

        **True parameter**: $p = {bernoulli_true_p:.3f}$  
        **Sample statistics**: {bernoulli_Y} successes out of {bernoulli_n} trials  
        **MLE estimate**: $\\hat{{p}} = \\frac{{{bernoulli_Y}}}{{{bernoulli_n}}} = {bernoulli_p_hat:.3f}$

        The plot on the right shows the log-likelihood function $LL(p) = Y\\log(p) + (n-Y)\\log(1-p)$. 
        The red dashed line marks the maximum likelihood estimate $\\hat{{p}}$, and the green dashed line 
        shows the true parameter value.

        /// note
        Try increasing the sample size to see how the MLE estimate gets closer to the true parameter value!
        ///
        """
    )

    # Display plot and explanation together
    mo.vstack([
        bernoulli_fig,
        bernoulli_explanation
    ])
    return (
        bernoulli_Y,
        bernoulli_ax1,
        bernoulli_ax2,
        bernoulli_button_value,
        bernoulli_count,
        bernoulli_data,
        bernoulli_explanation,
        bernoulli_fig,
        bernoulli_i,
        bernoulli_idx,
        bernoulli_ll_values,
        bernoulli_n,
        bernoulli_p,
        bernoulli_p_hat,
        bernoulli_p_values,
        bernoulli_true_p,
        bernoulli_y_counts,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## MLE for Normal Distribution

        Next, let's look at a more complex example: estimating the parameters $\mu$ and $\sigma^2$ of a Normal distribution.

        ### The Model

        A Normal (Gaussian) distribution has two parameters:
        - $\mu$: the mean
        - $\sigma^2$: the variance

        Its probability density function (PDF) is:

        $$f(x|\mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)$$

        ### Deriving the MLE

        Given $n$ independent samples $X_1, X_2, \ldots, X_n$ from a Normal distribution, we want to find the values of $\mu$ and $\sigma^2$ that maximize the likelihood of our observed data.

        Step 1: Write the likelihood function
        $$L(\mu, \sigma^2) = \prod_{i=1}^n \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x_i - \mu)^2}{2\sigma^2}\right)$$

        Step 2: Take the logarithm to get the log-likelihood
        $$\begin{align*}
        LL(\mu, \sigma^2) &= \log\prod_{i=1}^n \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x_i - \mu)^2}{2\sigma^2}\right) \\
        &= \sum_{i=1}^n \log\left[\frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x_i - \mu)^2}{2\sigma^2}\right)\right] \\
        &= \sum_{i=1}^n \left[-\frac{1}{2}\log(2\pi\sigma^2) - \frac{(x_i - \mu)^2}{2\sigma^2}\right] \\
        &= -\frac{n}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^n (x_i - \mu)^2
        \end{align*}$$

        Step 3: Find the values of $\mu$ and $\sigma^2$ that maximize $LL(\mu, \sigma^2)$ by setting the partial derivatives to zero.

        For $\mu$:
        $$\begin{align*}
        \frac{\partial LL(\mu, \sigma^2)}{\partial \mu} &= \frac{1}{\sigma^2}\sum_{i=1}^n (x_i - \mu) = 0 \\
        \sum_{i=1}^n (x_i - \mu) &= 0 \\
        \sum_{i=1}^n x_i &= n\mu \\
        \hat{\mu} &= \frac{1}{n}\sum_{i=1}^n x_i
        \end{align*}$$

        For $\sigma^2$:
        $$\begin{align*}
        \frac{\partial LL(\mu, \sigma^2)}{\partial \sigma^2} &= -\frac{n}{2\sigma^2} + \frac{1}{2(\sigma^2)^2}\sum_{i=1}^n (x_i - \mu)^2 = 0 \\
        \frac{n}{2\sigma^2} &= \frac{1}{2(\sigma^2)^2}\sum_{i=1}^n (x_i - \mu)^2 \\
        n\sigma^2 &= \sum_{i=1}^n (x_i - \mu)^2 \\
        \hat{\sigma}^2 &= \frac{1}{n}\sum_{i=1}^n (x_i - \hat{\mu})^2
        \end{align*}$$

        /// tip
        The MLE for a Normal distribution gives us:

        - $\hat{\mu}$ = sample mean
        - $\hat{\sigma}^2$ = sample variance (using $n$ in the denominator, not $n-1$)
        ///
        """
    )
    return


@app.cell(hide_code=True)
def _(normal_controls):
    normal_controls.center()
    return


@app.cell(hide_code=True)
def _(
    mo,
    normal_generate_button,
    normal_sample_size_slider,
    np,
    plt,
    true_mu_slider,
    true_sigma_slider,
):
    # generate normal samples when button is clicked
    normal_button_value = normal_generate_button.value

    # get parameter values
    normal_true_mu = true_mu_slider.value
    normal_true_sigma = true_sigma_slider.value
    normal_true_var = normal_true_sigma**2
    normal_n = normal_sample_size_slider.value

    # generate random data
    normal_data = np.random.normal(normal_true_mu, normal_true_sigma, size=normal_n)

    # calculate mle estimates
    normal_mu_hat = np.mean(normal_data)
    normal_sigma2_hat = np.mean((normal_data - normal_mu_hat)**2)  # mle variance using n
    normal_sigma_hat = np.sqrt(normal_sigma2_hat)

    # create visualization
    normal_fig, (normal_ax1, normal_ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # plot histogram and density curves
    normal_bins = np.linspace(min(normal_data) - 1, max(normal_data) + 1, 30)
    normal_ax1.hist(normal_data, bins=normal_bins, density=True, alpha=0.6, color='lightblue', label='Data Histogram')

    # plot range for density curves
    normal_x = np.linspace(min(normal_data) - 2*normal_true_sigma, max(normal_data) + 2*normal_true_sigma, 1000)

    # plot true and mle densities
    normal_true_pdf = (1/(normal_true_sigma * np.sqrt(2*np.pi))) * np.exp(-0.5 * ((normal_x - normal_true_mu)/normal_true_sigma)**2)
    normal_ax1.plot(normal_x, normal_true_pdf, 'g-', linewidth=2, label=f'True: N({normal_true_mu:.2f}, {normal_true_var:.2f})')

    normal_mle_pdf = (1/(normal_sigma_hat * np.sqrt(2*np.pi))) * np.exp(-0.5 * ((normal_x - normal_mu_hat)/normal_sigma_hat)**2)
    normal_ax1.plot(normal_x, normal_mle_pdf, 'r--', linewidth=2, label=f'MLE: N({normal_mu_hat:.2f}, {normal_sigma2_hat:.2f})')

    normal_ax1.set_xlabel('x')
    normal_ax1.set_ylabel('Density')
    normal_ax1.set_title(f'Normal Distribution: {normal_n} samples')
    normal_ax1.legend()

    # create contour plot of log-likelihood
    normal_mu_range = np.linspace(normal_mu_hat - 2, normal_mu_hat + 2, 100)
    normal_sigma_range = np.linspace(max(0.1, normal_sigma_hat - 1), normal_sigma_hat + 1, 100)

    normal_mu_grid, normal_sigma_grid = np.meshgrid(normal_mu_range, normal_sigma_range)
    normal_ll_grid = np.zeros_like(normal_mu_grid)

    # calculate log-likelihood for each grid point
    for normal_i in range(normal_mu_grid.shape[0]):
        for normal_j in range(normal_mu_grid.shape[1]):
            normal_mu = normal_mu_grid[normal_i, normal_j]
            normal_sigma = normal_sigma_grid[normal_i, normal_j]
            normal_ll = -normal_n/2 * np.log(2*np.pi*normal_sigma**2) - np.sum((normal_data - normal_mu)**2)/(2*normal_sigma**2)
            normal_ll_grid[normal_i, normal_j] = normal_ll

    # plot log-likelihood contour
    normal_contour = normal_ax2.contourf(normal_mu_grid, normal_sigma_grid, normal_ll_grid, levels=50, cmap='viridis')
    normal_ax2.set_xlabel('μ (mean)')
    normal_ax2.set_ylabel('σ (standard deviation)')
    normal_ax2.set_title('Log-Likelihood Contour')

    # mark mle and true params
    normal_ax2.plot(normal_mu_hat, normal_sigma_hat, 'rx', markersize=10, label='MLE Estimate')
    normal_ax2.plot(normal_true_mu, normal_true_sigma, 'g*', markersize=10, label='True Parameters')
    normal_ax2.legend()

    plt.colorbar(normal_contour, ax=normal_ax2, label='Log-Likelihood')
    plt.tight_layout()
    plt.gca()

    # relevant markdown for the results
    normal_explanation = mo.md(
        f"""
        ### Normal MLE Results

        **True parameters**: $\mu = {normal_true_mu:.3f}$, $\sigma^2 = {normal_true_var:.3f}$  
        **MLE estimates**: $\hat{{\mu}} = {normal_mu_hat:.3f}$, $\hat{{\sigma}}^2 = {normal_sigma2_hat:.3f}$

        The left plot shows the data histogram with the true Normal distribution (green) and the MLE-estimated distribution (red dashed).

        The right plot shows the log-likelihood function as a contour map in the $(\mu, \sigma)$ parameter space. The maximum likelihood estimates are marked with a red X, while the true parameters are marked with a green star.

        /// note
        Notice how the log-likelihood contour is more stretched along the σ axis than the μ axis. This indicates that we typically estimate the mean with greater precision than the standard deviation.
        ///

        /// tip
        Increase the sample size to see how the MLE estimates converge to the true parameter values!
        ///
        """
    )

    # plot and explanation together
    mo.vstack([
        normal_fig,
        normal_explanation
    ])
    return (
        normal_ax1,
        normal_ax2,
        normal_bins,
        normal_button_value,
        normal_contour,
        normal_data,
        normal_explanation,
        normal_fig,
        normal_i,
        normal_j,
        normal_ll,
        normal_ll_grid,
        normal_mle_pdf,
        normal_mu,
        normal_mu_grid,
        normal_mu_hat,
        normal_mu_range,
        normal_n,
        normal_sigma,
        normal_sigma2_hat,
        normal_sigma_grid,
        normal_sigma_hat,
        normal_sigma_range,
        normal_true_mu,
        normal_true_pdf,
        normal_true_sigma,
        normal_true_var,
        normal_x,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## MLE for Linear Regression

        Now let's look at a more practical example: using MLE to derive linear regression.

        ### The Model

        Consider a model where:
        - We have pairs of observations $(X_1, Y_1), (X_2, Y_2), \ldots, (X_n, Y_n)$
        - The relationship between $X$ and $Y$ follows: $Y = \theta X + Z$
        - $Z \sim N(0, \sigma^2)$ is random noise
        - Our goal is to estimate the parameter $\theta$

        This means that for a given $X_i$, the conditional distribution of $Y_i$ is:

        $$Y_i | X_i \sim N(\theta X_i, \sigma^2)$$

        ### Deriving the MLE

        Step 1: Write the likelihood function for each data point $(X_i, Y_i)$
        $$f(Y_i | X_i, \theta) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(Y_i - \theta X_i)^2}{2\sigma^2}\right)$$

        Step 2: Write the likelihood for all data
        $$\begin{align*}
        L(\theta) &= \prod_{i=1}^n f(Y_i, X_i | \theta) \\
        &= \prod_{i=1}^n f(Y_i | X_i, \theta) \cdot f(X_i)
        \end{align*}$$

        Since $f(X_i)$ doesn't depend on $\theta$, we can simplify:
        $$L(\theta) = \prod_{i=1}^n \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(Y_i - \theta X_i)^2}{2\sigma^2}\right) \cdot f(X_i)$$

        Step 3: Take the logarithm to get the log-likelihood
        $$\begin{align*}
        LL(\theta) &= \log \prod_{i=1}^n \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(Y_i - \theta X_i)^2}{2\sigma^2}\right) \cdot f(X_i) \\
        &= \sum_{i=1}^n \log\left[\frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(Y_i - \theta X_i)^2}{2\sigma^2}\right)\right] + \sum_{i=1}^n \log f(X_i) \\
        &= -\frac{n}{2} \log(2\pi\sigma^2) - \frac{1}{2\sigma^2} \sum_{i=1}^n (Y_i - \theta X_i)^2 + \sum_{i=1}^n \log f(X_i)
        \end{align*}$$

        Step 4: Since we only care about maximizing with respect to $\theta$, we can drop terms that don't contain $\theta$:
        $$\hat{\theta} = \underset{\theta}{\operatorname{argmax}} \left[ -\frac{1}{2\sigma^2} \sum_{i=1}^n (Y_i - \theta X_i)^2 \right]$$

        This is equivalent to:
        $$\hat{\theta} = \underset{\theta}{\operatorname{argmin}} \sum_{i=1}^n (Y_i - \theta X_i)^2$$

        Step 5: Find the value of $\theta$ that minimizes the sum of squared errors by setting the derivative to zero:
        $$\begin{align*}
        \frac{d}{d\theta} \sum_{i=1}^n (Y_i - \theta X_i)^2 &= 0 \\
        \sum_{i=1}^n -2X_i(Y_i - \theta X_i) &= 0 \\
        \sum_{i=1}^n X_i Y_i - \theta X_i^2 &= 0 \\
        \sum_{i=1}^n X_i Y_i &= \theta \sum_{i=1}^n X_i^2 \\
        \hat{\theta} &= \frac{\sum_{i=1}^n X_i Y_i}{\sum_{i=1}^n X_i^2}
        \end{align*}$$

        /// tip
        **Key Insight**: MLE for this simple linear model gives us the least squares estimator! This is an important connection between MLE and regression.
        ///
        """
    )
    return


@app.cell(hide_code=True)
def _(linear_controls):
    linear_controls.center()
    return


@app.cell(hide_code=True)
def _(
    linear_generate_button,
    linear_sample_size_slider,
    mo,
    noise_sigma_slider,
    np,
    plt,
    true_theta_slider,
):
    # linear model data calc when button is clicked
    linear_button_value = linear_generate_button.value

    # get parameter values
    linear_true_theta = true_theta_slider.value
    linear_noise_sigma = noise_sigma_slider.value
    linear_n = linear_sample_size_slider.value

    # generate x data (uniformly between -3 and 3)
    linear_X = np.random.uniform(-3, 3, size=linear_n)

    # generate y data according to the model y = θx + z
    linear_Z = np.random.normal(0, linear_noise_sigma, size=linear_n)
    linear_Y = linear_true_theta * linear_X + linear_Z

    # calculate mle estimate
    linear_theta_hat = np.sum(linear_X * linear_Y) / np.sum(linear_X**2)

    # calculate sse for different theta values
    linear_theta_range = np.linspace(linear_true_theta - 1.5, linear_true_theta + 1.5, 100)
    linear_sse_values = np.zeros_like(linear_theta_range)

    for linear_i, linear_theta in enumerate(linear_theta_range):
        linear_y_pred = linear_theta * linear_X
        linear_sse_values[linear_i] = np.sum((linear_Y - linear_y_pred)**2)

    # convert sse to log-likelihood (ignoring constant terms)
    linear_ll_values = -linear_sse_values / (2 * linear_noise_sigma**2)

    # create visualization
    linear_fig, (linear_ax1, linear_ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # plot scatter plot with regression lines
    linear_ax1.scatter(linear_X, linear_Y, color='blue', alpha=0.6, label='Data points')

    # plot range for regression lines
    linear_x_line = np.linspace(-3, 3, 100)

    # plot true and mle regression lines
    linear_ax1.plot(linear_x_line, linear_true_theta * linear_x_line, 'g-', linewidth=2, label=f'True: Y = {linear_true_theta:.2f}X')
    linear_ax1.plot(linear_x_line, linear_theta_hat * linear_x_line, 'r--', linewidth=2, label=f'MLE: Y = {linear_theta_hat:.2f}X')

    linear_ax1.set_xlabel('X')
    linear_ax1.set_ylabel('Y')
    linear_ax1.set_title(f'Linear Regression: {linear_n} data points')
    linear_ax1.grid(True, alpha=0.3)
    linear_ax1.legend()

    # plot log-likelihood function
    linear_ax2.plot(linear_theta_range, linear_ll_values, 'b-', linewidth=2)
    linear_ax2.axvline(x=linear_theta_hat, color='r', linestyle='--', label=f'MLE: θ = {linear_theta_hat:.3f}')
    linear_ax2.axvline(x=linear_true_theta, color='g', linestyle='--', label=f'True: θ = {linear_true_theta:.3f}')
    linear_ax2.set_xlabel('θ (slope parameter)')
    linear_ax2.set_ylabel('Log-Likelihood')
    linear_ax2.set_title('Log-Likelihood Function')
    linear_ax2.grid(True, alpha=0.3)
    linear_ax2.legend()

    plt.tight_layout()
    plt.gca()

    # relevant markdown to explain results
    linear_explanation = mo.md(
        f"""
        ### Linear Regression MLE Results

        **True parameter**: $\\theta = {linear_true_theta:.3f}$  
        **MLE estimate**: $\\hat{{\\theta}} = {linear_theta_hat:.3f}$

        The left plot shows the scatter plot of data points with the true regression line (green) and the MLE-estimated regression line (red dashed).

        The right plot shows the log-likelihood function for different values of $\\theta$. The maximum likelihood estimate is marked with a red dashed line, and the true parameter is marked with a green dashed line.

        /// note
        The MLE estimate $\\hat{{\\theta}} = \\frac{{\\sum_{{i=1}}^n X_i Y_i}}{{\\sum_{{i=1}}^n X_i^2}}$ minimizes the sum of squared errors between the predicted and actual Y values.
        ///

        /// tip
        Try increasing the noise level to see how it affects the precision of the estimate!
        ///
        """
    )

    # show plot and explanation
    mo.vstack([
        linear_fig,
        linear_explanation
    ])
    return (
        linear_X,
        linear_Y,
        linear_Z,
        linear_ax1,
        linear_ax2,
        linear_button_value,
        linear_explanation,
        linear_fig,
        linear_i,
        linear_ll_values,
        linear_n,
        linear_noise_sigma,
        linear_sse_values,
        linear_theta,
        linear_theta_hat,
        linear_theta_range,
        linear_true_theta,
        linear_x_line,
        linear_y_pred,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Interactive Concept: Likelihood vs. Probability

        To better understand the distinction between likelihood and probability, let's create an interactive visualization. This concept is crucial for understanding why MLE works.
        """
    )
    return


@app.cell(hide_code=True)
def _(concept_controls):
    concept_controls.center()
    return


@app.cell(hide_code=True)
def _(concept_dist_type, mo, np, perspective_selector, plt, stats):
    # current distribution type
    concept_dist_type_value = concept_dist_type.value

    # view mode from dropdown
    concept_view_mode = "likelihood" if perspective_selector.value == "Likelihood Perspective" else "probability"

    # visualization based on distribution type
    concept_fig, concept_ax = plt.subplots(figsize=(10, 6))

    if concept_dist_type_value == "Normal":
        if concept_view_mode == "probability":
            # probability perspective: fixed parameters, varying data
            concept_mu = 0      # fixed parameter
            concept_sigma = 1   # fixed parameter

            # generate x values for the pdf
            concept_x = np.linspace(-4, 4, 1000)

            # plot pdf
            concept_pdf = stats.norm.pdf(concept_x, concept_mu, concept_sigma)
            concept_ax.plot(concept_x, concept_pdf, 'b-', linewidth=2, label='PDF: N(0, 1)')

            # highlight specific data values
            concept_data_points = [-2, -1, 0, 1, 2]
            concept_colors = ['#FF9999', '#FFCC99', '#99FF99', '#99CCFF', '#CC99FF']

            for concept_i, concept_data in enumerate(concept_data_points):
                concept_prob = stats.norm.pdf(concept_data, concept_mu, concept_sigma)
                concept_ax.plot([concept_data, concept_data], [0, concept_prob], concept_colors[concept_i], linewidth=2)
                concept_ax.scatter(concept_data, concept_prob, color=concept_colors[concept_i], s=50, 
                           label=f'P(X={concept_data}|μ=0,σ=1) = {concept_prob:.3f}')

            concept_ax.set_xlabel('Data (x)')
            concept_ax.set_ylabel('Probability Density')
            concept_ax.set_title('Probability Perspective: Fixed Parameters (μ=0, σ=1), Different Data Points')

        else:  # likelihood perspective
            # likelihood perspective: fixed data, varying parameters
            concept_data_point = 1.5  # fixed observed data

            # different possible parameter values (means)
            concept_mus = [-1, 0, 1, 2, 3]
            concept_sigma = 1

            # generate x values for multiple pdfs
            concept_x = np.linspace(-4, 6, 1000)

            concept_colors = ['#FF9999', '#FFCC99', '#99FF99', '#99CCFF', '#CC99FF']

            for concept_i, concept_mu in enumerate(concept_mus):
                concept_pdf = stats.norm.pdf(concept_x, concept_mu, concept_sigma)
                concept_ax.plot(concept_x, concept_pdf, color=concept_colors[concept_i], linewidth=2, alpha=0.7,
                        label=f'N({concept_mu}, 1)')

                # mark the likelihood of the data point for this param
                concept_likelihood = stats.norm.pdf(concept_data_point, concept_mu, concept_sigma)
                concept_ax.plot([concept_data_point, concept_data_point], [0, concept_likelihood], concept_colors[concept_i], linewidth=2)
                concept_ax.scatter(concept_data_point, concept_likelihood, color=concept_colors[concept_i], s=50, 
                           label=f'L(μ={concept_mu}|X=1.5) = {concept_likelihood:.3f}')

            # add vertical line at the observed data point
            concept_ax.axvline(x=concept_data_point, color='black', linestyle='--', 
                       label=f'Observed data: X=1.5')

            concept_ax.set_xlabel('Data (x)')
            concept_ax.set_ylabel('Probability Density / Likelihood')
            concept_ax.set_title('Likelihood Perspective: Fixed Data Point (X=1.5), Different Parameter Values')

    elif concept_dist_type_value == "Bernoulli":
        if concept_view_mode == "probability":
            # probability perspective: fixed parameter, two possible data values
            concept_p = 0.3  # fixed parameter

            # bar chart for p(x=0) and p(x=1)
            concept_ax.bar([0, 1], [1-concept_p, concept_p], width=0.4, color=['#99CCFF', '#FF9999'], 
                   alpha=0.7, label=f'PMF: Bernoulli({concept_p})')

            # text showing probabilities
            concept_ax.text(0, (1-concept_p)/2, f'P(X=0|p={concept_p}) = {1-concept_p:.3f}', ha='center', va='center', fontweight='bold')
            concept_ax.text(1, concept_p/2, f'P(X=1|p={concept_p}) = {concept_p:.3f}', ha='center', va='center', fontweight='bold')

            concept_ax.set_xlabel('Data (x)')
            concept_ax.set_ylabel('Probability')
            concept_ax.set_xticks([0, 1])
            concept_ax.set_xticklabels(['X=0', 'X=1'])
            concept_ax.set_ylim(0, 1)
            concept_ax.set_title('Probability Perspective: Fixed Parameter (p=0.3), Different Data Values')

        else:  # likelihood perspective
            # likelihood perspective: fixed data, varying parameter
            concept_data_point = 1  # fixed observed data (success)

            # different possible parameter values
            concept_p_values = np.linspace(0.01, 0.99, 100)

            # calculate likelihood for each p value
            if concept_data_point == 1:
                # for x=1, likelihood is p
                concept_likelihood = concept_p_values
                concept_ax.plot(concept_p_values, concept_likelihood, 'b-', linewidth=2, 
                        label=f'L(p|X=1) = p')

                # highlight specific values
                concept_highlight_ps = [0.2, 0.5, 0.8]
                concept_colors = ['#FF9999', '#99FF99', '#99CCFF']

                for concept_i, concept_p in enumerate(concept_highlight_ps):
                    concept_ax.plot([concept_p, concept_p], [0, concept_p], concept_colors[concept_i], linewidth=2)
                    concept_ax.scatter(concept_p, concept_p, color=concept_colors[concept_i], s=50, 
                               label=f'L(p={concept_p}|X=1) = {concept_p:.3f}')

                concept_ax.set_title('Likelihood Perspective: Fixed Data Point (X=1), Different Parameter Values')

            else:  # x=0
                # for x = 0, likelihood is (1-p)
                concept_likelihood = 1 - concept_p_values
                concept_ax.plot(concept_p_values, concept_likelihood, 'r-', linewidth=2, 
                        label=f'L(p|X=0) = (1-p)')

                # highlight some specific values
                concept_highlight_ps = [0.2, 0.5, 0.8]
                concept_colors = ['#FF9999', '#99FF99', '#99CCFF']

                for concept_i, concept_p in enumerate(concept_highlight_ps):
                    concept_ax.plot([concept_p, concept_p], [0, 1-concept_p], concept_colors[concept_i], linewidth=2)
                    concept_ax.scatter(concept_p, 1-concept_p, color=concept_colors[concept_i], s=50, 
                               label=f'L(p={concept_p}|X=0) = {1-concept_p:.3f}')

                concept_ax.set_title('Likelihood Perspective: Fixed Data Point (X=0), Different Parameter Values')

            concept_ax.set_xlabel('Parameter (p)')
            concept_ax.set_ylabel('Likelihood')
            concept_ax.set_xlim(0, 1)
            concept_ax.set_ylim(0, 1)

    elif concept_dist_type_value == "Poisson":
        if concept_view_mode == "probability":
            # probability perspective: fixed parameter, different data values
            concept_lam = 2.5  # fixed parameter

            # pmf for different x values plot
            concept_x_values = np.arange(0, 10)
            concept_pmf_values = stats.poisson.pmf(concept_x_values, concept_lam)

            concept_ax.bar(concept_x_values, concept_pmf_values, width=0.4, color='#99CCFF', 
                   alpha=0.7, label=f'PMF: Poisson({concept_lam})')

            # highlight a few specific values
            concept_highlight_xs = [1, 2, 3, 4]
            concept_colors = ['#FF9999', '#99FF99', '#FFCC99', '#CC99FF']

            for concept_i, concept_x in enumerate(concept_highlight_xs):
                concept_prob = stats.poisson.pmf(concept_x, concept_lam)
                concept_ax.scatter(concept_x, concept_prob, color=concept_colors[concept_i], s=50, 
                           label=f'P(X={concept_x}|λ={concept_lam}) = {concept_prob:.3f}')

            concept_ax.set_xlabel('Data (x)')
            concept_ax.set_ylabel('Probability')
            concept_ax.set_xticks(concept_x_values)
            concept_ax.set_title('Probability Perspective: Fixed Parameter (λ=2.5), Different Data Values')

        else:  # likelihood perspective
            # likelihood perspective: fixed data, varying parameter
            concept_data_point = 4  # fixed observed data

            # different possible param values
            concept_lambda_values = np.linspace(0.1, 8, 100)

            # calc likelihood for each lambda value
            concept_likelihood = stats.poisson.pmf(concept_data_point, concept_lambda_values)

            concept_ax.plot(concept_lambda_values, concept_likelihood, 'b-', linewidth=2, 
                    label=f'L(λ|X={concept_data_point})')

            # highlight some specific values
            concept_highlight_lambdas = [1, 2, 4, 6]
            concept_colors = ['#FF9999', '#99FF99', '#99CCFF', '#FFCC99']

            for concept_i, concept_lam in enumerate(concept_highlight_lambdas):
                concept_like_val = stats.poisson.pmf(concept_data_point, concept_lam)
                concept_ax.plot([concept_lam, concept_lam], [0, concept_like_val], concept_colors[concept_i], linewidth=2)
                concept_ax.scatter(concept_lam, concept_like_val, color=concept_colors[concept_i], s=50, 
                           label=f'L(λ={concept_lam}|X={concept_data_point}) = {concept_like_val:.3f}')

            concept_ax.set_xlabel('Parameter (λ)')
            concept_ax.set_ylabel('Likelihood')
            concept_ax.set_title(f'Likelihood Perspective: Fixed Data Point (X={concept_data_point}), Different Parameter Values')

    concept_ax.legend(loc='best', fontsize=9)
    concept_ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.gca()

    # relevant explanation based on view mode
    if concept_view_mode == "probability":
        concept_explanation = mo.md(
            f"""
            ### Probability Perspective

            In the **probability perspective**, the parameters of the distribution are **fixed and known**, and we calculate the probability (or density) for **different possible data values**.

            For the {concept_dist_type_value} distribution, we've fixed the parameter{'s' if concept_dist_type_value == 'Normal' else ''} and shown the probability of observing different outcomes.

            This is the typical perspective when:

            - We know the true parameters of a distribution
            - We want to calculate the probability of different outcomes
            - We make predictions based on our model

            **Mathematical notation**: $P(X = x | \theta)$
            """
        )
    else:  # likelihood perspective
        concept_explanation = mo.md(
            f"""
            ### Likelihood Perspective

            In the **likelihood perspective**, the observed data is **fixed and known**, and we calculate how likely different parameter values are to have generated that data.

            For the {concept_dist_type_value} distribution, we've fixed the observed data point{'s' if concept_dist_type_value == 'Normal' else ''} and shown the likelihood of different parameter values.

            This is the perspective used in MLE:

            - We have observed data
            - We don't know the true parameters
            - We want to find parameters that best explain our observations

            **Mathematical notation**: $L(\theta | X = x)$

            /// tip
            The value of $\\theta$ that maximizes this likelihood function is the MLE estimate $\\hat{{\\theta}}$!
            ///
            """
        )

    # Display plot and explanation together
    mo.vstack([
        concept_fig,
        concept_explanation
    ])
    return (
        concept_ax,
        concept_colors,
        concept_data,
        concept_data_point,
        concept_data_points,
        concept_dist_type_value,
        concept_explanation,
        concept_fig,
        concept_highlight_lambdas,
        concept_highlight_ps,
        concept_highlight_xs,
        concept_i,
        concept_lam,
        concept_lambda_values,
        concept_like_val,
        concept_likelihood,
        concept_mu,
        concept_mus,
        concept_p,
        concept_p_values,
        concept_pdf,
        concept_pmf_values,
        concept_prob,
        concept_sigma,
        concept_view_mode,
        concept_x,
        concept_x_values,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 🤔 Test Your Understanding

        Which of the following statements about Maximum Likelihood Estimation are correct? Click each statement to check your answer.

        /// details | Probability and likelihood use the same formulas, but probability measures the chance of data given parameters, while likelihood measures how likely parameters are given data.
        ✅ **Correct!** 

        Probability measures how likely it is to observe particular data when we know the parameters. Likelihood measures how likely particular parameter values are, given observed data.

        Mathematically, probability is $P(X=x|\theta)$ while likelihood is $L(\theta|X=x)$. They use the same formula, but with different perspectives on what's fixed and what varies.
        ///

        /// details | We use log-likelihood instead of likelihood because it's mathematically simpler and numerically more stable.
        ✅ **Correct!**

        We work with log-likelihood for several reasons:
        1. It converts products into sums, which is easier to work with mathematically
        2. It avoids numerical underflow when multiplying many small probabilities
        3. Logarithm is a monotonically increasing function, so the maximum of the likelihood occurs at the same parameter values as the maximum of the log-likelihood
        ///

        /// details | For a Bernoulli distribution, the MLE for parameter p is the sample mean of the observations.
        ✅ **Correct!**

        For a Bernoulli distribution with parameter $p$, given $n$ independent samples $X_1, X_2, \ldots, X_n$, the MLE estimator is:

        $$\hat{p} = \frac{\sum_{i=1}^n X_i}{n}$$

        This is simply the sample mean, or the proportion of successes (1s) in the data.
        ///

        /// details | For a Normal distribution, MLE gives unbiased estimates for both mean and variance parameters.
        ❌ **Incorrect.**

        While the MLE for the mean ($\hat{\mu} = \frac{1}{n}\sum_{i=1}^n X_i$) is unbiased, the MLE for variance:

        $$\hat{\sigma}^2 = \frac{1}{n}\sum_{i=1}^n (X_i - \hat{\mu})^2$$

        is a biased estimator. It uses $n$ in the denominator rather than $n-1$ used in the unbiased estimator.
        ///

        /// details | MLE estimators are always unbiased regardless of the distribution.
        ❌ **Incorrect.**

        MLE is not always unbiased, though it often is asymptotically unbiased (meaning the bias approaches zero as the sample size increases).

        A notable example is the MLE estimator for the variance of a Normal distribution:
        $$\hat{\sigma}^2 = \frac{1}{n}\sum_{i=1}^n (X_i - \hat{\mu})^2$$

        This estimator is biased, which is why we often use the unbiased estimator:
        $$s^2 = \frac{1}{n-1}\sum_{i=1}^n (X_i - \hat{\mu})^2$$

        Despite occasional bias, MLE estimators have many desirable properties, including consistency and asymptotic efficiency.
        ///
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Summary

        Maximum Likelihood Estimation really is one of those elegant ideas that sits at the core of modern statistics. When you get down to it, MLE is just about finding the most plausible explanation for the data we've observed. It's like being a detective - you have some clues (your data), and you're trying to piece together the most likely story (your parameters) that explains them.

        We've seen how this works with different distributions. For the Bernoulli, it simply gives us the sample proportion. For the Normal, it gives us the sample mean and a slightly biased estimate of variance. And for linear regression, it provides a mathematical justification for the least squares method that everyone learns in basic stats classes.

        What makes MLE so useful in practice is that it tends to give us estimates with good properties. As you collect more data, the estimates generally get closer to the true values (consistency) and do so efficiently. That's why MLE is everywhere in statistics and machine learning - from simple regression models to complex neural networks.

        The most important takeaway? Next time you're fitting a model to data, remember that you're not just following a recipe - you're finding the parameters that make your observed data most likely to have occurred. That's the essence of statistical inference.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Further Reading

        If you're curious to dive deeper into this topic, check out "Statistical Inference" by Casella and Berger - it's the classic text that many statisticians learned from. For a more machine learning angle, Bishop's "Pattern Recognition and Machine Learning" shows how MLE connects to more advanced topics like EM algorithms and Bayesian methods.

        Beyond the basics we've covered, you might explore Bayesian estimation (which incorporates prior knowledge), Fisher Information (which tells us how precisely we can estimate parameters), or the EM algorithm (for when we have missing data or latent variables). Each of these builds on the foundation of likelihood that we've established here.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Appendix (helper functions and imports)""")
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats
    import plotly.graph_objects as go
    import polars as pl
    from matplotlib import cm

    # Set a consistent random seed for reproducibility
    np.random.seed(42)

    # Set a nice style for matplotlib
    plt.style.use('seaborn-v0_8-darkgrid')
    return cm, go, np, pl, plt, stats


@app.cell(hide_code=True)
def _(mo):
    # Create interactive elements
    true_p_slider = mo.ui.slider(
        start =0.01, 
        stop =0.99, 
        value=0.3, 
        step=0.01, 
        label="True probability (p)"
    )

    sample_size_slider = mo.ui.slider(
        start =10, 
        stop =1000, 
        value=100, 
        step=10, 
        label="Sample size (n)"
    )

    generate_button = mo.ui.button(label="Generate New Sample", kind="success")

    controls = mo.vstack([
        mo.vstack([true_p_slider, sample_size_slider]), 
        generate_button
    ], justify="space-between")
    return controls, generate_button, sample_size_slider, true_p_slider


@app.cell(hide_code=True)
def _(mo):
    # Create interactive elements for Normal distribution
    true_mu_slider = mo.ui.slider(
        start =-5, 
        stop =5, 
        value=0, 
        step=0.1, 
        label="True mean (μ)"
    )

    true_sigma_slider = mo.ui.slider(
        start =0.5, 
        stop =3, 
        value=1, 
        step=0.1, 
        label="True standard deviation (σ)"
    )

    normal_sample_size_slider = mo.ui.slider(
        start =10, 
        stop =500, 
        value=50, 
        step=10, 
        label="Sample size (n)"
    )

    normal_generate_button = mo.ui.button(label="Generate New Sample", kind="warn")

    normal_controls = mo.hstack([
        mo.vstack([true_mu_slider, true_sigma_slider, normal_sample_size_slider]), 
        normal_generate_button
    ], justify="space-between")
    return (
        normal_controls,
        normal_generate_button,
        normal_sample_size_slider,
        true_mu_slider,
        true_sigma_slider,
    )


@app.cell(hide_code=True)
def _(mo):
    # Create interactive elements for linear regression
    true_theta_slider = mo.ui.slider(
        start =-2, 
        stop =2, 
        value=0.5, 
        step=0.1, 
        label="True slope (θ)"
    )

    noise_sigma_slider = mo.ui.slider(
        start =0.1, 
        stop =2, 
        value=0.5, 
        step=0.1, 
        label="Noise level (σ)"
    )

    linear_sample_size_slider = mo.ui.slider(
        start =10, 
        stop =200, 
        value=50, 
        step=10, 
        label="Sample size (n)"
    )

    linear_generate_button = mo.ui.button(label="Generate New Sample", kind="warn")

    linear_controls = mo.hstack([
        mo.vstack([true_theta_slider, noise_sigma_slider, linear_sample_size_slider]), 
        linear_generate_button
    ], justify="space-between")
    return (
        linear_controls,
        linear_generate_button,
        linear_sample_size_slider,
        noise_sigma_slider,
        true_theta_slider,
    )


@app.cell(hide_code=True)
def _(mo):
    # Interactive elements for likelihood vs probability demo
    concept_dist_type = mo.ui.dropdown(
        options=["Normal", "Bernoulli", "Poisson"],
        value="Normal",
        label="Distribution"
    )

    # Replace buttons with a simple dropdown selector
    perspective_selector = mo.ui.dropdown(
        options=["Probability Perspective", "Likelihood Perspective"],
        value="Probability Perspective",
        label="View"
    )

    concept_controls = mo.vstack([
        mo.hstack([concept_dist_type, perspective_selector])
    ])
    return concept_controls, concept_dist_type, perspective_selector


if __name__ == "__main__":
    app.run()
