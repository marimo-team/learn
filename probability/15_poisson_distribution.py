# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "matplotlib==3.10.0",
#     "numpy==2.2.4",
#     "scipy==1.15.2",
#     "altair==5.2.0",
#     "wigglystuff==0.1.10",
#     "pandas==2.2.3",
# ]
# ///

import marimo

__generated_with = "0.11.24"
app = marimo.App(width="medium", app_title="Poisson Distribution")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Poisson Distribution

        _This notebook is a computational companion to ["Probability for Computer Scientists"](https://chrispiech.github.io/probabilityForComputerScientists/en/part2/poisson/), by Stanford professor Chris Piech._

        A Poisson random variable gives the probability of a given number of events in a fixed interval of time (or space). It makes the Poisson assumption that events occur with a known constant mean rate and independently of the time since the last event.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Poisson Random Variable Definition

        $X \sim \text{Poisson}(\lambda)$ represents a Poisson random variable where:

        - $X$ is our random variable (number of events)
        - $\text{Poisson}$ indicates it follows a Poisson distribution
        - $\lambda$ is the rate parameter (average number of events per time interval)

        ```
        X ~ Poisson(λ)
         ↑     ↑    ↑
         |     |    +-- Rate parameter:
         |     |        average number of
         |     |        events per interval
         |     +-- Indicates Poisson
         |         distribution
         |
        Our random variable
          counting number of events
        ```

        The Poisson distribution is particularly useful when:

        1. Events occur independently of each other
        2. The average rate of occurrence is constant
        3. Two events cannot occur at exactly the same instant
        4. The probability of an event is proportional to the length of the time interval
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Properties of Poisson Distribution

        | Property | Formula |
        |----------|---------|
        | Notation | $X \sim \text{Poisson}(\lambda)$ |
        | Description | Number of events in a fixed time frame if (a) events occur with a constant mean rate and (b) they occur independently of time since last event |
        | Parameters | $\lambda \in \mathbb{R}^{+}$, the constant average rate |
        | Support | $x \in \{0, 1, \dots\}$ |
        | PMF equation | $P(X=x) = \frac{\lambda^x e^{-\lambda}}{x!}$ |
        | Expectation | $E[X] = \lambda$ |
        | Variance | $\text{Var}(X) = \lambda$ |

        Note that unlike many other distributions, the Poisson distribution's mean and variance are equal, both being $\lambda$.

        Let's explore how the Poisson distribution changes with different rate parameters.
        """
    )
    return


@app.cell(hide_code=True)
def _(TangleSlider, mo):
    # Create interactive elements using TangleSlider
    lambda_slider = mo.ui.anywidget(TangleSlider(
        amount=5, 
        min_value=0.1, 
        max_value=20, 
        step=0.1,
        digits=1,
        suffix=" events"
    ))

    # interactive controls
    _controls = mo.vstack([
        mo.md("### Adjust the Rate Parameter to See How Poisson Distribution Changes"),
        mo.hstack([
            mo.md("**Rate parameter (λ):** "),
            lambda_slider,
            mo.md("**events per interval.** Higher values shift the distribution rightward and make it more spread out.")
        ], justify="start"),
    ])
    _controls
    return (lambda_slider,)


@app.cell(hide_code=True)
def _(lambda_slider, np, plt, stats):
    _lambda = lambda_slider.amount

    # PMF for values
    _max_x = max(20, int(_lambda * 3))  # Show at least up to 3*lambda
    _x = np.arange(0, _max_x + 1)
    _pmf = stats.poisson.pmf(_x, _lambda)

    # Relevant key statistics
    _mean = _lambda  # For Poisson, mean = lambda
    _variance = _lambda  # For Poisson, variance = lambda
    _std_dev = np.sqrt(_variance)

    # plot
    _fig, _ax = plt.subplots(figsize=(10, 6))

    # PMF as bars
    _ax.bar(_x, _pmf, color='royalblue', alpha=0.7, label=f'PMF: P(X=k)')

    #  for the PMF values
    _ax.plot(_x, _pmf, 'ro-', alpha=0.6, label='PMF line')

    # Vertical lines - mean and key values
    _ax.axvline(x=_mean, color='green', linestyle='--', linewidth=2, 
               label=f'Mean: {_mean:.2f}')

    # Stdev region
    _ax.axvspan(_mean - _std_dev, _mean + _std_dev, alpha=0.2, color='green',
               label=f'±1 Std Dev: {_std_dev:.2f}')

    _ax.set_xlabel('Number of Events (k)')
    _ax.set_ylabel('Probability: P(X=k)')
    _ax.set_title(f'Poisson Distribution with λ={_lambda:.1f}')

    # annotations
    _ax.annotate(f'E[X] = {_mean:.2f}', 
                xy=(_mean, stats.poisson.pmf(int(_mean), _lambda)), 
                xytext=(_mean + 1, max(_pmf) * 0.8),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1))

    _ax.annotate(f'Var(X) = {_variance:.2f}', 
                xy=(_mean, stats.poisson.pmf(int(_mean), _lambda) / 2), 
                xytext=(_mean + 1, max(_pmf) * 0.6),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1))

    _ax.grid(alpha=0.3)
    _ax.legend()

    plt.tight_layout()
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Poisson Intuition: Relation to Binomial Distribution

        The Poisson distribution can be derived as a limiting case of the [binomial distribution](http://marimo.app/https://github.com/marimo-team/learn/blob/main/probability/14_binomial_distribution.py). 

        Let's work on a practical example: predicting the number of ride-sharing requests in a specific area over a one-minute interval. From historical data, we know that the average number of requests per minute is $\lambda = 5$.

        We could approximate this using a binomial distribution by dividing our minute into smaller intervals. For example, we can divide a minute into 60 seconds and treat each second as a [Bernoulli trial](http://marimo.app/https://github.com/marimo-team/learn/blob/main/probability/13_bernoulli_distribution.py) - either there's a request (success) or there isn't (failure).

        Let's visualize this concept:
        """
    )
    return


@app.cell(hide_code=True)
def _(fig_to_image, mo, plt):
    # Create a visualization of dividing a minute into 60 seconds
    _fig, _ax = plt.subplots(figsize=(12, 2))

    # Example events at 2.75s and 7.12s
    _events = [2.75, 7.12]

    # Create an array of 60 rectangles
    for i in range(60):
        _color = 'royalblue' if any(i <= e < i+1 for e in _events) else 'lightgray'
        _ax.add_patch(plt.Rectangle((i, 0), 0.9, 1, color=_color))

    # markers for events
    for e in _events:
        _ax.plot(e, 0.5, 'ro', markersize=10)

    # labels
    _ax.set_xlim(0, 60)
    _ax.set_ylim(0, 1)
    _ax.set_yticks([])
    _ax.set_xticks([0, 15, 30, 45, 60])
    _ax.set_xticklabels(['0s', '15s', '30s', '45s', '60s'])
    _ax.set_xlabel('Time (seconds)')
    _ax.set_title('One Minute Divided into 60 Second Intervals')

    plt.tight_layout()

    # Convert plot to image for display
    _img = mo.image(fig_to_image(_fig), width="100%")

    # explanation
    _explanation = mo.md(
        r"""
        In this visualization:
        - Each rectangle represents a 1-second interval
        - Blue rectangles indicate intervals where an event occurred
        - Red dots show the actual event times (2.75s and 7.12s)

        If we treat this as a binomial experiment with 60 trials (seconds), we can calculate probabilities using the binomial PMF. But there's a problem: what if multiple events occur within the same second? To address this, we can divide our minute into smaller intervals.
        """
    )
    return e, i


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The total number of requests received over the minute can be approximated as the sum of the sixty indicator variables, which conveniently matches the description of a binomial — a sum of Bernoullis. 

        Specifically, if we define $X$ to be the number of requests in a minute, $X$ is a binomial with $n=60$ trials. What is the probability, $p$, of a success on a single trial? To make the expectation of $X$ equal the observed historical average $\lambda$, we should choose $p$ so that:

        \begin{align}
        \lambda &= E[X] && \text{Expectation matches historical average} \\
        \lambda &= n \cdot p && \text{Expectation of a Binomial is } n \cdot p \\
        p &= \frac{\lambda}{n} && \text{Solving for $p$}
        \end{align}

        In this case, since $\lambda=5$ and $n=60$, we should choose $p=\frac{5}{60}=\frac{1}{12}$ and state that $X \sim \text{Bin}(n=60, p=\frac{5}{60})$. Now we can calculate the probability of different numbers of requests using the binomial PMF:

        $P(X = x) = {n \choose x} p^x (1-p)^{n-x}$

        For example:

        \begin{align}
        P(X=1) &= {60 \choose 1} (5/60)^1 (55/60)^{60-1} \approx 0.0295 \\
        P(X=2) &= {60 \choose 2} (5/60)^2 (55/60)^{60-2} \approx 0.0790 \\
        P(X=3) &= {60 \choose 3} (5/60)^3 (55/60)^{60-3} \approx 0.1389
        \end{align}

        This is a good approximation, but it doesn't account for the possibility of multiple events in a single second. One solution is to divide our minute into even more fine-grained intervals. Let's try 600 deciseconds (tenths of a second):
        """
    )
    return


@app.cell(hide_code=True)
def _(e, fig_to_image, mo, plt):
    # Create a visualization of dividing a minute into 600 deciseconds
    # (Just showing the first 100 for clarity)
    _fig, _ax = plt.subplots(figsize=(12, 2))

    # Example events at 2.75s and 7.12s (convert to deciseconds)
    _events = [27.5, 71.2]

    # Create a representative portion of the 600 rectangles (first 100)
    for _i in range(100):
        _color = 'royalblue' if any(_i <= _e < _i + 1 for _e in _events) else 'lightgray'
        _ax.add_patch(plt.Rectangle((_i, 0), 0.9, 1, color=_color))

    # Add markers for events
    for _e in _events:
        if _e < 100:  # Only show events in our visible range
            _ax.plot(e, 0.5, 'ro', markersize=10)

    # Add labels
    _ax.set_xlim(0, 100)
    _ax.set_ylim(0, 1)
    _ax.set_yticks([])
    _ax.set_xticks([0, 20, 40, 60, 80, 100])
    _ax.set_xticklabels(['0s', '2s', '4s', '6s', '8s', '10s'])
    _ax.set_xlabel('Time (first 10 seconds shown)')
    _ax.set_title('One Minute Divided into 600 Decisecond Intervals (first 100 shown)')

    plt.tight_layout()

    # Convert plot to image for display
    _img = mo.image(fig_to_image(_fig), width="100%")

    # Add explanation
    _explanation = mo.md(
        r"""
        With $n=600$ and $p=\frac{5}{600}=\frac{1}{120}$, we can recalculate our probabilities:

        \begin{align}
        P(X=1) &= {600 \choose 1} (5/600)^1 (595/600)^{600-1} \approx 0.0333 \\
        P(X=2) &= {600 \choose 2} (5/600)^2 (595/600)^{600-2} \approx 0.0837 \\
        P(X=3) &= {600 \choose 3} (5/600)^3 (595/600)^{600-3} \approx 0.1402
        \end{align}

        As we make our intervals smaller (increasing $n$), our approximation becomes more accurate.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## The Binomial Distribution in the Limit

        What happens if we continue dividing our time interval into smaller and smaller pieces? Let's explore how the probabilities change as we increase the number of intervals:
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    # slider for number of intervals
    intervals_slider = mo.ui.slider(
        start = 60, 
        stop = 10000,
        step=100,
        value=600,
        label="Number of intervals to divide a minute")
    return (intervals_slider,)


@app.cell(hide_code=True)
def _(intervals_slider):
    intervals_slider
    return


@app.cell(hide_code=True)
def _(intervals_slider, np, pd, plt, stats):
    # number of intervals from the slider
    n = intervals_slider.value
    _lambda = 5  # Fixed lambda for our example
    p = _lambda / n

    # Calculate the binomial probabilities
    _x_values = np.arange(0, 15)
    _binom_pmf = stats.binom.pmf(_x_values, n, p)

    # Calculate the true Poisson probabilities
    _poisson_pmf = stats.poisson.pmf(_x_values, _lambda)

    # Create a DataFrame for comparison
    df = pd.DataFrame({
        'Events': _x_values,
        f'Binomial(n={n}, p={p:.6f})': _binom_pmf,
        f'Poisson(λ=5)': _poisson_pmf,
        'Difference': np.abs(_binom_pmf - _poisson_pmf)
    })

    # Plot both PMFs
    fig, _ax = plt.subplots(figsize=(10, 6))

    # Bar plot for the binomial
    _ax.bar(_x_values - 0.2, _binom_pmf, width=0.4, alpha=0.7, 
           color='royalblue', label=f'Binomial(n={n}, p={p:.6f})')

    # Bar plot for the Poisson
    _ax.bar(_x_values + 0.2, _poisson_pmf, width=0.4, alpha=0.7,
           color='crimson', label='Poisson(λ=5)')

    # Add labels and title
    _ax.set_xlabel('Number of Events (k)')
    _ax.set_ylabel('Probability')
    _ax.set_title(f'Comparison of Binomial and Poisson PMFs with n={n}')
    _ax.legend()
    _ax.set_xticks(_x_values)
    _ax.grid(alpha=0.3)

    plt.tight_layout()
    return df, fig, n, p


@app.cell(hide_code=True)
def _(df, fig, fig_to_image, mo, n, p):
    # table of values
    _styled_df = df.style.format({
        f'Binomial(n={n}, p={p:.6f})': '{:.6f}',
        f'Poisson(λ=5)': '{:.6f}',
        'Difference': '{:.6f}'
    })

    # Calculate the maximum absolute difference
    _max_diff = df['Difference'].max()

    # output
    _chart = mo.image(fig_to_image(fig), width="100%")
    _explanation = mo.md(f"**Maximum absolute difference between distributions: {_max_diff:.6f}**")
    _table = mo.ui.table(df)

    mo.vstack([_chart, _explanation, _table])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        As you can see from the interactive comparison above, as the number of intervals increases, the binomial distribution approaches the Poisson distribution! This is not a coincidence - the Poisson distribution is actually the limiting case of the binomial distribution when:

        - The number of trials $n$ approaches infinity
        - The probability of success $p$ approaches zero
        - The product $np = \lambda$ remains constant

        This relationship is why the Poisson distribution is so useful - it's easier to work with than a binomial with a very large number of trials and a very small probability of success.

        ## Derivation of the Poisson PMF

        Let's derive the Poisson PMF by taking the limit of the binomial PMF as $n \to \infty$. We start with:

        $P(X=x) = \lim_{n \rightarrow \infty} {n \choose x} (\lambda / n)^x(1-\lambda/n)^{n-x}$

        While this expression looks intimidating, it simplifies nicely:

        \begin{align}
        P(X=x) 
        &= \lim_{n \rightarrow \infty} {n \choose x} (\lambda / n)^x(1-\lambda/n)^{n-x}
            && \text{Start: binomial in the limit}\\
        &= \lim_{n \rightarrow \infty}
            {n \choose x} \cdot
            \frac{\lambda^x}{n^x} \cdot
            \frac{(1-\lambda/n)^{n}}{(1-\lambda/n)^{x}} 
            && \text{Expanding the power terms} \\
        &= \lim_{n \rightarrow \infty}
            \frac{n!}{(n-x)!x!} \cdot
            \frac{\lambda^x}{n^x} \cdot
            \frac{(1-\lambda/n)^{n}}{(1-\lambda/n)^{x}} 
            && \text{Expanding the binomial term} \\
        &= \lim_{n \rightarrow \infty}
            \frac{n!}{(n-x)!x!} \cdot
            \frac{\lambda^x}{n^x} \cdot
            \frac{e^{-\lambda}}{(1-\lambda/n)^{x}} 
            && \text{Using limit rule } \lim_{n \rightarrow \infty}(1-\lambda/n)^{n} = e^{-\lambda}\\
        &= \lim_{n \rightarrow \infty}
            \frac{n!}{(n-x)!x!} \cdot
            \frac{\lambda^x}{n^x} \cdot
            \frac{e^{-\lambda}}{1} 
            && \text{As } n \to \infty \text{, } \lambda/n \to 0\\
        &= \lim_{n \rightarrow \infty}
            \frac{n!}{(n-x)!} \cdot
            \frac{1}{x!} \cdot
            \frac{\lambda^x}{n^x} \cdot
            e^{-\lambda}
            && \text{Rearranging terms}\\
        &= \lim_{n \rightarrow \infty}
            \frac{n^x}{1} \cdot
            \frac{1}{x!} \cdot
            \frac{\lambda^x}{n^x} \cdot
            e^{-\lambda}
            && \text{As } n \to \infty \text{, } \frac{n!}{(n-x)!} \approx n^x\\
        &= \lim_{n \rightarrow \infty}
            \frac{\lambda^x}{x!} \cdot
            e^{-\lambda}
            && \text{Canceling } n^x\\
        &= 
            \frac{\lambda^x \cdot e^{-\lambda}}{x!} 
            && \text{Simplifying}\\
        \end{align}

        This gives us our elegant Poisson PMF formula: $P(X=x) = \frac{\lambda^x \cdot e^{-\lambda}}{x!}$
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Poisson Distribution in Python

        Python's `scipy.stats` module provides functions to work with the Poisson distribution. Let's see how to calculate probabilities and generate random samples.

        First, let's calculate some probabilities for our ride-sharing example with $\lambda = 5$:
        """
    )
    return


@app.cell
def _(stats):
    # Set lambda parameter
    _lambda = 5

    # Calculate probabilities for X = 1, 2, 3
    p_1 = stats.poisson.pmf(1, _lambda)
    p_2 = stats.poisson.pmf(2, _lambda)
    p_3 = stats.poisson.pmf(3, _lambda)

    print(f"P(X=1) = {p_1:.5f}")
    print(f"P(X=2) = {p_2:.5f}")
    print(f"P(X=3) = {p_3:.5f}")

    # Calculate cumulative probability P(X ≤ 3)
    p_leq_3 = stats.poisson.cdf(3, _lambda)
    print(f"P(X≤3) = {p_leq_3:.5f}")

    # Calculate probability P(X > 10)
    p_gt_10 = 1 - stats.poisson.cdf(10, _lambda)
    print(f"P(X>10) = {p_gt_10:.5f}")
    return p_1, p_2, p_3, p_gt_10, p_leq_3


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""We can also generate random samples from a Poisson distribution and visualize their distribution:""")
    return


@app.cell(hide_code=True)
def _(np, plt, stats):
    # 1000 random samples from Poisson(lambda=5)
    _lambda = 5
    _samples = stats.poisson.rvs(_lambda, size=1000)

    # theoretical PMF
    _x_values = np.arange(0, max(_samples) + 1)
    _pmf_values = stats.poisson.pmf(_x_values, _lambda)

    # histograms to compare
    _fig, _ax = plt.subplots(figsize=(10, 6))

    # samples as a histogram
    _ax.hist(_samples, bins=np.arange(-0.5, max(_samples) + 1.5, 1), 
            alpha=0.7, density=True, label='Random Samples')

    # theoretical PMF
    _ax.plot(_x_values, _pmf_values, 'ro-', label='Theoretical PMF')

    # labels and title
    _ax.set_xlabel('Number of Events')
    _ax.set_ylabel('Relative Frequency / Probability')
    _ax.set_title(f'1000 Random Samples from Poisson(λ={_lambda})')
    _ax.legend()
    _ax.grid(alpha=0.3)

    # annotations
    _ax.annotate(f'Sample Mean: {np.mean(_samples):.2f}', 
                xy=(0.7, 0.9), xycoords='axes fraction',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.3))
    _ax.annotate(f'Theoretical Mean: {_lambda:.2f}', 
                xy=(0.7, 0.8), xycoords='axes fraction',
                bbox=dict(boxstyle='round,pad=0.5', fc='lightgreen', alpha=0.3))

    plt.tight_layout()
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Changing Time Frames

        One important property of the Poisson distribution is that the rate parameter $\lambda$ scales linearly with the time interval. If events occur at a rate of $\lambda$ per unit time, then over a period of $t$ units, the rate parameter becomes $\lambda \cdot t$.

        For example, if a website receives an average of 5 requests per minute, what is the distribution of requests over a 20-minute period?

        The rate parameter for the 20-minute period would be $\lambda = 5 \cdot 20 = 100$ requests.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    # sliders for the rate and time period
    rate_slider = mo.ui.slider(
        start = 0.1,
        stop = 10,
        step=0.1,
        value=5,
        label="Rate per unit time (λ)"
    )

    time_slider = mo.ui.slider(
        start = 1,
        stop = 60,
        step=1,
        value=20,
        label="Time period (t units)"
    )

    controls = mo.vstack([
        mo.md("### Adjust Parameters to See How Time Scaling Works"),
        mo.hstack([rate_slider, time_slider], justify="space-between")
    ])
    return controls, rate_slider, time_slider


@app.cell(hide_code=True)
def _(mo, np, plt, rate_slider, stats, time_slider):
    # parameters from sliders
    _rate = rate_slider.value
    _time = time_slider.value

    # scaled rate parameter
    _lambda = _rate * _time

    # PMF for values
    _max_x = max(30, int(_lambda * 1.5))
    _x = np.arange(0, _max_x + 1)
    _pmf = stats.poisson.pmf(_x, _lambda)

    # plot
    _fig, _ax = plt.subplots(figsize=(10, 6))

    # PMF as bars
    _ax.bar(_x, _pmf, color='royalblue', alpha=0.7, 
           label=f'PMF: Poisson(λ={_lambda:.1f})')

    # vertical line for mean
    _ax.axvline(x=_lambda, color='red', linestyle='--', linewidth=2,
               label=f'Mean = {_lambda:.1f}')

    # labels and title
    _ax.set_xlabel('Number of Events')
    _ax.set_ylabel('Probability')
    _ax.set_title(f'Poisson Distribution Over {_time} Units (Rate = {_rate}/unit)')

    # better visualization if lambda is large
    if _lambda > 10:
        _ax.set_xlim(_lambda - 4*np.sqrt(_lambda), _lambda + 4*np.sqrt(_lambda))

    _ax.legend()
    _ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.gca()

    # additional information
    info = mo.md(
        f"""
        When the rate is **{_rate}** events per unit time and we observe for **{_time}** units:

        - The expected number of events is **{_lambda:.1f}**
        - The variance is also **{_lambda:.1f}**
        - The standard deviation is **{np.sqrt(_lambda):.2f}**
        - P(X=0) = {stats.poisson.pmf(0, _lambda):.4f} (probability of no events)
        - P(X≥10) = {1 - stats.poisson.cdf(9, _lambda):.4f} (probability of 10 or more events)
        """
    )
    return (info,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 🤔 Test Your Understanding
        Pick which of these statements about Poisson distributions you think are correct:

        /// details | The variance of a Poisson distribution is always equal to its mean
        ✅ Correct! For a Poisson distribution with parameter $\lambda$, both the mean and variance equal $\lambda$.
        ///

        /// details | The Poisson distribution can be used to model the number of successes in a fixed number of trials
        ❌ Incorrect! That's the binomial distribution. The Poisson distribution models the number of events in a fixed interval of time or space, not a fixed number of trials.
        ///

        /// details | If $X \sim \text{Poisson}(\lambda_1)$ and $Y \sim \text{Poisson}(\lambda_2)$ are independent, then $X + Y \sim \text{Poisson}(\lambda_1 + \lambda_2)$
        ✅ Correct! The sum of independent Poisson random variables is also a Poisson random variable with parameter equal to the sum of the individual parameters.
        ///

        /// details | As $\lambda$ increases, the Poisson distribution approaches a normal distribution
        ✅ Correct! For large values of $\lambda$ (generally $\lambda > 10$), the Poisson distribution is approximately normal with mean $\lambda$ and variance $\lambda$.
        ///

        /// details | The probability of zero events in a Poisson process is always less than the probability of one event
        ❌ Incorrect! For $\lambda < 1$, the probability of zero events ($e^{-\lambda}$) is actually greater than the probability of one event ($\lambda e^{-\lambda}$).
        ///

        /// details | The Poisson distribution has a single parameter $\lambda$, which always equals the average number of events per time period
        ✅ Correct! The parameter $\lambda$ represents the average rate of events, and it uniquely defines the distribution.
        ///
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Summary

        The Poisson distribution is one of those incredibly useful tools that shows up all over the place. I've always found it fascinating how such a simple formula can model so many real-world phenomena - from website traffic to radioactive decay.

        What makes the Poisson really cool is that it emerges naturally as we try to model rare events occurring over a continuous interval. Remember that visualization where we kept dividing time into smaller and smaller chunks? As we showed, when you take a binomial distribution and let the number of trials approach infinity while keeping the expected value constant, you end up with the elegant Poisson formula.

        The key things to remember about the Poisson distribution:

        - It models the number of events occurring in a fixed interval of time or space, assuming events happen at a constant average rate and independently of each other

        - Its PMF is given by the elegantly simple formula $P(X=k) = \frac{\lambda^k e^{-\lambda}}{k!}$

        - Both the mean and variance equal the parameter $\lambda$, which represents the average number of events per interval

        - It's related to the binomial distribution as a limiting case when $n \to \infty$, $p \to 0$, and $np = \lambda$ remains constant

        - The rate parameter scales linearly with the length of the interval - if events occur at rate $\lambda$ per unit time, then over $t$ units, the parameter becomes $\lambda t$

        From modeling website traffic and customer arrivals to defects in manufacturing and radioactive decay, the Poisson distribution provides a powerful and mathematically elegant way to understand random occurrences in our world.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Appendix code (helper functions, variables, etc.):""")
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy.stats as stats
    import pandas as pd
    import altair as alt
    from wigglystuff import TangleSlider
    return TangleSlider, alt, np, pd, plt, stats


@app.cell(hide_code=True)
def _():
    import io
    import base64
    from matplotlib.figure import Figure

    # Helper function to convert mpl figure to an image format mo.image can hopefully handle
    def fig_to_image(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        data = f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"
        return data
    return Figure, base64, fig_to_image, io


if __name__ == "__main__":
    app.run()
