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

__generated_with = "0.11.25"
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
    # interactive elements using TangleSlider
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
    def create_poisson_pmf_plot(lambda_value):
        """Create a visualization of Poisson PMF with annotations for mean and variance."""
        # PMF for values
        max_x = max(20, int(lambda_value * 3))  # Show at least up to 3*lambda
        x = np.arange(0, max_x + 1)
        pmf = stats.poisson.pmf(x, lambda_value)

        # Relevant key statistics
        mean = lambda_value  # For Poisson, mean = lambda
        variance = lambda_value  # For Poisson, variance = lambda
        std_dev = np.sqrt(variance)

        # plot
        fig, ax = plt.subplots(figsize=(10, 6))

        # PMF as bars
        ax.bar(x, pmf, color='royalblue', alpha=0.7, label=f'PMF: P(X=k)')

        # for the PMF values
        ax.plot(x, pmf, 'ro-', alpha=0.6, label='PMF line')

        # Vertical lines - mean and key values
        ax.axvline(x=mean, color='green', linestyle='--', linewidth=2, 
                label=f'Mean: {mean:.2f}')

        # Stdev region
        ax.axvspan(mean - std_dev, mean + std_dev, alpha=0.2, color='green',
                label=f'±1 Std Dev: {std_dev:.2f}')

        ax.set_xlabel('Number of Events (k)')
        ax.set_ylabel('Probability: P(X=k)')
        ax.set_title(f'Poisson Distribution with λ={lambda_value:.1f}')

        # annotations
        ax.annotate(f'E[X] = {mean:.2f}', 
                    xy=(mean, stats.poisson.pmf(int(mean), lambda_value)), 
                    xytext=(mean + 1, max(pmf) * 0.8),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1))

        ax.annotate(f'Var(X) = {variance:.2f}', 
                    xy=(mean, stats.poisson.pmf(int(mean), lambda_value) / 2), 
                    xytext=(mean + 1, max(pmf) * 0.6),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1))

        ax.grid(alpha=0.3)
        ax.legend()

        plt.tight_layout()
        return plt.gca()

    # Get parameter from slider and create plot
    _lambda = lambda_slider.amount
    create_poisson_pmf_plot(_lambda)
    return (create_poisson_pmf_plot,)


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
    def create_time_division_visualization():
        # visualization of dividing a minute into 60 seconds
        fig, ax = plt.subplots(figsize=(12, 2))

        # Example events hardcoded at 2.75s and 7.12s
        events = [2.75, 7.12]

        # array of 60 rectangles
        for i in range(60):
            color = 'royalblue' if any(i <= e < i+1 for e in events) else 'lightgray'
            ax.add_patch(plt.Rectangle((i, 0), 0.9, 1, color=color))

        # markers for events
        for e in events:
            ax.plot(e, 0.5, 'ro', markersize=10)

        # labels
        ax.set_xlim(0, 60)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_xticks([0, 15, 30, 45, 60])
        ax.set_xticklabels(['0s', '15s', '30s', '45s', '60s'])
        ax.set_xlabel('Time (seconds)')
        ax.set_title('One Minute Divided into 60 Second Intervals')

        plt.tight_layout()
        plt.gca()
        return fig, events, i

    # Create visualization and convert to image
    _fig, _events, i = create_time_division_visualization()
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
    mo.vstack([_fig, _explanation])
    return create_time_division_visualization, i


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
def _(fig_to_image, mo, plt):
    def create_decisecond_visualization(e_value):
        # (Just showing the first 100 for clarity)
        fig, ax = plt.subplots(figsize=(12, 2))

        # Example events at 2.75s and 7.12s (convert to deciseconds)
        events = [27.5, 71.2]
    
        for i in range(100):
            color = 'royalblue' if any(i <= event_val < i + 1 for event_val in events) else 'lightgray'
            ax.add_patch(plt.Rectangle((i, 0), 0.9, 1, color=color))

        # Markers for events
        for event in events:
            if event < 100:  # Only show events in our visible range
                ax.plot(event/10, 0.5, 'ro', markersize=10)  # Divide by 10 to convert to deciseconds

        # Add labels
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_xticks([0, 20, 40, 60, 80, 100])
        ax.set_xticklabels(['0s', '2s', '4s', '6s', '8s', '10s'])
        ax.set_xlabel('Time (first 10 seconds shown)')
        ax.set_title('One Minute Divided into 600 Decisecond Intervals (first 100 shown)')

        plt.tight_layout()
        plt.gca()
        return fig

    # Create viz and convert to image
    _fig = create_decisecond_visualization(e_value=5)
    _img = mo.image(fig_to_image(_fig), width="100%")

    # Explanation
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
    mo.vstack([_fig, _explanation])
    return (create_decisecond_visualization,)


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
    def create_comparison_plot(n, lambda_value):
        # Calculate probability
        p = lambda_value / n

        # Binomial probabilities
        x_values = np.arange(0, 15)
        binom_pmf = stats.binom.pmf(x_values, n, p)

        # True Poisson probabilities
        poisson_pmf = stats.poisson.pmf(x_values, lambda_value)

        # DF for comparison
        df = pd.DataFrame({
            'Events': x_values,
            f'Binomial(n={n}, p={p:.6f})': binom_pmf,
            f'Poisson(λ=5)': poisson_pmf,
            'Difference': np.abs(binom_pmf - poisson_pmf)
        })

        # Plot both PMFs
        fig, ax = plt.subplots(figsize=(10, 6))

        # Bar plot for the binomial
        ax.bar(x_values - 0.2, binom_pmf, width=0.4, alpha=0.7, 
            color='royalblue', label=f'Binomial(n={n}, p={p:.6f})')

        # Bar plot for the Poisson
        ax.bar(x_values + 0.2, poisson_pmf, width=0.4, alpha=0.7,
            color='crimson', label='Poisson(λ=5)')

        # Labels and title
        ax.set_xlabel('Number of Events (k)')
        ax.set_ylabel('Probability')
        ax.set_title(f'Comparison of Binomial and Poisson PMFs with n={n}')
        ax.legend()
        ax.set_xticks(x_values)
        ax.grid(alpha=0.3)

        plt.tight_layout()
        return df, fig, n, p

    # Number of intervals from the slider
    n = intervals_slider.value
    _lambda = 5  # Fixed lambda for our example

    # Cromparison plot
    df, fig, n, p = create_comparison_plot(n, _lambda)
    return create_comparison_plot, df, fig, n, p


@app.cell(hide_code=True)
def _(df, fig, fig_to_image, mo, n, p):
    # table of values
    _styled_df = df.style.format({
        f'Binomial(n={n}, p={p:.6f})': '{:.6f}',
        f'Poisson(λ=5)': '{:.6f}',
        'Difference': '{:.6f}'
    })

    # Calculate the max absolute difference
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
    def create_samples_plot(lambda_value, sample_size=1000):
        # Random samples
        samples = stats.poisson.rvs(lambda_value, size=sample_size)

        # theoretical PMF
        x_values = np.arange(0, max(samples) + 1)
        pmf_values = stats.poisson.pmf(x_values, lambda_value)

        # histograms to compare
        fig, ax = plt.subplots(figsize=(10, 6))

        # samples as a histogram
        ax.hist(samples, bins=np.arange(-0.5, max(samples) + 1.5, 1), 
                alpha=0.7, density=True, label='Random Samples')

        # theoretical PMF
        ax.plot(x_values, pmf_values, 'ro-', label='Theoretical PMF')

        # labels and title
        ax.set_xlabel('Number of Events')
        ax.set_ylabel('Relative Frequency / Probability')
        ax.set_title(f'1000 Random Samples from Poisson(λ={lambda_value})')
        ax.legend()
        ax.grid(alpha=0.3)

        # annotations
        ax.annotate(f'Sample Mean: {np.mean(samples):.2f}', 
                    xy=(0.7, 0.9), xycoords='axes fraction',
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.3))
        ax.annotate(f'Theoretical Mean: {lambda_value:.2f}', 
                    xy=(0.7, 0.8), xycoords='axes fraction',
                    bbox=dict(boxstyle='round,pad=0.5', fc='lightgreen', alpha=0.3))

        plt.tight_layout()
        return plt.gca()

    # Use a lambda value of 5 for this example
    _lambda = 5
    create_samples_plot(_lambda)
    return (create_samples_plot,)


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


@app.cell
def _(controls):
    controls.center()
    return


@app.cell(hide_code=True)
def _(mo, np, plt, rate_slider, stats, time_slider):
    def create_time_scaling_plot(rate, time_period):
        # scaled rate parameter
        lambda_value = rate * time_period

        # PMF for values
        max_x = max(30, int(lambda_value * 1.5))
        x = np.arange(0, max_x + 1)
        pmf = stats.poisson.pmf(x, lambda_value)

        # plot
        fig, ax = plt.subplots(figsize=(10, 6))

        # PMF as bars
        ax.bar(x, pmf, color='royalblue', alpha=0.7, 
            label=f'PMF: Poisson(λ={lambda_value:.1f})')

        # vertical line for mean
        ax.axvline(x=lambda_value, color='red', linestyle='--', linewidth=2,
                label=f'Mean = {lambda_value:.1f}')

        # labels and title
        ax.set_xlabel('Number of Events')
        ax.set_ylabel('Probability')
        ax.set_title(f'Poisson Distribution Over {time_period} Units (Rate = {rate}/unit)')

        # better visualization if lambda is large
        if lambda_value > 10:
            ax.set_xlim(lambda_value - 4*np.sqrt(lambda_value), 
                         lambda_value + 4*np.sqrt(lambda_value))

        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()

        # Create relevant info markdown
        info_text = f"""
        When the rate is **{rate}** events per unit time and we observe for **{time_period}** units:

        - The expected number of events is **{lambda_value:.1f}**
        - The variance is also **{lambda_value:.1f}**
        - The standard deviation is **{np.sqrt(lambda_value):.2f}**
        - P(X=0) = {stats.poisson.pmf(0, lambda_value):.4f} (probability of no events)
        - P(X≥10) = {1 - stats.poisson.cdf(9, lambda_value):.4f} (probability of 10 or more events)
        """

        return plt.gca(), info_text

    # parameters from sliders
    _rate = rate_slider.value
    _time = time_slider.value

    # store
    _plot, _info_text = create_time_scaling_plot(_rate, _time)

    # Display info as markdown
    info = mo.md(_info_text)

    mo.vstack([_plot, info], justify="center")
    return create_time_scaling_plot, info


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
