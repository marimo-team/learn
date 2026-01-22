# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "matplotlib==3.10.1",
#     "scipy==1.15.2",
#     "numpy==2.2.4",
#     "polars==1.26.0",
#     "plotly==5.18.0",
#     "scikit-learn==1.6.1",
# ]
# ///

import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium", app_title="Naive Bayes Classification")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Naive Bayes Classification

    _This notebook is a computational companion to ["Probability for Computer Scientists"](https://chrispiech.github.io/probabilityForComputerScientists/en/part5/naive_bayes/), by Stanford professor Chris Piech._

    Naive Bayes is one of those classic machine learning algorithms that seems almost too simple to work, yet it's surprisingly effective for many classification tasks. I've always found it fascinating how this algorithm applies Bayes' theorem with a strong (but knowingly incorrect) "naive" assumption that all features are independent of each other.

    In this notebook, we'll dive into why this supposedly "wrong" assumption still leads to good results. We'll walk through the training process, learn how to make predictions, and see some interactive visualizations that helped me understand the concept better when I was first learning it. We'll also explore why Naive Bayes excels particularly in text classification problems like spam filtering.

    If you're new to Naive Bayes, I highly recommend checking out [this excellent explanation by Mahesh Huddar](https://youtu.be/XzSlEA4ck2I?si=AASeh_KP68BAbzy5), which provides a step-by-step walkthrough with a helpful example (which we take a dive into, down below).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Why "Naive"?

    So why is it called "naive"? It's because the algorithm makes an assumption — it assumes all features are completely independent of each other when given the class label.

    The math way of saying this is:

    $$P(X_1, X_2, \ldots, X_n | Y) = P(X_1 | Y) \times P(X_2 | Y) \times \ldots \times P(X_n | Y) = \prod_{i=1}^{n} P(X_i | Y)$$

    This independence assumption is almost always wrong in real data. Think about text classification — if you see the word "cloudy" in a weather report, you're much more likely to also see "rain" than you would be to see "sunshine". These words clearly depend on each other! Or in medical diagnosis, symptoms often occur together as part of syndromes.

    But here's the cool part — even though we know this assumption is _technically_ wrong, the algorithm still works remarkably well in practice. By making this simplifying assumption, we:

    - Make the math way easier to compute
    - Need way less training data to get decent results
    - Can handle thousands of features without blowing up computationally
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The Math Behind Naive Bayes

    At its core, Naive Bayes is just an application of Bayes' theorem from our earlier probability notebooks. Let's break it down:

    We have some features $\mathbf{X} = [X_1, X_2, \ldots, X_m]$ (like words in an email or symptoms of a disease) and we want to predict a class label $Y$ (like "spam/not spam" or "has disease/doesn't have disease").

    What we're really trying to find is:

    $$P(Y|\mathbf{X})$$

    In other words, "what's the probability of a certain class given the features we observed?" Once we have these probabilities, we simply pick the class with the highest probability:

    $$\hat{y} = \underset{y}{\operatorname{argmax}} \text{ } P(Y=y|\mathbf{X}=\mathbf{x})$$

    Applying Bayes' theorem (from our earlier probability work), we get:

    $$P(Y=y|\mathbf{X}=\mathbf{x}) = \frac{P(Y=y) \times P(\mathbf{X}=\mathbf{x}|Y=y)}{P(\mathbf{X}=\mathbf{x})}$$

    Since we're comparing different possible classes for the same input features, the denominator $P(\mathbf{X}=\mathbf{x})$ is the same for all classes. So we can drop it and just compare:

    $$\hat{y} = \underset{y}{\operatorname{argmax}} \text{ } P(Y=y) \times P(\mathbf{X}=\mathbf{x}|Y=y)$$

    Here's where the "naive" part comes in. Calculating $P(\mathbf{X}=\mathbf{x}|Y=y)$ directly would be a computational nightmare - we'd need counts for every possible combination of feature values. Instead, we make that simplifying "naive" assumption that features are independent of each other:

    $$P(\mathbf{X}=\mathbf{x}|Y=y) = \prod_{i=1}^{m} P(X_i=x_i|Y=y)$$

    Which gives us our final formula:

    $$\hat{y} = \underset{y}{\operatorname{argmax}} \text{ } P(Y=y) \times \prod_{i=1}^{m} P(X_i=x_i|Y=y)$$

    In actual implementations, we usually use logarithms to avoid the numerical problems that come with multiplying many small probabilities (they can _underflow_ to zero):

    $$\hat{y} = \underset{y}{\operatorname{argmax}} \text{ } \log P(Y=y) + \sum_{i=1}^{m} \log P(X_i=x_i|Y=y)$$

    That's it! The really cool thing is that despite this massive simplification, the algorithm often gives surprisingly good results.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Example Problem
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's apply Naive Bayes principles to this data (Tennis Training Dataset):
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## A Simple Example: Play Tennis

        Let's understand Naive Bayes with a classic example: predicting whether someone will play tennis based on weather conditions. This is the same example used in Mahesh Huddar's excellent video.

        Our dataset has these features:
        - **Outlook**: Sunny, Overcast, Rainy
        - **Temperature**: Hot, Mild, Cool
        - **Humidity**: High, Normal
        - **Wind**: Strong, Weak

        And the target variable:
        - **Play Tennis**: Yes, No

        ### Example Dataset
        """
    )

    # Create a dataset matching the image (in dict format for proper table rendering)
    example_data = [
        {"Day": "D1", "Outlook": "Sunny", "Temperature": "Hot", "Humidity": "High", "Wind": "Weak", "Play Tennis": "No"},
        {"Day": "D2", "Outlook": "Sunny", "Temperature": "Hot", "Humidity": "High", "Wind": "Strong", "Play Tennis": "No"},
        {"Day": "D3", "Outlook": "Overcast", "Temperature": "Hot", "Humidity": "High", "Wind": "Weak", "Play Tennis": "Yes"},
        {"Day": "D4", "Outlook": "Rain", "Temperature": "Mild", "Humidity": "High", "Wind": "Weak", "Play Tennis": "Yes"},
        {"Day": "D5", "Outlook": "Rain", "Temperature": "Cool", "Humidity": "Normal", "Wind": "Weak", "Play Tennis": "Yes"},
        {"Day": "D6", "Outlook": "Rain", "Temperature": "Cool", "Humidity": "Normal", "Wind": "Strong", "Play Tennis": "No"},
        {"Day": "D7", "Outlook": "Overcast", "Temperature": "Cool", "Humidity": "Normal", "Wind": "Strong", "Play Tennis": "Yes"},
        {"Day": "D8", "Outlook": "Sunny", "Temperature": "Mild", "Humidity": "High", "Wind": "Weak", "Play Tennis": "No"},
        {"Day": "D9", "Outlook": "Sunny", "Temperature": "Cool", "Humidity": "Normal", "Wind": "Weak", "Play Tennis": "Yes"},
        {"Day": "D10", "Outlook": "Rain", "Temperature": "Mild", "Humidity": "Normal", "Wind": "Weak", "Play Tennis": "Yes"},
        {"Day": "D11", "Outlook": "Sunny", "Temperature": "Mild", "Humidity": "Normal", "Wind": "Strong", "Play Tennis": "Yes"},
        {"Day": "D12", "Outlook": "Overcast", "Temperature": "Mild", "Humidity": "High", "Wind": "Strong", "Play Tennis": "Yes"},
        {"Day": "D13", "Outlook": "Overcast", "Temperature": "Hot", "Humidity": "Normal", "Wind": "Weak", "Play Tennis": "Yes"},
        {"Day": "D14", "Outlook": "Rain", "Temperature": "Mild", "Humidity": "High", "Wind": "Strong", "Play Tennis": "No"}
    ]

    # Display the tennis dataset using a table
    example_table = mo.ui.table(
        data=example_data,
        selection=None
    )

    mo.vstack([
        mo.md("#### Tennis Training Dataset"),
        example_table
    ])
    return (example_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's predict whether someone will play tennis given these weather conditions:

    - Outlook: Sunny
    - Temperature: Cool
    - Humidity: High
    - Wind: Strong

    Let's walk through the calculations step by step:

    #### Step 1: Calculate Prior Probabilities

    First, we calculate $P(Y=\text{Yes})$ and $P(Y=\text{No})$:

    - $P(Y=\text{Yes}) = \frac{9}{14} = 0.64$
    - $P(Y=\text{No}) = \frac{5}{14} = 0.36$

    #### Step 2: Calculate Conditional Probabilities

    Next, we calculate the conditional probabilities for each feature value given each class:
    """)
    return


@app.cell(hide_code=True)
def _(humidity_data, mo, outlook_data, summary_table, temp_data, wind_data):
    # Display tables with appropriate styling
    mo.vstack([
        mo.md("#### Class Distribution"),
        summary_table,
        mo.md("#### Conditional Probabilities"),
        mo.hstack([
            mo.vstack([
                mo.md("**Outlook**"),
                mo.ui.table(
                    data=outlook_data,
                    selection=None
                )
            ]),
            mo.vstack([
                mo.md("**Temperature**"),
                mo.ui.table(
                    data=temp_data,
                    selection=None
                )
            ])
        ]),
        mo.hstack([
            mo.vstack([
                mo.md("**Humidity**"),
                mo.ui.table(
                    data=humidity_data,
                    selection=None
                )
            ]),
            mo.vstack([
                mo.md("**Wind**"),
                mo.ui.table(
                    data=wind_data,
                    selection=None
                )
            ])
        ])
    ])
    return


@app.cell
def _():
    # DIY
    return


@app.cell(hide_code=True)
def _(mo, solution_accordion):
    # Display the accordion
    mo.accordion(solution_accordion)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try a Different Example

    What if the conditions were different? Let's say:

    - Outlook: Overcast
    - Temperature: Hot
    - Humidity: Normal
    - Wind: Weak

    Try working through this example on your own. If you get stuck, you can use the tables above and apply the same method we used in the solution.
    """)
    return


@app.cell
def _():
    # DIY
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Interactive Naive Bayes

    Let's explore Naive Bayes with an interactive visualization. This will help build intuition about how the algorithm makes predictions and how the naive independence assumption affects results.
    """)
    return


@app.cell(hide_code=True)
def gaussian_viz(
    Ellipse,
    GaussianNB,
    ListedColormap,
    class_sep_slider,
    controls,
    make_classification,
    mo,
    n_samples_slider,
    noise_slider,
    np,
    pl,
    plt,
    regenerate_button,
    train_test_split,
):
    # get values from sliders
    class_sep = class_sep_slider.value
    noise_val = noise_slider.value
    n_samples = int(n_samples_slider.value)

    # check if regenerate button was clicked
    regenerate_state = regenerate_button.value

    # make a dataset with current settings
    X, y = make_classification(
        n_samples=n_samples,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        n_clusters_per_class=1,
        class_sep=class_sep * (1 - noise_val),  # use noise to reduce separation
        random_state=42 if not regenerate_state else np.random.randint(1000)
    )

    # put data in a dataframe
    viz_df = pl.DataFrame({
        "Feature1": X[:, 0],
        "Feature2": X[:, 1],
        "Class": y
    })

    # split into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # create naive bayes classifier
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)

    # setup grid for boundary visualization
    h = 0.1  # step size
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # predict on grid points
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = gnb.predict(grid_points).reshape(xx.shape)

    # calculate class stats
    class0_mean = np.mean(X_train[y_train == 0], axis=0)
    class1_mean = np.mean(X_train[y_train == 1], axis=0)
    class0_var = np.var(X_train[y_train == 0], axis=0)
    class1_var = np.var(X_train[y_train == 1], axis=0)

    # format for display
    class_stats = [
        {"Class": "Class 0", "Feature1_Mean": f"{class0_mean[0]:.4f}", "Feature1_Variance": f"{class0_var[0]:.4f}",
         "Feature2_Mean": f"{class0_mean[1]:.4f}", "Feature2_Variance": f"{class0_var[1]:.4f}"},
        {"Class": "Class 1", "Feature1_Mean": f"{class1_mean[0]:.4f}", "Feature1_Variance": f"{class1_var[0]:.4f}",
         "Feature2_Mean": f"{class1_mean[1]:.4f}", "Feature2_Variance": f"{class1_var[1]:.4f}"}
    ]

    # setup plot with two panels
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # colors for our plots
    cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])  # bg colors
    cmap_bold = ListedColormap(['#FF0000', '#0000FF'])   # point colors

    # left: decision boundary
    ax1.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_light)
    scatter1 = ax1.scatter(X_train[:, 0], X_train[:, 1], c=y_train, 
                          cmap=cmap_bold, edgecolor='k', s=50, alpha=0.8)
    scatter2 = ax1.scatter(X_test[:, 0], X_test[:, 1], c=y_test, 
                          cmap=cmap_bold, edgecolor='k', s=25, alpha=0.5)

    ax1.set_xlabel('Feature 1')
    ax1.set_ylabel('Feature 2')
    ax1.set_title('Gaussian Naive Bayes Decision Boundary')
    ax1.legend([scatter1.legend_elements()[0][0], scatter2.legend_elements()[0][0]],
              ['Training Data', 'Test Data'], loc='upper right')

    # right: distribution visualization
    class0_data = viz_df.filter(pl.col("Class") == 0)
    class1_data = viz_df.filter(pl.col("Class") == 1)

    ax2.scatter(class0_data["Feature1"], class0_data["Feature2"], 
               color='red', edgecolor='k', s=50, alpha=0.8, label='Class 0')
    ax2.scatter(class1_data["Feature1"], class1_data["Feature2"], 
               color='blue', edgecolor='k', s=50, alpha=0.8, label='Class 1')

    # draw ellipses function
    def plot_ellipse(ax, mean, cov, color):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals = vals[order]
        vecs = vecs[:, order]

        theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        width, height = 2 * np.sqrt(5.991 * vals)

        ellip = Ellipse(xy=mean, width=width, height=height, angle=theta, 
                       edgecolor=color, fc='None', lw=2, alpha=0.7)
        ax.add_patch(ellip)

    # add ellipses for each class accordingly
    class0_cov = np.diag(np.var(X_train[y_train == 0], axis=0))
    class1_cov = np.diag(np.var(X_train[y_train == 1], axis=0))

    plot_ellipse(ax2, class0_mean, class0_cov, 'red')
    plot_ellipse(ax2, class1_mean, class1_cov, 'blue')

    ax2.set_xlabel('Feature 1')
    ax2.set_ylabel('Feature 2')
    ax2.set_title('Class-Conditional Distributions (Gaussian)')
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.gca()

    # show interactive plot
    mpl_fig = mo.mpl.interactive(fig)

    # show parameters info
    mo.md(
        r"""
        ### gaussian parameters by class

        each feature follows a normal distribution per class. here are the parameters:
        """
    )

    # make stats table
    stats_table = mo.ui.table(
        data=class_stats,
        selection="single"
    )

    mo.md(
        r"""
        ### how it works

        1. calculate mean & variance for each feature per class
        2. use gaussian pdf to get probabilities for new points
        3. apply bayes' theorem to pick most likely class

        ellipses show the distributions, decision boundary is where probabilities equal
        """
    )

    # stack everything together
    mo.vstack([
        controls.center(),
        mpl_fig,
        stats_table
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### what's going on in this demo?

    Playing with the sliders changes how our data looks and how the classifier behaves. Class separation controls how far apart the two classes are — higher values make them easier to tell apart. The noise slider adds randomness by reducing that separation, making boundaries fuzzier and classification harder. More samples just gives you more data points to work with.

    The left graph shows the decision boundary — that curved line where the classifier switches from predicting one class to another. Red and blue regions show where naive bayes would classify new points. The right graph shows the actual distribution of both classes, with those ellipses representing the gaussian distributions naive bayes is using internally.

    Try cranking up the noise and watch how the boundary gets messier. increase separation and see how confident the classifier becomes. This is basically what's happening inside naive bayes — it's looking at each feature's distribution per class and making the best guess based on probabilities. The table below shows the actual parameters (means and variances) the model calculates.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Types of Naive Bayes Classifiers

    ### Multinomial Naive Bayes
    Ideal for text classification where features represent word counts or frequencies.

    Mathematical form:

    \[P(x_i|y) = \frac{\text{count}(x_i, y) + \alpha}{\sum_{i=1}^{|V|} \text{count}(x_i, y) + \alpha|V|}\]

    where:

    - \(\alpha\) is the smoothing parameter
    - \(|V|\) is the size of the vocabulary
    - \(\text{count}(x_i, y)\) is the count of feature \(i\) in class \(y\)

    ### Bernoulli Naive Bayes
    Best for binary features (0/1) — either a word appears or it doesn't.

    Mathematical form:

    \[P(x_i|y) = p_{iy}^{x_i}(1-p_{iy})^{(1-x_i)}\]

    where:

    - \(p_{iy}\) is the probability of feature \(i\) occurring in class \(y\)
    - \(x_i\) is 1 if the feature is present, 0 otherwise

    ### Gaussian Naive Bayes
    Designed for continuous features, assuming they follow a normal distribution.

    Mathematical form:

    \[P(x_i|y) = \frac{1}{\sqrt{2\pi\sigma_y^2}} \exp\left(-\frac{(x_i - \mu_y)^2}{2\sigma_y^2}\right)\]

    where:

    - \(\mu_y\) is the mean of feature values for class \(y\)
    - \(\sigma_y^2\) is the variance of feature values for class \(y\)

    ### Complement Naive Bayes
    Particularly effective for imbalanced datasets.

    Mathematical form:

    \[P(x_i|y) = \frac{\text{count}(x_i, \bar{y}) + \alpha}{\sum_{i=1}^{|V|} \text{count}(x_i, \bar{y}) + \alpha|V|}\]

    where:

    - \(\bar{y}\) represents all classes except \(y\)
    - Other parameters are similar to Multinomial Naive Bayes
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 🤔 Test Your Understanding

    Test your understanding of Naive Bayes with these statements:

    /// details | Multiplying small probabilities in Naive Bayes can lead to numerical underflow.
    ✅ **Correct!** Multiplying many small probabilities can indeed lead to numerical underflow.

    That's why in practice, we often use log probabilities and add them instead of multiplying the original probabilities. This prevents numerical underflow and improves computational stability.
    ///

    /// details | Laplace smoothing is unnecessary if your training data covers all possible feature values.
    ❌ **Incorrect.** Laplace smoothing is still beneficial even with complete feature coverage.

    While Laplace smoothing is crucial for handling unseen feature values, it also helps with small sample sizes by preventing overfitting to the training data. Even with complete feature coverage, some combinations might have very few examples, leading to unreliable probability estimates.
    ///

    /// details | Naive Bayes performs poorly on high-dimensional data compared to other classifiers.
    ❌ **Incorrect.** Naive Bayes actually excels with high-dimensional data.

    Due to its simplicity and the independence assumption, Naive Bayes scales very well to high-dimensional data. It's particularly effective for text classification where each word is a dimension and there can be thousands of dimensions. Other classifiers might overfit in such high-dimensional spaces.
    ///

    /// details | For text classification, Multinomial Naive Bayes typically outperforms Gaussian Naive Bayes.
    ✅ **Correct!** Multinomial NB is better suited for text classification than Gaussian NB.

    Text data typically involves discrete counts (word frequencies) which align better with a multinomial distribution. Gaussian Naive Bayes assumes features follow a normal distribution, which doesn't match the distribution of word frequencies in text documents.
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary

    Throughout this notebook, we've explored Naive Bayes classification. What makes this algorithm particularly interesting is its elegant simplicity combined with surprising effectiveness. Despite making what seems like an overly simplistic assumption — that features are independent given the class — it consistently delivers reasonable performance across a wide range of applications.

    The algorithm's power lies in its probabilistic foundation, built upon Bayes' theorem. During training, it simply learns probability distributions: the likelihood of seeing each class (prior probabilities) and the probability of feature values within each class (conditional probabilities). When making predictions, it combines these probabilities using the naive independence assumption, which dramatically simplifies the computation while still maintaining remarkable predictive power.

    We've seen how different variants of Naive Bayes adapt to various types of data. Multinomial Naive Bayes excels at text classification by modeling word frequencies, Bernoulli Naive Bayes handles binary features elegantly, and Gaussian Naive Bayes tackles continuous data through normal distributions. Each variant maintains the core simplicity of the algorithm while adapting its probability calculations to match the data's characteristics.

    Perhaps most importantly, we've learned that sometimes the most straightforward approaches can be the most practical. Naive Bayes demonstrates that a simple model, well-understood and properly applied, can often outperform more complex alternatives, especially in domains like text classification or when working with limited computational resources or training data.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    appendix (helper code)
    """)
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
    import polars as pl
    from scipy import stats
    from sklearn.naive_bayes import GaussianNB
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Ellipse

    # for consistent results
    np.random.seed(42)

    # nicer plots
    plt.style.use('seaborn-v0_8-darkgrid')
    return (
        Ellipse,
        GaussianNB,
        ListedColormap,
        make_classification,
        np,
        pl,
        plt,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(example_data, mo):
    # occurrences count in example data
    yes_count = sum(1 for row in example_data if row["Play Tennis"] == "Yes")
    no_count = sum(1 for row in example_data if row["Play Tennis"] == "No")
    total = len(example_data)

    # summary table with dict format
    summary_data = [
        {"Class": "Yes", "Count": f"{yes_count}", "Probability": f"{yes_count/total:.2f}"},
        {"Class": "No", "Count": f"{no_count}", "Probability": f"{no_count/total:.2f}"},
        {"Class": "Total", "Count": f"{total}", "Probability": "1.00"}
    ]

    summary_table = mo.ui.table(
        data=summary_data,
        selection=None
    )

    # tables for conditional probabilities matching the image (in dict format)
    outlook_data = [
        {"Outlook": "Sunny", "Y": "2/9", "N": "3/5"},
        {"Outlook": "Overcast", "Y": "4/9", "N": "0"},
        {"Outlook": "Rain", "Y": "3/9", "N": "2/5"}
    ]

    temp_data = [
        {"Temperature": "Hot", "Y": "2/9", "N": "2/5"},
        {"Temperature": "Mild", "Y": "4/9", "N": "2/5"},
        {"Temperature": "Cool", "Y": "3/9", "N": "1/5"}
    ]

    humidity_data = [
        {"Humidity": "High", "Y": "3/9", "N": "4/5"},
        {"Humidity": "Normal", "Y": "6/9", "N": "1/5"}
    ]

    wind_data = [
        {"Wind": "Strong", "Y": "3/9", "N": "3/5"},
        {"Wind": "Weak", "Y": "6/9", "N": "2/5"}
    ]
    return humidity_data, outlook_data, summary_table, temp_data, wind_data


@app.cell(hide_code=True)
def _(mo):
    # accordion with solution (step-by-step)
    solution_accordion = {
        "step-by-step solution (click to expand)": mo.md(r"""
        #### step 1: gather probabilities

        from our tables:

        **prior probabilities:**

        - $P(Yes) = 9/14 = 0.64$
        - $P(No) = 5/14 = 0.36$

        **conditional probabilities:**

        - $P(Outlook=Sunny|Yes) = 2/9$
        - $P(Outlook=Sunny|No) = 3/5$
        - $P(Temperature=Cool|Yes) = 3/9$
        - $P(Temperature=Cool|No) = 1/5$
        - $P(Humidity=High|Yes) = 3/9$
        - $P(Humidity=High|No) = 4/5$
        - $P(Wind=Strong|Yes) = 3/9$
        - $P(Wind=Strong|No) = 3/5$

        #### step 2: calculate for yes

        $P(Yes) \times P(Sunny|Yes) \times P(Cool|Yes) \times P(High|Yes) \times P(Strong|Yes)$

        $= \frac{9}{14} \times \frac{2}{9} \times \frac{3}{9} \times \frac{3}{9} \times \frac{3}{9}$

        $= \frac{9}{14} \times \frac{2 \times 3 \times 3 \times 3}{9^4}$

        $= \frac{9}{14} \times \frac{54}{6561}$

        $= \frac{9 \times 54}{14 \times 6561}$

        $= \frac{486}{91854}$

        $= 0.0053$

        #### step 3: calculate for no

        $P(No) \times P(Sunny|No) \times P(Cool|No) \times P(High|No) \times P(Strong|No)$

        $= \frac{5}{14} \times \frac{3}{5} \times \frac{1}{5} \times \frac{4}{5} \times \frac{3}{5}$

        $= \frac{5}{14} \times \frac{3 \times 1 \times 4 \times 3}{5^4}$

        $= \frac{5}{14} \times \frac{36}{625}$

        $= \frac{5 \times 36}{14 \times 625}$

        $= \frac{180}{8750}$

        $= 0.0206$

        #### step 4: normalize

        sum of probabilities: $0.0053 + 0.0206 = 0.0259$

        normalizing:

        - $P(Yes|evidence) = \frac{0.0053}{0.0259} = 0.205$ (20.5%)
        - $P(No|evidence) = \frac{0.0206}{0.0259} = 0.795$ (79.5%)

        #### step 5: predict

        since $P(No|evidence) > P(Yes|evidence)$, prediction: **No**

        person would **not play tennis** under these conditions.
        """)
    }
    return (solution_accordion,)


@app.cell(hide_code=True)
def create_gaussian_controls(mo):
    # sliders for controlling viz parameters
    class_sep_slider = mo.ui.slider(1.0, 3.0, value=1.5, label="Class Separation")
    noise_slider = mo.ui.slider(0.1, 0.5, step=0.1, value=0.1, label="Noise (reduces class separation)")
    n_samples_slider = mo.ui.slider(50, 200, value=100, step=10, label="Number of Samples")

    # Create a run button to regenerate data
    regenerate_button = mo.ui.run_button(label="Regenerate Data", kind="success")

    # stack controls vertically
    controls = mo.vstack([
        mo.md("### visualization controls"),
        class_sep_slider,
        noise_slider,
        n_samples_slider,
        regenerate_button
    ])
    return (
        class_sep_slider,
        controls,
        n_samples_slider,
        noise_slider,
        regenerate_button,
    )


if __name__ == "__main__":
    app.run()
