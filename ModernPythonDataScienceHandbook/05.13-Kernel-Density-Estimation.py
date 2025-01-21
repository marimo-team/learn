import marimo

__generated_with = "0.10.15"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # In Depth: Kernel Density Estimation
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        In the previous chapter we covered Gaussian mixture models, which are a kind of hybrid between a clustering estimator and a density estimator.
        Recall that a density estimator is an algorithm that takes a $D$-dimensional dataset and produces an estimate of the $D$-dimensional probability distribution that data is drawn from.
        The GMM algorithm accomplishes this by representing the density as a weighted sum of Gaussian distributions.
        *Kernel density estimation* (KDE) is in some senses an algorithm that takes the mixture-of-Gaussians idea to its logical extreme: it uses a mixture consisting of one Gaussian component *per point*, resulting in an essentially nonparametric estimator of density.
        In this chapter, we will explore the motivation and uses of KDE.

        We begin with the standard imports:
        """
    )
    return


@app.cell
def _():
    # "%matplotlib inline\nimport matplotlib.pyplot as plt\nplt.style.use('seaborn-whitegrid')\nimport numpy as np" command supported automatically in marimo
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Motivating Kernel Density Estimation: Histograms

        As mentioned previously, a density estimator is an algorithm that seeks to model the probability distribution that generated a dataset.
        For one-dimensional data, you are probably already familiar with one simple density estimator: the histogram.
        A histogram divides the data into discrete bins, counts the number of points that fall in each bin, and then visualizes the results in an intuitive manner.

        For example, let's create some data that is drawn from two normal distributions:
        """
    )
    return


@app.cell
def _(np):
    def make_data(N, f=0.3, rseed=1):
        rand = np.random.RandomState(rseed)
        x = rand.randn(N)
        x[int(f * N):] = x[int(f * N):] + 5
        return x
    x = make_data(1000)
    return make_data, x


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We have previously seen that the standard count-based histogram can be created with the `plt.hist` function.
        By specifying the `density` parameter of the histogram, we end up with a normalized histogram where the height of the bins does not reflect counts, but instead reflects probability density (see the following figure):
        """
    )
    return


@app.cell
def _(plt, x):
    hist = plt.hist(x, bins=30, density=True)
    return (hist,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Notice that for equal binning, this normalization simply changes the scale on the y-axis, leaving the relative heights essentially the same as in a histogram built from counts.
        This normalization is chosen so that the total area under the histogram is equal to 1, as we can confirm by looking at the output of the histogram function:
        """
    )
    return


@app.cell
def _(hist):
    _density, bins, patches = hist
    widths = bins[1:] - bins[:-1]
    (_density * widths).sum()
    return bins, patches, widths


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        One of the issues with using a histogram as a density estimator is that the choice of bin size and location can lead to representations that have qualitatively different features.
        For example, if we look at a version of this data with only 20 points, the choice of how to draw the bins can lead to an entirely different interpretation of the data!
        Consider this example, visualized in the following figure:
        """
    )
    return


@app.cell
def _(make_data, np):
    x_1 = make_data(20)
    bins_1 = np.linspace(-5, 10, 10)
    return bins_1, x_1


@app.cell
def _(bins_1, np, plt, x_1):
    _fig, _ax = plt.subplots(1, 2, figsize=(12, 4), sharex=True, sharey=True, subplot_kw={'xlim': (-4, 9), 'ylim': (-0.02, 0.3)})
    _fig.subplots_adjust(wspace=0.05)
    for _i, offset in enumerate([0.0, 0.6]):
        _ax[_i].hist(x_1, bins=bins_1 + offset, density=True)
        _ax[_i].plot(x_1, np.full_like(x_1, -0.01), '|k', markeredgewidth=1)
    return (offset,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        On the left, the histogram makes clear that this is a bimodal distribution.
        On the right, we see a unimodal distribution with a long tail.
        Without seeing the preceding code, you would probably not guess that these two histograms were built from the same data. With that in mind, how can you trust the intuition that histograms confer?
        And how might we improve on this?

        Stepping back, we can think of a histogram as a stack of blocks, where we stack one block within each bin on top of each point in the dataset.
        Let's view this directly (see the following figure):
        """
    )
    return


@app.cell
def _(np, plt, x_1):
    _fig, _ax = plt.subplots()
    bins_2 = np.arange(-3, 8)
    _ax.plot(x_1, np.full_like(x_1, -0.1), '|k', markeredgewidth=1)
    for count, edge in zip(*np.histogram(x_1, bins_2)):
        for _i in range(count):
            _ax.add_patch(plt.Rectangle((edge, _i), 1, 1, ec='black', alpha=0.5))
    _ax.set_xlim(-4, 8)
    _ax.set_ylim(-0.2, 8)
    return bins_2, count, edge


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The problem with our two binnings stems from the fact that the height of the block stack often reflects not the actual density of points nearby, but coincidences of how the bins align with the data points.
        This misalignment between points and their blocks is a potential cause of the poor histogram results seen here.
        But what if, instead of stacking the blocks aligned with the *bins*, we were to stack the blocks aligned with the *points they represent*?
        If we do this, the blocks won't be aligned, but we can add their contributions at each location along the x-axis to find the result.
        Let's try this (see the following figure):
        """
    )
    return


@app.cell
def _(np, plt, x_1):
    x_d = np.linspace(-4, 8, 2000)
    _density = sum((abs(xi - x_d) < 0.5 for xi in x_1))
    plt.fill_between(x_d, _density, alpha=0.5)
    plt.plot(x_1, np.full_like(x_1, -0.1), '|k', markeredgewidth=1)
    plt.axis([-4, 8, -0.2, 8])
    return (x_d,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The result looks a bit messy, but it's a much more robust reflection of the actual data characteristics than is the standard histogram.
        Still, the rough edges are not aesthetically pleasing, nor are they reflective of any true properties of the data.
        In order to smooth them out, we might decide to replace the blocks at each location with a smooth function, like a Gaussian.
        Let's use a standard normal curve at each point instead of a block (see the following figure):
        """
    )
    return


@app.cell
def _(np, plt, x_1):
    from scipy.stats import norm
    x_d_1 = np.linspace(-4, 8, 1000)
    _density = sum((norm(xi).pdf(x_d_1) for xi in x_1))
    plt.fill_between(x_d_1, _density, alpha=0.5)
    plt.plot(x_1, np.full_like(x_1, -0.1), '|k', markeredgewidth=1)
    plt.axis([-4, 8, -0.2, 5])
    return norm, x_d_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        This smoothed-out plot, with a Gaussian distribution contributed at the location of each input point, gives a much more accurate idea of the shape of the data distribution, and one that has much less variance (i.e., changes much less in response to differences in sampling).

        What we've landed on in the last two plots is what's called kernel density estimation in one dimension: we have placed a "kernel"—a square or "tophat"-shaped kernel in the former, a Gaussian kernel in the latter—at the location of each point, and used their sum as an estimate of density.
        With this intuition in mind, we'll now explore kernel density estimation in more detail.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Kernel Density Estimation in Practice

        The free parameters of kernel density estimation are the *kernel*, which specifies the shape of the distribution placed at each point, and the *kernel bandwidth*, which controls the size of the kernel at each point.
        In practice, there are many kernels you might use for kernel density estimation: in particular, the Scikit-Learn KDE implementation supports six kernels, which you can read about in the ["Density Estimation" section](http://scikit-learn.org/stable/modules/density.html) of the documentation.

        While there are several versions of KDE implemented in Python (notably in the SciPy and `statsmodels` packages), I prefer to use Scikit-Learn's version because of its efficiency and flexibility.
        It is implemented in the `sklearn.neighbors.KernelDensity` estimator, which handles KDE in multiple dimensions with one of six kernels and one of a couple dozen distance metrics.
        Because KDE can be fairly computationally intensive, the Scikit-Learn estimator uses a tree-based algorithm under the hood and can trade off computation time for accuracy using the `atol` (absolute tolerance) and `rtol` (relative tolerance) parameters.
        The kernel bandwidth can be determined using Scikit-Learn's standard cross-validation tools, as we will soon see.

        Let's first show a simple example of replicating the previous plot using the Scikit-Learn `KernelDensity` estimator (see the following figure):
        """
    )
    return


@app.cell
def _(np, plt, x_1, x_d_1):
    from sklearn.neighbors import KernelDensity
    kde = KernelDensity(bandwidth=1.0, kernel='gaussian')
    kde.fit(x_1[:, None])
    logprob = kde.score_samples(x_d_1[:, None])
    plt.fill_between(x_d_1, np.exp(logprob), alpha=0.5)
    plt.plot(x_1, np.full_like(x_1, -0.01), '|k', markeredgewidth=1)
    plt.ylim(-0.02, 0.22)
    return KernelDensity, kde, logprob


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The result here is normalized such that the area under the curve is equal to 1.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Selecting the Bandwidth via Cross-Validation

        The final estimate produced by a KDE procedure can be quite sensitive to the choice of bandwidth, which is the knob that controls the bias–variance trade-off in the estimate of density.
        Too narrow a bandwidth leads to a high-variance estimate (i.e., overfitting), where the presence or absence of a single point makes a large difference. Too wide a bandwidth leads to a high-bias estimate (i.e., underfitting), where the structure in the data is washed out by the wide kernel.

        There is a long history in statistics of methods to quickly estimate the best bandwidth based on rather stringent assumptions about the data: if you look up the KDE implementations in the SciPy and `statsmodels` packages, for example, you will see implementations based on some of these rules.

        In machine learning contexts, we've seen that such hyperparameter tuning often is done empirically via a cross-validation approach.
        With this in mind, Scikit-Learn's `KernelDensity` estimator is designed such that it can be used directly within the package's standard grid search tools.
        Here we will use `GridSearchCV` to optimize the bandwidth for the preceding dataset.
        Because we are looking at such a small dataset, we will use leave-one-out cross-validation, which minimizes the reduction in training set size for each cross-validation trial:
        """
    )
    return


@app.cell
def _(KernelDensity, np, x_1):
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import LeaveOneOut
    bandwidths = 10 ** np.linspace(-1, 1, 100)
    grid = GridSearchCV(KernelDensity(kernel='gaussian'), {'bandwidth': bandwidths}, cv=LeaveOneOut())
    grid.fit(x_1[:, None])
    return GridSearchCV, LeaveOneOut, bandwidths, grid


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Now we can find the choice of bandwidth that maximizes the score (which in this case defaults to the log-likelihood):
        """
    )
    return


@app.cell
def _(grid):
    grid.best_params_
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The optimal bandwidth happens to be very close to what we used in the example plot earlier, where the bandwidth was 1.0 (i.e., the default width of `scipy.stats.norm`).
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Example: Not-so-Naive Bayes

        This example looks at Bayesian generative classification with KDE, and demonstrates how to use the Scikit-Learn architecture to create a custom estimator.

        In [In Depth: Naive Bayes Classification](05.05-Naive-Bayes.ipynb) we explored naive Bayesian classification, in which we create a simple generative model for each class, and use these models to build a fast classifier.
        For Gaussian naive Bayes, the generative model is a simple axis-aligned Gaussian.
        With a density estimation algorithm like KDE, we can remove the "naive" element and perform the same classification with a more sophisticated generative model for each class.
        It's still Bayesian classification, but it's no longer naive.

        The general approach for generative classification is this:

        1. Split the training data by label.

        2. For each set, fit a KDE to obtain a generative model of the data.
           This allows you, for any observation $x$ and label $y$, to compute a likelihood $P(x~|~y)$.
           
        3. From the number of examples of each class in the training set, compute the *class prior*, $P(y)$.

        4. For an unknown point $x$, the posterior probability for each class is $P(y~|~x) \propto P(x~|~y)P(y)$.
           The class that maximizes this posterior is the label assigned to the point.

        The algorithm is straightforward and intuitive to understand; the more difficult piece is couching it within the Scikit-Learn framework in order to make use of the grid search and cross-validation architecture.

        This is the code that implements the algorithm within the Scikit-Learn framework; we will step through it following the code block:
        """
    )
    return


@app.cell
def _(KernelDensity, np):
    from sklearn.base import BaseEstimator, ClassifierMixin


    class KDEClassifier(BaseEstimator, ClassifierMixin):
        """Bayesian generative classification based on KDE
        
        Parameters
        ----------
        bandwidth : float
            the kernel bandwidth within each class
        kernel : str
            the kernel name, passed to KernelDensity
        """
        def __init__(self, bandwidth=1.0, kernel='gaussian'):
            self.bandwidth = bandwidth
            self.kernel = kernel
            
        def fit(self, X, y):
            self.classes_ = np.sort(np.unique(y))
            training_sets = [X[y == yi] for yi in self.classes_]
            self.models_ = [KernelDensity(bandwidth=self.bandwidth,
                                          kernel=self.kernel).fit(Xi)
                            for Xi in training_sets]
            self.logpriors_ = [np.log(Xi.shape[0] / X.shape[0])
                               for Xi in training_sets]
            return self
            
        def predict_proba(self, X):
            logprobs = np.array([model.score_samples(X)
                                 for model in self.models_]).T
            result = np.exp(logprobs + self.logpriors_)
            return result / result.sum(axis=1, keepdims=True)
            
        def predict(self, X):
            return self.classes_[np.argmax(self.predict_proba(X), 1)]
    return BaseEstimator, ClassifierMixin, KDEClassifier


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Anatomy of a Custom Estimator
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        Let's step through this code and discuss the essential features:

        ```python

        class KDEClassifier(BaseEstimator, ClassifierMixin):
            \"\"\"Bayesian generative classification based on KDE
            
            Parameters
            ----------
            bandwidth : float
                the kernel bandwidth within each class
            kernel : str
                the kernel name, passed to KernelDensity
            \"\"\"
        ```

        Each estimator in Scikit-Learn is a class, and it is most convenient for this class to inherit from the `BaseEstimator` class as well as the appropriate mixin, which provides standard functionality.
        For example, here the `BaseEstimator` contains (among other things) the logic necessary to clone/copy an estimator for use in a cross-validation procedure, and `ClassifierMixin` defines a default `score` method used by such routines.
        We also provide a docstring, which will be captured by IPython's help functionality (see [Help and Documentation in IPython](01.01-Help-And-Documentation.ipynb)).
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Next comes the class initialization method:

        ```python
            def __init__(self, bandwidth=1.0, kernel='gaussian'):
                self.bandwidth = bandwidth
                self.kernel = kernel
        ```

        This is the actual code that is executed when the object is instantiated with `KDEClassifier`.
        In Scikit-Learn, it is important that *initialization contains no operations* other than assigning the passed values by name to `self`.
        This is due to the logic contained in `BaseEstimator` required for cloning and modifying estimators for cross-validation, grid search, and other functions.
        Similarly, all arguments to `__init__` should be explicit: i.e., `*args` or `**kwargs` should be avoided, as they will not be correctly handled within cross-validation routines.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Next comes the `fit` method, where we handle training data:

        ```python 
            def fit(self, X, y):
                self.classes_ = np.sort(np.unique(y))
                training_sets = [X[y == yi] for yi in self.classes_]
                self.models_ = [KernelDensity(bandwidth=self.bandwidth,
                                              kernel=self.kernel).fit(Xi)
                                for Xi in training_sets]
                self.logpriors_ = [np.log(Xi.shape[0] / X.shape[0])
                                   for Xi in training_sets]
                return self
        ```

        Here we find the unique classes in the training data, train a `KernelDensity` model for each class, and compute the class priors based on the number of input samples.
        Finally, `fit` should always return `self` so that we can chain commands. For example:
        ```python
        label = model.fit(X, y).predict(X)
        ```
        Notice that each persistent result of the fit is stored with a trailing underscore (e.g., `self.logpriors_`).
        This is a convention used in Scikit-Learn so that you can quickly scan the members of an estimator (using IPython's tab completion) and see exactly which members are fit to training data.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Finally, we have the logic for predicting labels on new data:
        ```python
            def predict_proba(self, X):
                logprobs = np.vstack([model.score_samples(X)
                                      for model in self.models_]).T
                result = np.exp(logprobs + self.logpriors_)
                return result / result.sum(axis=1, keepdims=True)
                
            def predict(self, X):
                return self.classes_[np.argmax(self.predict_proba(X), 1)]
        ```
        Because this is a probabilistic classifier, we first implement `predict_proba`, which returns an array of class probabilities of shape `[n_samples, n_classes]`.
        Entry `[i, j]` of this array is the posterior probability that sample `i` is a member of class `j`, computed by multiplying the likelihood by the class prior and normalizing.

        The `predict` method uses these probabilities and simply returns the class with the largest probability.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Using Our Custom Estimator

        Let's try this custom estimator on a problem we have seen before: the classification of handwritten digits.
        Here we will load the digits and compute the cross-validation score for a range of candidate bandwidths using the ``GridSearchCV`` meta-estimator (refer back to [Hyperparameters and Model Validation](05.03-Hyperparameters-and-Model-Validation.ipynb)):
        """
    )
    return


@app.cell
def _(GridSearchCV, KDEClassifier, np):
    from sklearn.datasets import load_digits
    digits = load_digits()
    grid_1 = GridSearchCV(KDEClassifier(), {'bandwidth': np.logspace(0, 2, 100)})
    grid_1.fit(digits.data, digits.target)
    return digits, grid_1, load_digits


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Next we can plot the cross-validation score as a function of bandwidth (see the following figure):
        """
    )
    return


@app.cell
def _(grid_1, np, plt):
    _fig, _ax = plt.subplots()
    _ax.semilogx(np.array(grid_1.cv_results_['param_bandwidth']), grid_1.cv_results_['mean_test_score'])
    _ax.set(title='KDE Model Performance', ylim=(0, 1), xlabel='bandwidth', ylabel='accuracy')
    print(f'best param: {grid_1.best_params_}')
    print(f'accuracy = {grid_1.best_score_}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        This indicates that our KDE classifier reaches a cross-validation accuracy of over 96%, compared to around 80% for the naive Bayes classifier:
        """
    )
    return


@app.cell
def _(digits):
    from sklearn.naive_bayes import GaussianNB
    from sklearn.model_selection import cross_val_score
    cross_val_score(GaussianNB(), digits.data, digits.target).mean()
    return GaussianNB, cross_val_score


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        One benefit of such a generative classifier is interpretability of results: for each unknown sample, we not only get a probabilistic classification, but a *full model* of the distribution of points we are comparing it to!
        If desired, this offers an intuitive window into the reasons for a particular classification that algorithms like SVMs and random forests tend to obscure.

        If you would like to take this further, here are some ideas for improvements that could be made to our KDE classifier model:

        - You could allow the bandwidth in each class to vary independently.
        - You could optimize these bandwidths not based on their prediction score, but on the likelihood of the training data under the generative model within each class (i.e. use the scores from `KernelDensity` itself rather than the global prediction accuracy).

        Finally, if you want some practice building your own estimator, you might tackle building a similar Bayesian classifier using Gaussian mixture models instead of KDE.
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
