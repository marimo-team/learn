import marimo

__generated_with = "0.10.15"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # In Depth: Gaussian Mixture Models
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The *k*-means clustering model explored in the previous chapter is simple and relatively easy to understand, but its simplicity leads to practical challenges in its application.
        In particular, the nonprobabilistic nature of *k*-means and its use of simple distance from cluster center to assign cluster membership leads to poor performance for many real-world situations.
        In this chapter we will take a look at Gaussian mixture models, which can be viewed as an extension of the ideas behind *k*-means, but can also be a powerful tool for estimation beyond simple clustering.

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
        ## Motivating Gaussian Mixtures: Weaknesses of k-Means

        Let's take a look at some of the weaknesses of *k*-means and think about how we might improve the cluster model.
        As we saw in the previous chapter, given simple, well-separated data, *k*-means finds suitable clustering results.

        For example, if we have simple blobs of data, the *k*-means algorithm can quickly label those clusters in a way that closely matches what we might do by eye (see the following figure):
        """
    )
    return


@app.cell
def _():
    # Generate some data
    from sklearn.datasets import make_blobs
    X, y_true = make_blobs(n_samples=400, centers=4,
                           cluster_std=0.60, random_state=0)
    X = X[:, ::-1] # flip axes for better plotting
    return X, make_blobs, y_true


@app.cell
def _(X, plt):
    from sklearn.cluster import KMeans
    _kmeans = KMeans(4, random_state=0)
    labels = _kmeans.fit(X).predict(X)
    plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis')
    return KMeans, labels


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        From an intuitive standpoint, we might expect that the clustering assignment for some points is more certain than others: for example, there appears to be a very slight overlap between the two middle clusters, such that we might not have complete confidence in the cluster assignment of points between them.
        Unfortunately, the *k*-means model has no intrinsic measure of probability or uncertainty of cluster assignments (although it may be possible to use a bootstrap approach to estimate this uncertainty).
        For this, we must think about generalizing the model.

        One way to think about the *k*-means model is that it places a circle (or, in higher dimensions, a hypersphere) at the center of each cluster, with a radius defined by the most distant point in the cluster.
        This radius acts as a hard cutoff for cluster assignment within the training set: any point outside this circle is not considered a member of the cluster.
        We can visualize this cluster model with the following function (see the following figure):
        """
    )
    return


@app.cell
def _(plt):
    from scipy.spatial.distance import cdist

    def plot_kmeans(kmeans, X, n_clusters=4, rseed=0, ax=None):
        labels = _kmeans.fit_predict(X)
        ax = ax or plt.gca()
        ax.axis('equal')
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)
        centers = _kmeans.cluster_centers_
        radii = [cdist(X[labels == i], [center]).max() for i, center in enumerate(centers)]
        for c, r in zip(centers, radii):
            ax.add_patch(plt.Circle(c, r, ec='black', fc='lightgray', lw=3, alpha=0.5, zorder=1))
    return cdist, plot_kmeans


@app.cell
def _(KMeans, X, plot_kmeans):
    _kmeans = KMeans(n_clusters=4, random_state=0)
    plot_kmeans(_kmeans, X)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        An important observation for *k*-means is that these cluster models *must be circular*: *k*-means has no built-in way of accounting for oblong or elliptical clusters.
        So, for example, if we take the same data and transform it, the cluster assignments end up becoming muddled, as you can see in the following figure:
        """
    )
    return


@app.cell
def _(KMeans, X, np, plot_kmeans):
    rng = np.random.RandomState(13)
    X_stretched = np.dot(X, rng.randn(2, 2))
    _kmeans = KMeans(n_clusters=4, random_state=0)
    plot_kmeans(_kmeans, X_stretched)
    return X_stretched, rng


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        By eye, we recognize that these transformed clusters are noncircular, and thus circular clusters would be a poor fit.
        Nevertheless, *k*-means is not flexible enough to account for this, and tries to force-fit the data into four circular clusters.
        This results in a mixing of cluster assignments where the resulting circles overlap: see especially the bottom-right of this plot.
        One might imagine addressing this particular situation by preprocessing the data with PCA (see [In Depth: Principal Component Analysis](05.09-Principal-Component-Analysis.ipynb)), but in practice there is no guarantee that such a global operation will circularize the individual groups.

        These two disadvantages of *k*-means—its lack of flexibility in cluster shape and lack of probabilistic cluster assignment—mean that for many datasets (especially low-dimensional datasets) it may not perform as well as you might hope.

        You might imagine addressing these weaknesses by generalizing the *k*-means model: for example, you could measure uncertainty in cluster assignment by comparing the distances of each point to *all* cluster centers, rather than focusing on just the closest.
        You might also imagine allowing the cluster boundaries to be ellipses rather than circles, so as to account for noncircular clusters.
        It turns out these are two essential components of a different type of clustering model, Gaussian mixture models.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Generalizing E–M: Gaussian Mixture Models

        A Gaussian mixture model (GMM) attempts to find a mixture of multidimensional Gaussian probability distributions that best model any input dataset.
        In the simplest case, GMMs can be used for finding clusters in the same manner as *k*-means (see the following figure):
        """
    )
    return


@app.cell
def _(X, plt):
    from sklearn.mixture import GaussianMixture
    gmm = GaussianMixture(n_components=4).fit(X)
    labels_1 = gmm.predict(X)
    plt.scatter(X[:, 0], X[:, 1], c=labels_1, s=40, cmap='viridis')
    return GaussianMixture, gmm, labels_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        But because a GMM contains a probabilistic model under the hood, it is also possible to find probabilistic cluster assignments—in Scikit-Learn this is done using the `predict_proba` method.
        This returns a matrix of size `[n_samples, n_clusters]` which measures the probability that any point belongs to the given cluster:
        """
    )
    return


@app.cell
def _(X, gmm):
    probs = gmm.predict_proba(X)
    print(probs[:5].round(3))
    return (probs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We can visualize this uncertainty by, for example, making the size of each point proportional to the certainty of its prediction; looking at the following figure, we can see that it is precisely the points at the boundaries between clusters that reflect this uncertainty of cluster assignment:
        """
    )
    return


@app.cell
def _(X, labels_1, plt, probs):
    size = 50 * probs.max(1) ** 2
    plt.scatter(X[:, 0], X[:, 1], c=labels_1, cmap='viridis', s=size)
    return (size,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Under the hood, a Gaussian mixture model is very similar to *k*-means: it uses an expectation–maximization approach, which qualitatively does the following:

        1. Choose starting guesses for the location and shape.

        2. Repeat until converged:

           1. *E-step*: For each point, find weights encoding the probability of membership in each cluster.
           2. *M-step*: For each cluster, update its location, normalization, and shape based on *all* data points, making use of the weights.

        The result of this is that each cluster is associated not with a hard-edged sphere, but with a smooth Gaussian model.
        Just as in the *k*-means expectation–maximization approach, this algorithm can sometimes miss the globally optimal solution, and thus in practice multiple random initializations are used.

        Let's create a function that will help us visualize the locations and shapes of the GMM clusters by drawing ellipses based on the GMM output:
        """
    )
    return


@app.cell
def _(np, plt):
    from matplotlib.patches import Ellipse

    def draw_ellipse(position, covariance, ax=None, **kwargs):
        """Draw an ellipse with a given position and covariance"""
        ax = ax or plt.gca()
        
        # Convert covariance to principal axes
        if covariance.shape == (2, 2):
            U, s, Vt = np.linalg.svd(covariance)
            angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
            width, height = 2 * np.sqrt(s)
        else:
            angle = 0
            width, height = 2 * np.sqrt(covariance)
        
        # Draw the ellipse
        for nsig in range(1, 4):
            ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                                 angle, **kwargs))
            
    def plot_gmm(gmm, X, label=True, ax=None):
        ax = ax or plt.gca()
        labels = gmm.fit(X).predict(X)
        if label:
            ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)
        else:
            ax.scatter(X[:, 0], X[:, 1], s=40, zorder=2)
        ax.axis('equal')
        
        w_factor = 0.2 / gmm.weights_.max()
        for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
            draw_ellipse(pos, covar, alpha=w * w_factor)
    return Ellipse, draw_ellipse, plot_gmm


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        With this in place, we can take a look at what the four-component GMM gives us for our initial data (see the following figure):
        """
    )
    return


@app.cell
def _(GaussianMixture, X, plot_gmm):
    gmm_1 = GaussianMixture(n_components=4, random_state=42)
    plot_gmm(gmm_1, X)
    return (gmm_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Similarly, we can use the GMM approach to fit our stretched dataset; allowing for a full covariance the model will fit even very oblong, stretched-out clusters, as we can see in the following figure:
        """
    )
    return


@app.cell
def _(GaussianMixture, X_stretched, plot_gmm):
    gmm_2 = GaussianMixture(n_components=4, covariance_type='full', random_state=42)
    plot_gmm(gmm_2, X_stretched)
    return (gmm_2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        This makes clear that GMMs address the two main practical issues with *k*-means encountered before.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Choosing the Covariance Type

        If you look at the details of the preceding fits, you will see that the `covariance_type` option was set differently within each.
        This hyperparameter controls the degrees of freedom in the shape of each cluster; it is essential to set this carefully for any given problem.
        The default is `covariance_type="diag"`, which means that the size of the cluster along each dimension can be set independently, with the resulting ellipse constrained to align with the axes.
        A slightly simpler and faster model is `covariance_type="spherical"`, which constrains the shape of the cluster such that all dimensions are equal. The resulting clustering will have similar characteristics to that of *k*-means, though it is not entirely equivalent.
        A more complicated and computationally expensive model (especially as the number of dimensions grows) is to use `covariance_type="full"`, which allows each cluster to be modeled as an ellipse with arbitrary orientation.

        We can see a visual representation of these three choices for a single cluster within the following figure:
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ![(Covariance Type)](images/05.12-covariance-type.png)
        [figure source in Appendix](https://github.com/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/06.00-Figure-Code.ipynb#Covariance-Type)
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Gaussian Mixture Models as Density Estimation

        Though the GMM is often categorized as a clustering algorithm, fundamentally it is an algorithm for *density estimation*.
        That is to say, the result of a GMM fit to some data is technically not a clustering model, but a generative probabilistic model describing the distribution of the data.

        As an example, consider some data generated from Scikit-Learn's `make_moons` function, introduced in [In Depth: K-Means Clustering](05.11-K-Means.ipynb) (see the following figure):
        """
    )
    return


@app.cell
def _(plt):
    from sklearn.datasets import make_moons
    Xmoon, ymoon = make_moons(200, noise=.05, random_state=0)
    plt.scatter(Xmoon[:, 0], Xmoon[:, 1]);
    return Xmoon, make_moons, ymoon


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        If we try to fit this with a two-component GMM viewed as a clustering model, the results are not particularly useful (see the following figure):
        """
    )
    return


@app.cell
def _(GaussianMixture, Xmoon, plot_gmm):
    gmm2 = GaussianMixture(n_components=2, covariance_type='full', random_state=0)
    plot_gmm(gmm2, Xmoon)
    return (gmm2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        But if we instead use many more components and ignore the cluster labels, we find a fit that is much closer to the input data (see the following figure):
        """
    )
    return


@app.cell
def _(GaussianMixture, Xmoon, plot_gmm):
    gmm16 = GaussianMixture(n_components=16, covariance_type='full', random_state=0)
    plot_gmm(gmm16, Xmoon, label=False)
    return (gmm16,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Here the mixture of 16 Gaussian components serves not to find separated clusters of data, but rather to model the overall *distribution* of the input data.
        This is a generative model of the distribution, meaning that the GMM gives us the recipe to generate new random data distributed similarly to our input.
        For example, here are 400 new points drawn from this 16-component GMM fit to our original data (see the following figure):
        """
    )
    return


@app.cell
def _(gmm16, plt):
    Xnew, ynew = gmm16.sample(400)
    plt.scatter(Xnew[:, 0], Xnew[:, 1]);
    return Xnew, ynew


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        A GMM is convenient as a flexible means of modeling an arbitrary multidimensional distribution of data.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### How Many Components?

        The fact that a GMM is a generative model gives us a natural means of determining the optimal number of components for a given dataset.
        A generative model is inherently a probability distribution for the dataset, and so we can simply evaluate the *likelihood* of the data under the model, using cross-validation to avoid overfitting.
        Another means of correcting for overfitting is to adjust the model likelihoods using some analytic criterion such as the [Akaike information criterion (AIC)](https://en.wikipedia.org/wiki/Akaike_information_criterion) or the [Bayesian information criterion (BIC)](https://en.wikipedia.org/wiki/Bayesian_information_criterion).
        Scikit-Learn's `GaussianMixture` estimator actually includes built-in methods that compute both of these, so it is very easy to operate using this approach.

        Let's look at the AIC and BIC versus the number of GMM components for our moons dataset (see the following figure):
        """
    )
    return


@app.cell
def _(GaussianMixture, Xmoon, np, plt):
    _n_components = np.arange(1, 21)
    _models = [GaussianMixture(n, covariance_type='full', random_state=0).fit(Xmoon) for n in _n_components]
    plt.plot(_n_components, [m.bic(Xmoon) for m in _models], label='BIC')
    plt.plot(_n_components, [m.aic(Xmoon) for m in _models], label='AIC')
    plt.legend(loc='best')
    plt.xlabel('n_components')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The optimal number of clusters is the value that minimizes the AIC or BIC, depending on which approximation we wish to use. The AIC tells us that our choice of 16 components earlier was probably too many: around 8–12 components would have been a better choice.
        As is typical with this sort of problem, the BIC recommends a simpler model.

        Notice the important point: this choice of number of components measures how well a GMM works *as a density estimator*, not how well it works *as a clustering algorithm*.
        I'd encourage you to think of the GMM primarily as a density estimator, and use it for clustering only when warranted within simple datasets.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Example: GMMs for Generating New Data

        We just saw a simple example of using a GMM as a generative model in order to create new samples from the distribution defined by the input data.
        Here we will run with this idea and generate *new handwritten digits* from the standard digits corpus that we have used before.

        To start with, let's load the digits data using Scikit-Learn's data tools:
        """
    )
    return


@app.cell
def _():
    from sklearn.datasets import load_digits
    digits = load_digits()
    digits.data.shape
    return digits, load_digits


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Next, let's plot the first 50 of these to recall exactly what we're looking at (see the following figure):
        """
    )
    return


@app.cell
def _(digits, plt):
    def plot_digits(data):
        fig, ax = plt.subplots(5, 10, figsize=(8, 4),
                               subplot_kw=dict(xticks=[], yticks=[]))
        fig.subplots_adjust(hspace=0.05, wspace=0.05)
        for i, axi in enumerate(ax.flat):
            im = axi.imshow(data[i].reshape(8, 8), cmap='binary')
            im.set_clim(0, 16)
    plot_digits(digits.data)
    return (plot_digits,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We have nearly 1,800 digits in 64 dimensions, and we can build a GMM on top of these to generate more.
        GMMs can have difficulty converging in such a high-dimensional space, so we will start with an invertible dimensionality reduction algorithm on the data.
        Here we will use a straightforward PCA, asking it to preserve 99% of the variance in the projected data:
        """
    )
    return


@app.cell
def _(digits):
    from sklearn.decomposition import PCA
    pca = PCA(0.99, whiten=True)
    data = pca.fit_transform(digits.data)
    data.shape
    return PCA, data, pca


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The result is 41 dimensions, a reduction of nearly 1/3 with almost no information loss.
        Given this projected data, let's use the AIC to get a gauge for the number of GMM components we should use (see the following figure):
        """
    )
    return


@app.cell
def _(GaussianMixture, data, np, plt):
    _n_components = np.arange(50, 210, 10)
    _models = [GaussianMixture(n, covariance_type='full', random_state=0) for n in _n_components]
    aics = [model.fit(data).aic(data) for model in _models]
    plt.plot(_n_components, aics)
    return (aics,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        It appears that around 140 components minimizes the AIC; we will use this model.
        Let's quickly fit this to the data and confirm that it has converged:
        """
    )
    return


@app.cell
def _(GaussianMixture, data):
    gmm_3 = GaussianMixture(140, covariance_type='full', random_state=0)
    gmm_3.fit(data)
    print(gmm_3.converged_)
    return (gmm_3,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Now we can draw samples of 100 new points within this 41-dimensional projected space, using the GMM as a generative model:
        """
    )
    return


@app.cell
def _(gmm_3):
    data_new, label_new = gmm_3.sample(100)
    data_new.shape
    return data_new, label_new


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Finally, we can use the inverse transform of the PCA object to construct the new digits (see the following figure):
        """
    )
    return


@app.cell
def _(data_new, pca, plot_digits):
    digits_new = pca.inverse_transform(data_new)
    plot_digits(digits_new)
    return (digits_new,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The results for the most part look like plausible digits from the dataset!

        Consider what we've done here: given a sampling of handwritten digits, we have modeled the distribution of that data in such a way that we can generate brand new samples of digits from the data: these are "handwritten digits," which do not individually appear in the original dataset, but rather capture the general features of the input data as modeled by the mixture model.
        Such a generative model of digits can prove very useful as a component of a Bayesian generative classifier, as we shall see in the next chapter.
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
