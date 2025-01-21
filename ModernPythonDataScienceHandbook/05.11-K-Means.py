import marimo

__generated_with = "0.10.15"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # In Depth: k-Means Clustering
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        In the previous chapters we explored unsupervised machine learning models for dimensionality reduction.
        Now we will move on to another class of unsupervised machine learning models: clustering algorithms.
        Clustering algorithms seek to learn, from the properties of the data, an optimal division or discrete labeling of groups of points.

        Many clustering algorithms are available in Scikit-Learn and elsewhere, but perhaps the simplest to understand is an algorithm known as *k-means clustering*, which is implemented in `sklearn.cluster.KMeans`.

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
        ## Introducing k-Means
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The *k*-means algorithm searches for a predetermined number of clusters within an unlabeled multidimensional dataset.
        It accomplishes this using a simple conception of what the optimal clustering looks like:

        - The *cluster center* is the arithmetic mean of all the points belonging to the cluster.
        - Each point is closer to its own cluster center than to other cluster centers.

        Those two assumptions are the basis of the *k*-means model.
        We will soon dive into exactly *how* the algorithm reaches this solution, but for now let's take a look at a simple dataset and see the *k*-means result.

        First, let's generate a two-dimensional dataset containing four distinct blobs.
        To emphasize that this is an unsupervised algorithm, we will leave the labels out of the visualization (see the following figure):
        """
    )
    return


@app.cell
def _(plt):
    from sklearn.datasets import make_blobs
    X, y_true = make_blobs(n_samples=300, centers=4,
                           cluster_std=0.60, random_state=0)
    plt.scatter(X[:, 0], X[:, 1], s=50);
    return X, make_blobs, y_true


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        By eye, it is relatively easy to pick out the four clusters.
        The *k*-means algorithm does this automatically, and in Scikit-Learn uses the typical estimator API:
        """
    )
    return


@app.cell
def _(X):
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=4)
    kmeans.fit(X)
    y_kmeans = kmeans.predict(X)
    return KMeans, kmeans, y_kmeans


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Let's visualize the results by plotting the data colored by these labels (the following figure).
        We will also plot the cluster centers as determined by the *k*-means estimator:
        """
    )
    return


@app.cell
def _(X, kmeans, plt, y_kmeans):
    plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
    _centers = kmeans.cluster_centers_
    plt.scatter(_centers[:, 0], _centers[:, 1], c='black', s=200)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The good news is that the *k*-means algorithm (at least in this simple case) assigns the points to clusters very similarly to how we might assign them by eye.
        But you might wonder how this algorithm finds these clusters so quickly: after all, the number of possible combinations of cluster assignments is exponential in the number of data points—an exhaustive search would be very, very costly.
        Fortunately for us, such an exhaustive search is not necessary: instead, the typical approach to *k*-means involves an intuitive iterative approach known as *expectation–maximization*.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Expectation–Maximization
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Expectation–maximization (E–M) is a powerful algorithm that comes up in a variety of contexts within data science.
        *k*-means is a particularly simple and easy-to-understand application of the algorithm, and we will walk through it briefly here.
        In short, the expectation–maximization approach here consists of the following procedure:

        1. Guess some cluster centers.
        2. Repeat until converged:
           1. *E-step*: Assign points to the nearest cluster center.
           2. *M-step*: Set the cluster centers to the mean of their assigned points.

        Here the *E-step* or *expectation step* is so named because it involves updating our expectation of which cluster each point belongs to.
        The *M-step* or *maximization step* is so named because it involves maximizing some fitness function that defines the locations of the cluster centers—in this case, that maximization is accomplished by taking a simple mean of the data in each cluster.

        The literature about this algorithm is vast, but can be summarized as follows: under typical circumstances, each repetition of the E-step and M-step will always result in a better estimate of the cluster characteristics.

        We can visualize the algorithm as shown in the following figure.
        For the particular initialization shown here, the clusters converge in just three iterations.
        (For an interactive version of this figure, refer to the code in the online [appendix](https://github.com/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/06.00-Figure-Code.ipynb#Interactive-K-Means).)
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ![(run code in Appendix to generate image)](images/05.11-expectation-maximization.png)
        [figure source in Appendix](https://github.com/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/06.00-Figure-Code.ipynb#Expectation-Maximization)
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The *k*-means algorithm is simple enough that we can write it in a few lines of code.
        The following is a very basic implementation (see the following figure):
        """
    )
    return


@app.cell
def _(X, np, plt):
    from sklearn.metrics import pairwise_distances_argmin

    def find_clusters(X, n_clusters, rseed=2):
        rng = np.random.RandomState(rseed)
        _i = rng.permutation(X.shape[0])[:n_clusters]
        _centers = X[_i]
        while True:
            labels = pairwise_distances_argmin(X, _centers)
            new_centers = np.array([X[labels == _i].mean(0) for _i in range(n_clusters)])
            if np.all(_centers == new_centers):
                break
            _centers = new_centers
        return (_centers, labels)
    _centers, labels = find_clusters(X, 4)
    plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
    return find_clusters, labels, pairwise_distances_argmin


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Most well-tested implementations will do a bit more than this under the hood, but the preceding function gives the gist of the expectation–maximization approach.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        There are a few caveats to be aware of when using the expectation–maximization algorithm:
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### The globally optimal result may not be achieved
        First, although the E–M procedure is guaranteed to improve the result in each step, there is no assurance that it will lead to the *global* best solution.
        For example, if we use a different random seed in our simple procedure, the particular starting guesses lead to poor results (see the following figure):
        """
    )
    return


@app.cell
def _(X, find_clusters, plt):
    _centers, labels_1 = find_clusters(X, 4, rseed=0)
    plt.scatter(X[:, 0], X[:, 1], c=labels_1, s=50, cmap='viridis')
    return (labels_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Here the E–M approach has converged, but has not converged to a globally optimal configuration. For this reason, it is common for the algorithm to be run for multiple starting guesses, as indeed Scikit-Learn does by default (the number is set by the ``n_init`` parameter, which defaults to 10).
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### The number of clusters must be selected beforehand
        Another common challenge with *k*-means is that you must tell it how many clusters you expect: it cannot learn the number of clusters from the data.
        For example, if we ask the algorithm to identify six clusters, it will happily proceed and find the best six clusters, as shown in Figure 47-6:
        """
    )
    return


@app.cell
def _(KMeans, X, plt):
    labels_2 = KMeans(6, random_state=0).fit_predict(X)
    plt.scatter(X[:, 0], X[:, 1], c=labels_2, s=50, cmap='viridis')
    return (labels_2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Whether the result is meaningful is a question that is difficult to answer definitively; one approach that is rather intuitive, but that we won't discuss further here, is called [silhouette analysis](http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html).

        Alternatively, you might use a more complicated clustering algorithm that has a better quantitative measure of the fitness per number of clusters (e.g., Gaussian mixture models; see [In Depth: Gaussian Mixture Models](05.12-Gaussian-Mixtures.ipynb)) or which *can* choose a suitable number of clusters (e.g., DBSCAN, mean-shift, or affinity propagation, all available in the `sklearn.cluster` submodule).
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### k-means is limited to linear cluster boundaries
        The fundamental model assumptions of *k*-means (points will be closer to their own cluster center than to others) means that the algorithm will often be ineffective if the clusters have complicated geometries.

        In particular, the boundaries between *k*-means clusters will always be linear, which means that it will fail for more complicated boundaries.
        Consider the following data, along with the cluster labels found by the typical *k*-means approach (see the following figure):
        """
    )
    return


@app.cell
def _():
    from sklearn.datasets import make_moons
    X_1, y = make_moons(200, noise=0.05, random_state=0)
    return X_1, make_moons, y


@app.cell
def _(KMeans, X_1, plt):
    labels_3 = KMeans(2, random_state=0).fit_predict(X_1)
    plt.scatter(X_1[:, 0], X_1[:, 1], c=labels_3, s=50, cmap='viridis')
    return (labels_3,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        This situation is reminiscent of the discussion in [In-Depth: Support Vector Machines](05.07-Support-Vector-Machines.ipynb), where we used a kernel transformation to project the data into a higher dimension where a linear separation is possible.
        We might imagine using the same trick to allow *k*-means to discover non-linear boundaries.

        One version of this kernelized *k*-means is implemented in Scikit-Learn within the ``SpectralClustering`` estimator.
        It uses the graph of nearest neighbors to compute a higher-dimensional representation of the data, and then assigns labels using a *k*-means algorithm (see the following figure):
        """
    )
    return


@app.cell
def _(X_1, plt):
    from sklearn.cluster import SpectralClustering
    model = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', assign_labels='kmeans')
    labels_4 = model.fit_predict(X_1)
    plt.scatter(X_1[:, 0], X_1[:, 1], c=labels_4, s=50, cmap='viridis')
    return SpectralClustering, labels_4, model


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We see that with this kernel transform approach, the kernelized *k*-means is able to find the more complicated nonlinear boundaries between clusters.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### k-means can be slow for large numbers of samples
        Because each iteration of *k*-means must access every point in the dataset, the algorithm can be relatively slow as the number of samples grows.
        You might wonder if this requirement to use all data at each iteration can be relaxed; for example, you might just use a subset of the data to update the cluster centers at each step.
        This is the idea behind batch-based *k*-means algorithms, one form of which is implemented in `sklearn.cluster.MiniBatchKMeans`.
        The interface for this is the same as for standard `KMeans`; we will see an example of its use as we continue our discussion.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Examples

        Being careful about these limitations of the algorithm, we can use *k*-means to our advantage in a variety of situations.
        We'll now take a look at a couple of examples.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Example 1: k-Means on Digits

        To start, let's take a look at applying *k*-means on the same simple digits data that we saw in [In-Depth: Decision Trees and Random Forests](05.08-Random-Forests.ipynb) and [In Depth: Principal Component Analysis](05.09-Principal-Component-Analysis.ipynb).
        Here we will attempt to use *k*-means to try to identify similar digits *without using the original label information*; this might be similar to a first step in extracting meaning from a new dataset about which you don't have any *a priori* label information.

        We will start by loading the dataset, then find the clusters.
        Recall that the digits dataset consists of 1,797 samples with 64 features, where each of the 64 features is the brightness of one pixel in an 8 × 8 image:
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
        The clustering can be performed as we did before:
        """
    )
    return


@app.cell
def _(KMeans, digits):
    kmeans_1 = KMeans(n_clusters=10, random_state=0)
    clusters = kmeans_1.fit_predict(digits.data)
    kmeans_1.cluster_centers_.shape
    return clusters, kmeans_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The result is 10 clusters in 64 dimensions.
        Notice that the cluster centers themselves are 64-dimensional points, and can be interpreted as representing the "typical" digit within the cluster.
        Let's see what these cluster centers look like (see the following figure):
        """
    )
    return


@app.cell
def _(kmeans_1, plt):
    _fig, _ax = plt.subplots(2, 5, figsize=(8, 3))
    _centers = kmeans_1.cluster_centers_.reshape(10, 8, 8)
    for axi, center in zip(_ax.flat, _centers):
        axi.set(xticks=[], yticks=[])
        axi.imshow(center, interpolation='nearest', cmap=plt.cm.binary)
    return axi, center


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We see that *even without the labels*, ``KMeans`` is able to find clusters whose centers are recognizable digits, with perhaps the exception of 1 and 8.

        Because *k*-means knows nothing about the identities of the clusters, the 0–9 labels may be permuted.
        We can fix this by matching each learned cluster label with the true labels found in the clusters:
        """
    )
    return


@app.cell
def _(clusters, digits, np):
    from scipy.stats import mode
    labels_5 = np.zeros_like(clusters)
    for _i in range(10):
        _mask = clusters == _i
        labels_5[_mask] = mode(digits.target[_mask])[0]
    return labels_5, mode


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Now we can check how accurate our unsupervised clustering was in finding similar digits within the data:
        """
    )
    return


@app.cell
def _(digits, labels_5):
    from sklearn.metrics import accuracy_score
    accuracy_score(digits.target, labels_5)
    return (accuracy_score,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        With just a simple *k*-means algorithm, we discovered the correct grouping for 80% of the input digits!
        Let's check the confusion matrix for this, visualized in the following figure:
        """
    )
    return


@app.cell
def _(digits, labels_5, plt):
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    mat = confusion_matrix(digits.target, labels_5)
    sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False, cmap='Blues', xticklabels=digits.target_names, yticklabels=digits.target_names)
    plt.xlabel('true label')
    plt.ylabel('predicted label')
    return confusion_matrix, mat, sns


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        As we might expect from the cluster centers we visualized before, the main point of confusion is between the eights and ones.
        But this still shows that using *k*-means, we can essentially build a digit classifier *without reference to any known labels*!

        Just for fun, let's try to push this even farther.
        We can use the t-distributed stochastic neighbor embedding algorithm (mentioned in [In-Depth: Manifold Learning](05.10-Manifold-Learning.ipynb)) to preprocess the data before performing *k*-means.
        t-SNE is a nonlinear embedding algorithm that is particularly adept at preserving points within clusters.
        Let's see how it does:
        """
    )
    return


@app.cell
def _(KMeans, accuracy_score, digits, mode, np):
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, init='random', learning_rate='auto', random_state=0)
    digits_proj = tsne.fit_transform(digits.data)
    kmeans_2 = KMeans(n_clusters=10, random_state=0)
    clusters_1 = kmeans_2.fit_predict(digits_proj)
    labels_6 = np.zeros_like(clusters_1)
    for _i in range(10):
        _mask = clusters_1 == _i
        labels_6[_mask] = mode(digits.target[_mask])[0]
    accuracy_score(digits.target, labels_6)
    return TSNE, clusters_1, digits_proj, kmeans_2, labels_6, tsne


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        That's a 94% classification accuracy *without using the labels*.
        This is the power of unsupervised learning when used carefully: it can extract information from the dataset that it might be difficult to extract by hand or by eye.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Example 2: k-Means for Color Compression

        One interesting application of clustering is in color compression within images (this example is adapted from Scikit-Learn's ["Color Quantization Using K-Means"](https://scikit-learn.org/stable/auto_examples/cluster/plot_color_quantization.html).
        For example, imagine you have an image with millions of colors.
        In most images, a large number of the colors will be unused, and many of the pixels in the image will have similar or even identical colors.

        For example, consider the image shown in the following figure, which is from the Scikit-Learn `datasets` module (for this to work, you'll have to have the `PIL` Python package installed):
        (For a color version of this and following images, see the online version of this book).
        """
    )
    return


@app.cell
def _(plt):
    from sklearn.datasets import load_sample_image
    china = load_sample_image('china.jpg')
    _ax = plt.axes(xticks=[], yticks=[])
    _ax.imshow(china)
    return china, load_sample_image


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The image itself is stored in a three-dimensional array of size `(height, width, RGB)`, containing red/blue/green contributions as integers from 0 to 255:
        """
    )
    return


@app.cell
def _(china):
    china.shape
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        One way we can view this set of pixels is as a cloud of points in a three-dimensional color space.
        We will reshape the data to `[n_samples, n_features]` and rescale the colors so that they lie between 0 and 1:
        """
    )
    return


@app.cell
def _(china):
    data = china / 255.0  # use 0...1 scale
    data = data.reshape(-1, 3)
    data.shape
    return (data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We can visualize these pixels in this color space, using a subset of 10,000 pixels for efficiency (see the following figure):
        """
    )
    return


@app.cell
def _(np, plt):
    def plot_pixels(data, title, colors=None, N=10000):
        if colors is None:
            colors = data
        rng = np.random.default_rng(0)
        _i = rng.permutation(data.shape[0])[:N]
        colors = colors[_i]
        R, G, B = data[_i].T
        _fig, _ax = plt.subplots(1, 2, figsize=(16, 6))
        _ax[0].scatter(R, G, color=colors, marker='.')
        _ax[0].set(xlabel='Red', ylabel='Green', xlim=(0, 1), ylim=(0, 1))
        _ax[1].scatter(R, B, color=colors, marker='.')
        _ax[1].set(xlabel='Red', ylabel='Blue', xlim=(0, 1), ylim=(0, 1))
        _fig.suptitle(title, size=20)
    return (plot_pixels,)


@app.cell
def _(data, plot_pixels):
    plot_pixels(data, title='Input color space: 16 million possible colors')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Now let's reduce these 16 million colors to just 16 colors, using a *k*-means clustering across the pixel space.
        Because we are dealing with a very large dataset, we will use the mini-batch *k*-means, which operates on subsets of the data to compute the result (shown in the following figure) much more quickly than the standard *k*-means algorithm:
        """
    )
    return


@app.cell
def _(data, plot_pixels):
    from sklearn.cluster import MiniBatchKMeans
    kmeans_3 = MiniBatchKMeans(16)
    kmeans_3.fit(data)
    new_colors = kmeans_3.cluster_centers_[kmeans_3.predict(data)]
    plot_pixels(data, colors=new_colors, title='Reduced color space: 16 colors')
    return MiniBatchKMeans, kmeans_3, new_colors


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The result is a recoloring of the original pixels, where each pixel is assigned the color of its closest cluster center.
        Plotting these new colors in the image space rather than the pixel space shows us the effect of this (see the following figure):
        """
    )
    return


@app.cell
def _(china, new_colors, plt):
    china_recolored = new_colors.reshape(china.shape)
    _fig, _ax = plt.subplots(1, 2, figsize=(16, 6), subplot_kw=dict(xticks=[], yticks=[]))
    _fig.subplots_adjust(wspace=0.05)
    _ax[0].imshow(china)
    _ax[0].set_title('Original Image', size=16)
    _ax[1].imshow(china_recolored)
    _ax[1].set_title('16-color Image', size=16)
    return (china_recolored,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Some detail is certainly lost in the rightmost panel, but the overall image is still easily recognizable.
        In terms of the bytes required to store the raw data, the image on the right achieves a compression factor of around 1 million!
        Now, this kind of approach is not going to match the fidelity of purpose-built image compression schemes like JPEG, but the example shows the power of thinking outside of the box with unsupervised methods like *k*-means.
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
