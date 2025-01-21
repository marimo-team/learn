import marimo

__generated_with = "0.10.15"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # In Depth: Decision Trees and Random Forests
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Previously we have looked in depth at a simple generative classifier (naive Bayes; see [In Depth: Naive Bayes Classification](05.05-Naive-Bayes.ipynb)) and a powerful discriminative classifier (support vector machines; see [In-Depth: Support Vector Machines](05.07-Support-Vector-Machines.ipynb)).
        Here we'll take a look at another powerful algorithm: a nonparametric algorithm called *random forests*.
        Random forests are an example of an *ensemble* method, meaning one that relies on aggregating the results of a set of simpler estimators.
        The somewhat surprising result with such ensemble methods is that the sum can be greater than the parts: that is, the predictive accuracy of a majority vote among a number of estimators can end up being better than that of any of the individual estimators doing the voting!
        We will see examples of this in the following sections.

        We begin with the standard imports:
        """
    )
    return


@app.cell
def _():
    # "%matplotlib inline\nimport numpy as np\nimport matplotlib.pyplot as plt\nplt.style.use('seaborn-whitegrid')" command supported automatically in marimo
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Motivating Random Forests: Decision Trees
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Random forests are an example of an ensemble learner built on decision trees.
        For this reason, we'll start by discussing decision trees themselves.

        Decision trees are extremely intuitive ways to classify or label objects: you simply ask a series of questions designed to zero in on the classification.
        For example, if you wanted to build a decision tree to classify animals you come across while on a hike, you might construct the one shown in the following figure.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ![](images/05.08-decision-tree.png)
        [figure source in Appendix](https://github.com/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/06.00-Figure-Code.ipynb#Decision-Tree-Example)
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The binary splitting makes this extremely efficient: in a well-constructed tree, each question will cut the number of options by approximately half, very quickly narrowing the options even among a large number of classes.
        The trick, of course, comes in deciding which questions to ask at each step.
        In machine learning implementations of decision trees, the questions generally take the form of axis-aligned splits in the data: that is, each node in the tree splits the data into two groups using a cutoff value within one of the features.
        Let's now look at an example of this.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Creating a Decision Tree

        Consider the following two-dimensional data, which has one of four class labels (see the following figure):
        """
    )
    return


@app.cell
def _(plt):
    from sklearn.datasets import make_blobs

    X, y = make_blobs(n_samples=300, centers=4,
                      random_state=0, cluster_std=1.0)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='rainbow');
    return X, make_blobs, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        A simple decision tree built on this data will iteratively split the data along one or the other axis according to some quantitative criterion, and at each level assign the label of the new region according to a majority vote of points within it.
        The following figure presents a visualization of the first four levels of a decision tree classifier for this data.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ![](images/05.08-decision-tree-levels.png)
        [figure source in Appendix](https://github.com/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/06.00-Figure-Code.ipynb#Decision-Tree-Levels)
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Notice that after the first split, every point in the upper branch remains unchanged, so there is no need to further subdivide this branch.
        Except for nodes that contain all of one color, at each level *every* region is again split along one of the two features.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        This process of fitting a decision tree to our data can be done in Scikit-Learn with the ``DecisionTreeClassifier`` estimator:
        """
    )
    return


@app.cell
def _(X, y):
    from sklearn.tree import DecisionTreeClassifier
    _tree = DecisionTreeClassifier().fit(X, y)
    return (DecisionTreeClassifier,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Let's write a utility function to help us visualize the output of the classifier:
        """
    )
    return


@app.cell
def _(np, plt):
    def visualize_classifier(model, X, y, ax=None, cmap='rainbow'):
        ax = ax or plt.gca()
        
        # Plot the training points
        ax.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=cmap,
                   clim=(y.min(), y.max()), zorder=3)
        ax.axis('tight')
        ax.axis('off')
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        
        # fit the estimator
        model.fit(X, y)
        xx, yy = np.meshgrid(np.linspace(*xlim, num=200),
                             np.linspace(*ylim, num=200))
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

        # Create a color plot with the results
        n_classes = len(np.unique(y))
        contours = ax.contourf(xx, yy, Z, alpha=0.3,
                               levels=np.arange(n_classes + 1) - 0.5,
                               cmap=cmap, zorder=1)

        ax.set(xlim=xlim, ylim=ylim)
    return (visualize_classifier,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Now we can examine what the decision tree classification looks like (see the following figure):
        """
    )
    return


@app.cell
def _(DecisionTreeClassifier, X, visualize_classifier, y):
    visualize_classifier(DecisionTreeClassifier(), X, y)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        If you're running this notebook live, you can use the helper script included in the online [appendix](https://github.com/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/06.00-Figure-Code.ipynb#Helper-Code) to bring up an interactive visualization of the decision tree building process:
        """
    )
    return


@app.cell
def _(X, y):
    # helpers_05_08 is found in the online appendix
    import helpers_05_08
    helpers_05_08.plot_tree_interactive(X, y);
    return (helpers_05_08,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Notice that as the depth increases, we tend to get very strangely shaped classification regions; for example, at a depth of five, there is a tall and skinny purple region between the yellow and blue regions.
        It's clear that this is less a result of the true, intrinsic data distribution, and more a result of the particular sampling or noise properties of the data.
        That is, this decision tree, even at only five levels deep, is clearly overfitting our data.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Decision Trees and Overfitting

        Such overfitting turns out to be a general property of decision trees: it is very easy to go too deep in the tree, and thus to fit details of the particular data rather than the overall properties of the distributions it is drawn from.
        Another way to see this overfitting is to look at models trained on different subsets of the data—for example, in this figure we train two different trees, each on half of the original data.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ![](images/05.08-decision-tree-overfitting.png)
        [figure source in Appendix](https://github.com/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/06.00-Figure-Code.ipynb#Decision-Tree-Overfitting)
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        It is clear that in some places the two trees produce consistent results (e.g., in the four corners), while in other places the two trees give very different classifications (e.g., in the regions between any two clusters).
        The key observation is that the inconsistencies tend to happen where the classification is less certain, and thus by using information from *both* of these trees, we might come up with a better result!
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        If you are running this notebook live, the following function will allow you to interactively display the fits of trees trained on a random subset of the data:
        """
    )
    return


@app.cell
def _(X, helpers_05_08, y):
    # helpers_05_08 is found in the online appendix
    helpers_05_08.randomized_tree_interactive(X, y)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Just as using information from two trees improves our results, we might expect that using information from many trees would improve our results even further.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Ensembles of Estimators: Random Forests

        This notion—that multiple overfitting estimators can be combined to reduce the effect of this overfitting—is what underlies an ensemble method called *bagging*.
        Bagging makes use of an ensemble (a grab bag, perhaps) of parallel estimators, each of which overfits the data, and averages the results to find a better classification.
        An ensemble of randomized decision trees is known as a *random forest*.

        This type of bagging classification can be done manually using Scikit-Learn's `BaggingClassifier` meta-estimator, as shown here (see the following figure):
        """
    )
    return


@app.cell
def _(DecisionTreeClassifier, X, visualize_classifier, y):
    from sklearn.ensemble import BaggingClassifier
    _tree = DecisionTreeClassifier()
    bag = BaggingClassifier(_tree, n_estimators=100, max_samples=0.8, random_state=1)
    bag.fit(X, y)
    visualize_classifier(bag, X, y)
    return BaggingClassifier, bag


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        In this example, we have randomized the data by fitting each estimator with a random subset of 80% of the training points.
        In practice, decision trees are more effectively randomized by injecting some stochasticity in how the splits are chosen: this way all the data contributes to the fit each time, but the results of the fit still have the desired randomness.
        For example, when determining which feature to split on, the randomized tree might select from among the top several features.
        You can read more technical details about these randomization strategies in the [Scikit-Learn documentation](http://scikit-learn.org/stable/modules/ensemble.html#forest) and references within.

        In Scikit-Learn, such an optimized ensemble of randomized decision trees is implemented in the `RandomForestClassifier` estimator, which takes care of all the randomization automatically.
        All you need to do is select a number of estimators, and it will very quickly—in parallel, if desired—fit the ensemble of trees (see the following figure):
        """
    )
    return


@app.cell
def _(X, visualize_classifier, y):
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(n_estimators=100, random_state=0)
    visualize_classifier(model, X, y);
    return RandomForestClassifier, model


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We see that by averaging over 100 randomly perturbed models, we end up with an overall model that is much closer to our intuition about how the parameter space should be split.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Random Forest Regression

        In the previous section we considered random forests within the context of classification.
        Random forests can also be made to work in the case of regression (that is, with continuous rather than categorical variables). The estimator to use for this is the `RandomForestRegressor`, and the syntax is very similar to what we saw earlier.

        Consider the following data, drawn from the combination of a fast and slow oscillation (see the following figure):
        """
    )
    return


@app.cell
def _(np, plt):
    rng = np.random.RandomState(42)
    x = 10 * rng.rand(200)

    def model_1(x, sigma=0.3):
        fast_oscillation = np.sin(5 * x)
        slow_oscillation = np.sin(0.5 * x)
        noise = sigma * rng.randn(len(x))
        return slow_oscillation + fast_oscillation + noise
    y_1 = model_1(x)
    plt.errorbar(x, y_1, 0.3, fmt='o')
    return model_1, rng, x, y_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Using the random forest regressor, we can find the best-fit curve as follows (see the following figure):
        """
    )
    return


@app.cell
def _(model_1, np, plt, x, y_1):
    from sklearn.ensemble import RandomForestRegressor
    forest = RandomForestRegressor(200)
    forest.fit(x[:, None], y_1)
    xfit = np.linspace(0, 10, 1000)
    yfit = forest.predict(xfit[:, None])
    ytrue = model_1(xfit, sigma=0)
    plt.errorbar(x, y_1, 0.3, fmt='o', alpha=0.5)
    plt.plot(xfit, yfit, '-r')
    plt.plot(xfit, ytrue, '-k', alpha=0.5)
    return RandomForestRegressor, forest, xfit, yfit, ytrue


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Here the true model is shown in the smooth gray curve, while the random forest model is shown by the jagged red curve.
        The nonparametric random forest model is flexible enough to fit the multiperiod data, without us needing to specifying a multi-period model!
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Example: Random Forest for Classifying Digits

        In Chapter 38 we worked through an example using the digits dataset included with Scikit-Learn.
        Let's use that again here to see how the random forest classifier can be applied in this context:
        """
    )
    return


@app.cell
def _():
    from sklearn.datasets import load_digits
    digits = load_digits()
    digits.keys()
    return digits, load_digits


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        To remind us what we're looking at, we'll visualize the first few data points (see the following figure):
        """
    )
    return


@app.cell
def _(digits, plt):
    # set up the figure
    fig = plt.figure(figsize=(6, 6))  # figure size in inches
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    # plot the digits: each image is 8x8 pixels
    for i in range(64):
        ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
        ax.imshow(digits.images[i], cmap=plt.cm.binary, interpolation='nearest')
        
        # label the image with the target value
        ax.text(0, 7, str(digits.target[i]))
    return ax, fig, i


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We can classify the digits using a random forest as follows:
        """
    )
    return


@app.cell
def _(RandomForestClassifier, digits):
    from sklearn.model_selection import train_test_split
    Xtrain, Xtest, ytrain, ytest = train_test_split(digits.data, digits.target, random_state=0)
    model_2 = RandomForestClassifier(n_estimators=1000)
    model_2.fit(Xtrain, ytrain)
    ypred = model_2.predict(Xtest)
    return Xtest, Xtrain, model_2, train_test_split, ypred, ytest, ytrain


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Let's look at the classification report for this classifier:
        """
    )
    return


@app.cell
def _(ypred, ytest):
    from sklearn import metrics
    print(metrics.classification_report(ypred, ytest))
    return (metrics,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        And for good measure, plot the confusion matrix (see the following figure):
        """
    )
    return


@app.cell
def _(plt, ypred, ytest):
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    mat = confusion_matrix(ytest, ypred)
    sns.heatmap(mat.T, square=True, annot=True, fmt='d',
                cbar=False, cmap='Blues')
    plt.xlabel('true label')
    plt.ylabel('predicted label');
    return confusion_matrix, mat, sns


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We find that a simple, untuned random forest results in a quite accurate classification of the digits data.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Summary

        This chapter provided a brief introduction to the concept of ensemble estimators, and in particular the random forest, an ensemble of randomized decision trees.
        Random forests are a powerful method with several advantages:

        - Both training and prediction are very fast, because of the simplicity of the underlying decision trees. In addition, both tasks can be straightforwardly parallelized, because the individual trees are entirely independent entities.
        - The multiple trees allow for a probabilistic classification: a majority vote among estimators gives an estimate of the probability (accessed in Scikit-Learn with the `predict_proba` method).
        - The nonparametric model is extremely flexible and can thus perform well on tasks that are underfit by other estimators.

        A primary disadvantage of random forests is that the results are not easily interpretable: that is, if you would like to draw conclusions about the *meaning* of the classification model, random forests may not be the best choice.
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
