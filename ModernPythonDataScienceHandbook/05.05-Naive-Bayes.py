import marimo

__generated_with = "0.10.15"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # In Depth: Naive Bayes Classification
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The previous four chapters have given a general overview of the concepts of machine learning.
        In this chapter and the ones that follow, we will be taking a
        closer look first at four algorithms for  supervised learning,
        and then at four algorithms for unsupervised learning.
        We start here with our first supervised method, naive Bayes classification.

        Naive Bayes models are a group of extremely fast and simple classification algorithms that are often suitable for very high-dimensional datasets.
        Because they are so fast and have so few tunable parameters, they end up being useful as a quick-and-dirty baseline for a classification problem.
        This chapter will provide an intuitive explanation of how naive Bayes classifiers work, followed by a few examples of them in action on some datasets.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Bayesian Classification

        Naive Bayes classifiers are built on Bayesian classification methods.
        These rely on Bayes's theorem, which is an equation describing the relationship of conditional probabilities of statistical quantities.
        In Bayesian classification, we're interested in finding the probability of a label $L$ given some observed features, which we can write as $P(L~|~{\rm features})$.
        Bayes's theorem tells us how to express this in terms of quantities we can compute more directly:

        $$
        P(L~|~{\rm features}) = \frac{P({\rm features}~|~L)P(L)}{P({\rm features})}
        $$

        If we are trying to decide between two labels—let's call them $L_1$ and $L_2$—then one way to make this decision is to compute the ratio of the posterior probabilities for each label:

        $$
        \frac{P(L_1~|~{\rm features})}{P(L_2~|~{\rm features})} = \frac{P({\rm features}~|~L_1)}{P({\rm features}~|~L_2)}\frac{P(L_1)}{P(L_2)}
        $$

        All we need now is some model by which we can compute $P({\rm features}~|~L_i)$ for each label.
        Such a model is called a *generative model* because it specifies the hypothetical random process that generates the data.
        Specifying this generative model for each label is the main piece of the training of such a Bayesian classifier.
        The general version of such a training step is a very difficult task, but we can make it simpler through the use of some simplifying assumptions about the form of this model.

        This is where the "naive" in "naive Bayes" comes in: if we make very naive assumptions about the generative model for each label, we can find a rough approximation of the generative model for each class, and then proceed with the Bayesian classification.
        Different types of naive Bayes classifiers rest on different naive assumptions about the data, and we will examine a few of these in the following sections.

        We begin with the standard imports:
        """
    )
    return


@app.cell
def _():
    # "%matplotlib inline\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nplt.style.use('seaborn-whitegrid')" command supported automatically in marimo
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Gaussian Naive Bayes

        Perhaps the easiest naive Bayes classifier to understand is Gaussian naive Bayes.
        With this classifier, the assumption is that *data from each label is drawn from a simple Gaussian distribution*.
        Imagine that we have the following data, shown in Figure 41-1:
        """
    )
    return


@app.cell
def _(plt):
    from sklearn.datasets import make_blobs
    X, y = make_blobs(100, 2, centers=2, random_state=2, cluster_std=1.5)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='RdBu');
    return X, make_blobs, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The simplest Gaussian model is to assume that the data is described by a Gaussian distribution with no covariance between dimensions.
        This model can be fit by computing the mean and standard deviation of the points within each label, which is all we need to define such a distribution.
        The result of this naive Gaussian assumption is shown in the following figure:
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ![(run code in Appendix to generate image)](images/05.05-gaussian-NB.png)
        [figure source in Appendix](https://github.com/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/06.00-Figure-Code.ipynb#Gaussian-Naive-Bayes)
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The ellipses here represent the Gaussian generative model for each label, with larger probability toward the center of the ellipses.
        With this generative model in place for each class, we have a simple recipe to compute the likelihood $P({\rm features}~|~L_1)$ for any data point, and thus we can quickly compute the posterior ratio and determine which label is the most probable for a given point.

        This procedure is implemented in Scikit-Learn's `sklearn.naive_bayes.GaussianNB` estimator:
        """
    )
    return


@app.cell
def _(X, y):
    from sklearn.naive_bayes import GaussianNB
    model = GaussianNB()
    model.fit(X, y);
    return GaussianNB, model


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Let's generate some new data and predict the label:
        """
    )
    return


@app.cell
def _(model, np):
    rng = np.random.RandomState(0)
    Xnew = [-6, -14] + [14, 18] * rng.rand(2000, 2)
    ynew = model.predict(Xnew)
    return Xnew, rng, ynew


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Now we can plot this new data to get an idea of where the decision boundary is (see the following figure):
        """
    )
    return


@app.cell
def _(X, Xnew, plt, y, ynew):
    plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='RdBu')
    lim = plt.axis()
    plt.scatter(Xnew[:, 0], Xnew[:, 1], c=ynew, s=20, cmap='RdBu', alpha=0.1)
    plt.axis(lim);
    return (lim,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We see a slightly curved boundary in the classifications—in general, the boundary produced by a Gaussian naive Bayes model will be quadratic.

        A nice aspect of this Bayesian formalism is that it naturally allows for probabilistic classification, which we can compute using the `predict_proba` method:
        """
    )
    return


@app.cell
def _(Xnew, model):
    yprob = model.predict_proba(Xnew)
    yprob[-8:].round(2)
    return (yprob,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The columns give the posterior probabilities of the first and second labels, respectively.
        If you are looking for estimates of uncertainty in your classification, Bayesian approaches like this can be a good place to start.

        Of course, the final classification will only be as good as the model assumptions that lead to it, which is why Gaussian naive Bayes often does not produce very good results.
        Still, in many cases—especially as the number of features becomes large—this assumption is not detrimental enough to prevent Gaussian naive Bayes from being a reliable method.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Multinomial Naive Bayes

        The Gaussian assumption just described is by no means the only simple assumption that could be used to specify the generative distribution for each label.
        Another useful example is multinomial naive Bayes, where the features are assumed to be generated from a simple multinomial distribution.
        The multinomial distribution describes the probability of observing counts among a number of categories, and thus multinomial naive Bayes is most appropriate for features that represent counts or count rates.

        The idea is precisely the same as before, except that instead of modeling the data distribution with the best-fit Gaussian, we model it with a best-fit multinomial distribution.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Example: Classifying Text

        One place where multinomial naive Bayes is often used is in text classification, where the features are related to word counts or frequencies within the documents to be classified.
        We discussed the extraction of such features from text in [Feature Engineering](05.04-Feature-Engineering.ipynb); here we will use the sparse word count features from the 20 Newsgroups corpus made available through Scikit-Learn to show how we might classify these short documents into categories.

        Let's download the data and take a look at the target names:
        """
    )
    return


@app.cell
def _():
    from sklearn.datasets import fetch_20newsgroups

    data = fetch_20newsgroups()
    data.target_names
    return data, fetch_20newsgroups


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        For simplicity here, we will select just a few of these categories and download the training and testing sets:
        """
    )
    return


@app.cell
def _(fetch_20newsgroups):
    categories = ['talk.religion.misc', 'soc.religion.christian',
                  'sci.space', 'comp.graphics']
    train = fetch_20newsgroups(subset='train', categories=categories)
    test = fetch_20newsgroups(subset='test', categories=categories)
    return categories, test, train


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Here is a representative entry from the data:
        """
    )
    return


@app.cell
def _(train):
    print(train.data[5][48:])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        In order to use this data for machine learning, we need to be able to convert the content of each string into a vector of numbers.
        For this we will use the TF-IDF vectorizer (introduced in [Feature Engineering](05.04-Feature-Engineering.ipynb)), and create a pipeline that attaches it to a multinomial naive Bayes classifier:
        """
    )
    return


@app.cell
def _():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import make_pipeline
    model_1 = make_pipeline(TfidfVectorizer(), MultinomialNB())
    return MultinomialNB, TfidfVectorizer, make_pipeline, model_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        With this pipeline, we can apply the model to the training data and predict labels for the test data:
        """
    )
    return


@app.cell
def _(model_1, test, train):
    model_1.fit(train.data, train.target)
    labels = model_1.predict(test.data)
    return (labels,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Now that we have predicted the labels for the test data, we can evaluate them to learn about the performance of the estimator.
        For example, let's take a look at the confusion matrix between the true and predicted labels for the test data (see the following figure):
        """
    )
    return


@app.cell
def _(labels, plt, sns, test, train):
    from sklearn.metrics import confusion_matrix
    mat = confusion_matrix(test.target, labels)
    sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
                xticklabels=train.target_names, yticklabels=train.target_names,
                cmap='Blues')
    plt.xlabel('true label')
    plt.ylabel('predicted label');
    return confusion_matrix, mat


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Evidently, even this very simple classifier can successfully separate space discussions from computer discussions, but it gets confused between discussions about religion and discussions about Christianity.
        This is perhaps to be expected!

        The cool thing here is that we now have the tools to determine the category for *any* string, using the `predict` method of this pipeline.
        Here's a utility function that will return the prediction for a single string:
        """
    )
    return


@app.cell
def _(model_1, train):
    def predict_category(s, train=train, model=model_1):
        pred = model.predict([s])
        return train.target_names[pred[0]]
    return (predict_category,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Let's try it out:
        """
    )
    return


@app.cell
def _(predict_category):
    predict_category('sending a payload to the ISS')
    return


@app.cell
def _(predict_category):
    predict_category('discussing the existence of God')
    return


@app.cell
def _(predict_category):
    predict_category('determining the screen resolution')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Remember that this is nothing more sophisticated than a simple probability model for the (weighted) frequency of each word in the string; nevertheless, the result is striking.
        Even a very naive algorithm, when used carefully and trained on a large set of high-dimensional data, can be surprisingly effective.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## When to Use Naive Bayes

        Because naive Bayes classifiers make such stringent assumptions about data, they will generally not perform as well as more complicated models.
        That said, they have several advantages:

        - They are fast for both training and prediction.
        - They provide straightforward probabilistic prediction.
        - They are often easily interpretable.
        - They have few (if any) tunable parameters.

        These advantages mean a naive Bayes classifier is often a good choice as an initial baseline classification.
        If it performs suitably, then congratulations: you have a very fast, very interpretable classifier for your problem.
        If it does not perform well, then you can begin exploring more sophisticated models, with some baseline knowledge of how well they should perform.

        Naive Bayes classifiers tend to perform especially well in the following situations:

        - When the naive assumptions actually match the data (very rare in practice)
        - For very well-separated categories, when model complexity is less important
        - For very high-dimensional data, when model complexity is less important

        The last two points seem distinct, but they actually are related: as the dimensionality of a dataset grows, it is much less likely for any two points to be found close together (after all, they must be close in *every single dimension* to be close overall).
        This means that clusters in high dimensions tend to be more separated, on average, than clusters in low dimensions, assuming the new dimensions actually add information.
        For this reason, simplistic classifiers like the ones discussed here tend to work as well or better than more complicated classifiers as the dimensionality grows: once you have enough data, even a simple model can be very powerful.
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
