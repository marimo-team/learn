import marimo

__generated_with = "0.10.15"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Application: A Face Detection Pipeline
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        This part of the book has explored a number of the central concepts and algorithms of machine learning.
        But moving from these concepts to a real-world application can be a challenge.
        Real-world datasets are noisy and heterogeneous; they may have missing features, and data may be in a form that is difficult to map to a clean `[n_samples, n_features]` matrix.
        Before applying any of the methods discussed here, you must first extract these features from your data: there is no formula for how to do this that applies across all domains, and thus this is where you as a data scientist must exercise your own intuition and expertise.

        One interesting and compelling application of machine learning is to images, and we have already seen a few examples of this where pixel-level features are used for classification.
        Again, the real world data is rarely so uniform, and simple pixels will not be suitable: this has led to a large literature on *feature extraction* methods for image data (see [Feature Engineering](05.04-Feature-Engineering.ipynb)).

        In this chapter we will take a look at one such feature extraction technique: the [histogram of oriented gradients (HOG)](https://en.wikipedia.org/wiki/Histogram_of_oriented_gradients), which transforms image pixels into a vector representation that is sensitive to broadly informative image features regardless of confounding factors like illumination.
        We will use these features to develop a simple face detection pipeline, using machine learning algorithms and concepts we've seen throughout this part of the book. 

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
        ## HOG Features

        HOG is a straightforward feature extraction procedure that was developed in the context of identifying pedestrians within images.
        It involves the following steps:

        1. Optionally prenormalize the images. This leads to features that resist dependence on variations in illumination.
        2. Convolve the image with two filters that are sensitive to horizontal and vertical brightness gradients. These capture edge, contour, and texture information.
        3. Subdivide the image into cells of a predetermined size, and compute a histogram of the gradient orientations within each cell.
        4. Normalize the histograms in each cell by comparing to the block of neighboring cells. This further suppresses the effect of illumination across the image.
        5. Construct a one-dimensional feature vector from the information in each cell.

        A fast HOG extractor is built into the Scikit-Image project, and we can try it out relatively quickly and visualize the oriented gradients within each cell (see the following figure):
        """
    )
    return


@app.cell
def _(plt):
    from skimage import data, color, feature
    import skimage.data
    image = color.rgb2gray(data.chelsea())
    hog_vec, hog_vis = feature.hog(image, visualize=True)
    _fig, _ax = plt.subplots(1, 2, figsize=(12, 6), subplot_kw=dict(xticks=[], yticks=[]))
    _ax[0].imshow(image, cmap='gray')
    _ax[0].set_title('input image')
    _ax[1].imshow(hog_vis)
    _ax[1].set_title('visualization of HOG features')
    return color, data, feature, hog_vec, hog_vis, image, skimage


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## HOG in Action: A Simple Face Detector

        Using these HOG features, we can build up a simple facial detection algorithm with any Scikit-Learn estimator; here we will use a linear support vector machine (refer back to [In-Depth: Support Vector Machines](05.07-Support-Vector-Machines.ipynb) if you need a refresher on this).
        The steps are as follows:

        1. Obtain a set of image thumbnails of faces to constitute "positive" training samples.
        2. Obtain a set of image thumbnails of non-faces to constitute "negative" training samples.
        3. Extract HOG features from these training samples.
        4. Train a linear SVM classifier on these samples.
        5. For an "unknown" image, pass a sliding window across the image, using the model to evaluate whether that window contains a face or not.
        6. If detections overlap, combine them into a single window.

        Let's go through these steps and try it out.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### 1. Obtain a Set of Positive Training Samples

        We'll start by finding some positive training samples that show a variety of faces.
        We have one easy set of data to work with—the Labeled Faces in the Wild dataset, which can be downloaded by Scikit-Learn:
        """
    )
    return


@app.cell
def _():
    from sklearn.datasets import fetch_lfw_people
    faces = fetch_lfw_people()
    positive_patches = faces.images
    positive_patches.shape
    return faces, fetch_lfw_people, positive_patches


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        This gives us a sample of 13,000 face images to use for training.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### 2. Obtain a Set of Negative Training Samples

        Next we need a set of similarly sized thumbnails that *do not* have a face in them.
        One way to obtain this is to take any corpus of input images, and extract thumbnails from them at a variety of scales.
        Here we'll use some of the images shipped with Scikit-Image, along with Scikit-Learn's `PatchExtractor`:
        """
    )
    return


@app.cell
def _(data):
    data.camera().shape
    return


@app.cell
def _(color, data_1):
    from skimage import data, transform
    imgs_to_use = ['camera', 'text', 'coins', 'moon', 'page', 'clock', 'immunohistochemistry', 'chelsea', 'coffee', 'hubble_deep_field']
    raw_images = (getattr(data_1, name)() for name in imgs_to_use)
    images = [color.rgb2gray(image) if image.ndim == 3 else image for image in raw_images]
    return data, images, imgs_to_use, raw_images, transform


@app.cell
def _(images, np, positive_patches, transform):
    from sklearn.feature_extraction.image import PatchExtractor

    def extract_patches(img, N, scale=1.0, patch_size=positive_patches[0].shape):
        extracted_patch_size = tuple((scale * np.array(patch_size)).astype(int))
        extractor = PatchExtractor(patch_size=extracted_patch_size,
                                   max_patches=N, random_state=0)
        patches = extractor.transform(img[np.newaxis])
        if scale != 1:
            patches = np.array([transform.resize(patch, patch_size)
                                for patch in patches])
        return patches

    negative_patches = np.vstack([extract_patches(im, 1000, scale)
                                  for im in images for scale in [0.5, 1.0, 2.0]])
    negative_patches.shape
    return PatchExtractor, extract_patches, negative_patches


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We now have 30,000 suitable image patches that do not contain faces.
        Let's visualize a few of them to get an idea of what they look like (see the following figure):
        """
    )
    return


@app.cell
def _(negative_patches, plt):
    _fig, _ax = plt.subplots(6, 10)
    for _i, axi in enumerate(_ax.flat):
        axi.imshow(negative_patches[500 * _i], cmap='gray')
        axi.axis('off')
    return (axi,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Our hope is that these will sufficiently cover the space of "non-faces" that our algorithm is likely to see.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### 3. Combine Sets and Extract HOG Features

        Now that we have these positive samples and negative samples, we can combine them and compute HOG features.
        This step takes a little while, because it involves a nontrivial computation for each image:
        """
    )
    return


@app.cell
def _(feature, negative_patches, np, positive_patches):
    from itertools import chain
    X_train = np.array([feature.hog(im)
                        for im in chain(positive_patches,
                                        negative_patches)])
    y_train = np.zeros(X_train.shape[0])
    y_train[:positive_patches.shape[0]] = 1
    return X_train, chain, y_train


@app.cell
def _(X_train):
    X_train.shape
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We are left with 43,000 training samples in 1,215 dimensions, and we now have our data in a form that we can feed into Scikit-Learn!
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### 4. Train a Support Vector Machine

        Next we use the tools we have been exploring here to create a classifier of thumbnail patches.
        For such a high-dimensional binary classification task, a linear support vector machine is a good choice.
        We will use Scikit-Learn's `LinearSVC`, because in comparison to `SVC` it often has better scaling for a large number of samples.

        First, though, let's use a simple Gaussian naive Bayes estimator to get a quick baseline:
        """
    )
    return


@app.cell
def _(X_train, y_train):
    from sklearn.naive_bayes import GaussianNB
    from sklearn.model_selection import cross_val_score

    cross_val_score(GaussianNB(), X_train, y_train)
    return GaussianNB, cross_val_score


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We see that on our training data, even a simple naive Bayes algorithm gets us upwards of 95% accuracy.
        Let's try the support vector machine, with a grid search over a few choices of the `C` parameter:
        """
    )
    return


@app.cell
def _(X_train, y_train):
    from sklearn.svm import LinearSVC
    from sklearn.model_selection import GridSearchCV
    grid = GridSearchCV(LinearSVC(), {'C': [1.0, 2.0, 4.0, 8.0]})
    grid.fit(X_train, y_train)
    grid.best_score_
    return GridSearchCV, LinearSVC, grid


@app.cell
def _(grid):
    grid.best_params_
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        This pushes us up to near 99% accuracy. Let's take the best estimator and retrain it on the full dataset:
        """
    )
    return


@app.cell
def _(X_train, grid, y_train):
    model = grid.best_estimator_
    model.fit(X_train, y_train)
    return (model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### 5. Find Faces in a New Image

        Now that we have this model in place, let's grab a new image and see how the model does.
        We will use one portion of the astronaut image shown in the following figure for simplicity (see discussion of this in the following section, and run a sliding window over it and evaluate each patch:
        """
    )
    return


@app.cell
def _(plt, skimage):
    test_image = skimage.data.astronaut()
    test_image = skimage.color.rgb2gray(test_image)
    test_image = skimage.transform.rescale(test_image, 0.5)
    test_image = test_image[:160, 40:180]

    plt.imshow(test_image, cmap='gray')
    plt.axis('off');
    return (test_image,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Next, let's create a window that iterates over patches of this image, and compute HOG features for each patch:
        """
    )
    return


@app.cell
def _(feature, np, positive_patches, test_image, transform):
    def sliding_window(img, patch_size=positive_patches[0].shape, istep=2, jstep=2, scale=1.0):
        Ni, Nj = (int(scale * s) for s in patch_size)
        for _i in range(0, img.shape[0] - Ni, istep):
            for j in range(0, img.shape[1] - Ni, jstep):
                patch = img[_i:_i + Ni, j:j + Nj]
                if scale != 1:
                    patch = transform.resize(patch, patch_size)
                yield ((_i, j), patch)
    indices, patches = zip(*sliding_window(test_image))
    patches_hog = np.array([feature.hog(patch) for patch in patches])
    patches_hog.shape
    return indices, patches, patches_hog, sliding_window


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Finally, we can take these HOG-featured patches and use our model to evaluate whether each patch contains a face:
        """
    )
    return


@app.cell
def _(model, patches_hog):
    labels = model.predict(patches_hog)
    labels.sum()
    return (labels,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We see that out of nearly 2,000 patches, we have found 48 detections.
        Let's use the information we have about these patches to show where they lie on our test image, drawing them as rectangles (see the following figure):
        """
    )
    return


@app.cell
def _(indices, labels, np, plt, positive_patches, test_image):
    _fig, _ax = plt.subplots()
    _ax.imshow(test_image, cmap='gray')
    _ax.axis('off')
    Ni, Nj = positive_patches[0].shape
    indices_1 = np.array(indices)
    for _i, j in indices_1[labels == 1]:
        _ax.add_patch(plt.Rectangle((j, _i), Nj, Ni, edgecolor='red', alpha=0.3, lw=2, facecolor='none'))
    return Ni, Nj, indices_1, j


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        All of the detected patches overlap and found the face in the image!
        Not bad for a few lines of Python.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Caveats and Improvements

        If you dig a bit deeper into the preceding code and examples, you'll see that we still have a bit of work to do before we can claim a production-ready face detector.
        There are several issues with what we've done, and several improvements that could be made. In particular:

        **Our training set, especially for negative features, is not very complete**

        The central issue is that there are many face-like textures that are not in the training set, and so our current model is very prone to false positives.
        You can see this if you try out the algorithm on the *full* astronaut image: the current model leads to many false detections in other regions of the image.

        We might imagine addressing this by adding a wider variety of images to the negative training set, and this would probably yield some improvement.
        Another option would be to use a more directed approach, such as *hard negative mining*, where we take a new set of images that our classifier has not seen, find all the patches representing false positives, and explicitly add them as negative instances in the training set before retraining the classifier.

        **Our current pipeline searches only at one scale**

        As currently written, our algorithm will miss faces that are not approximately 62 × 47 pixels.
        This can be straightforwardly addressed by using sliding windows of a variety of sizes, and resizing each patch using `skimage.transform.resize` before feeding it into the model.
        In fact, the `sliding_window` utility used here is already built with this in mind.

        **We should combine overlapped detection patches**

        For a production-ready pipeline, we would prefer not to have 30 detections of the same face, but to somehow reduce overlapping groups of detections down to a single detection.
        This could be done via an unsupervised clustering approach (mean shift clustering is one good candidate for this), or via a procedural approach such as *non-maximum suppression*, an algorithm common in machine vision.

        **The pipeline should be streamlined**

        Once we address the preceding issues, it would also be nice to create a more streamlined pipeline for ingesting training images and predicting sliding-window outputs.
        This is where Python as a data science tool really shines: with a bit of work, we could take our prototype code and package it with a well-designed object-oriented API that gives the user the ability to use it easily.
        I will leave this as a proverbial "exercise for the reader."

        **More recent advances: deep learning**

        Finally, I should add that in machine learning contexts, HOG and other procedural feature extraction methods are not always used.
        Instead, many modern object detection pipelines use variants of deep neural networks (often referred to as *deep learning*): one way to think of neural networks is as estimators that determine optimal feature extraction strategies from the data, rather than relying on the intuition of the user.

        Though the field has produced fantastic results in recent years, deep learning is not all that conceptually different from the machine learning models explored in the previous chapters.
        The main advance is the ability to utilize modern computing hardware (often large clusters of powerful machines) to train much more flexible models on much larger corpuses of training data.
        But though the scale differs, the end goal is very much the same the same: building models from data.

        If you're interested in going further, the list of references in the following section should provide a useful place to start!
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
