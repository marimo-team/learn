import marimo

__generated_with = "0.10.15"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Appendix: Figure Code
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Many of the figures used throughout this text are created in-place by code that appears in print.
        In a few cases, however, the required code is long enough (or not immediately relevant enough) that we instead put it here for reference.
        """
    )
    return


@app.cell
def _():
    # '%matplotlib inline\nimport matplotlib.pyplot as plt\nimport numpy as np\nimport seaborn as sns' command supported automatically in marimo
    return


@app.cell
def _():
    import os
    if not os.path.exists('figures'):
        os.makedirs('figures')
    return (os,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Broadcasting

        [Figure Context](02.05-Computation-on-arrays-broadcasting.ipynb#Introducing-Broadcasting)
        """
    )
    return


@app.cell
def _():
    import numpy as np
    from matplotlib import pyplot as plt
    _fig = plt.figure(figsize=(6, 4.5), facecolor='w')
    _ax = plt.axes([0, 0, 1, 1], xticks=[], yticks=[], frameon=False)

    def draw_cube(ax, xy, size, depth=0.4, edges=None, label=None, label_kwargs=None, **kwargs):
        """draw and label a cube.  edges is a list of numbers between
        1 and 12, specifying which of the 12 cube edges to draw"""
        if edges is None:
            edges = range(1, 13)
        x, y = xy
        if 1 in edges:
            _ax.plot([x, x + size], [y + size, y + size], **kwargs)
        if 2 in edges:
            _ax.plot([x + size, x + size], [y, y + size], **kwargs)
        if 3 in edges:
            _ax.plot([x, x + size], [y, y], **kwargs)
        if 4 in edges:
            _ax.plot([x, x], [y, y + size], **kwargs)
        if 5 in edges:
            _ax.plot([x, x + _depth], [y + size, y + _depth + size], **kwargs)
        if 6 in edges:
            _ax.plot([x + size, x + size + _depth], [y + size, y + _depth + size], **kwargs)
        if 7 in edges:
            _ax.plot([x + size, x + size + _depth], [y, y + _depth], **kwargs)
        if 8 in edges:
            _ax.plot([x, x + _depth], [y, y + _depth], **kwargs)
        if 9 in edges:
            _ax.plot([x + _depth, x + _depth + size], [y + _depth + size, y + _depth + size], **kwargs)
        if 10 in edges:
            _ax.plot([x + _depth + size, x + _depth + size], [y + _depth, y + _depth + size], **kwargs)
        if 11 in edges:
            _ax.plot([x + _depth, x + _depth + size], [y + _depth, y + _depth], **kwargs)
        if 12 in edges:
            _ax.plot([x + _depth, x + _depth], [y + _depth, y + _depth + size], **kwargs)
        if label:
            if label_kwargs is None:
                label_kwargs = {}
            _ax.text(x + 0.5 * size, y + 0.5 * size, label, ha='center', va='center', **label_kwargs)
    solid = dict(c='black', ls='-', lw=1, label_kwargs=dict(color='k'))
    dotted = dict(c='black', ls='-', lw=0.5, alpha=0.5, label_kwargs=dict(color='gray'))
    _depth = 0.3
    draw_cube(_ax, (1, 10), 1, _depth, [1, 2, 3, 4, 5, 6, 9], '0', **solid)
    draw_cube(_ax, (2, 10), 1, _depth, [1, 2, 3, 6, 9], '1', **solid)
    draw_cube(_ax, (3, 10), 1, _depth, [1, 2, 3, 6, 7, 9, 10], '2', **solid)
    draw_cube(_ax, (6, 10), 1, _depth, [1, 2, 3, 4, 5, 6, 7, 9, 10], '5', **solid)
    draw_cube(_ax, (7, 10), 1, _depth, [1, 2, 3, 6, 7, 9, 10, 11], '5', **dotted)
    draw_cube(_ax, (8, 10), 1, _depth, [1, 2, 3, 6, 7, 9, 10, 11], '5', **dotted)
    draw_cube(_ax, (12, 10), 1, _depth, [1, 2, 3, 4, 5, 6, 9], '5', **solid)
    draw_cube(_ax, (13, 10), 1, _depth, [1, 2, 3, 6, 9], '6', **solid)
    draw_cube(_ax, (14, 10), 1, _depth, [1, 2, 3, 6, 7, 9, 10], '7', **solid)
    _ax.text(5, 10.5, '+', size=12, ha='center', va='center')
    _ax.text(10.5, 10.5, '=', size=12, ha='center', va='center')
    _ax.text(1, 11.5, '${\\tt np.arange(3) + 5}$', size=12, ha='left', va='bottom')
    draw_cube(_ax, (1, 7.5), 1, _depth, [1, 2, 3, 4, 5, 6, 9], '1', **solid)
    draw_cube(_ax, (2, 7.5), 1, _depth, [1, 2, 3, 6, 9], '1', **solid)
    draw_cube(_ax, (3, 7.5), 1, _depth, [1, 2, 3, 6, 7, 9, 10], '1', **solid)
    draw_cube(_ax, (1, 6.5), 1, _depth, [2, 3, 4], '1', **solid)
    draw_cube(_ax, (2, 6.5), 1, _depth, [2, 3], '1', **solid)
    draw_cube(_ax, (3, 6.5), 1, _depth, [2, 3, 7, 10], '1', **solid)
    draw_cube(_ax, (1, 5.5), 1, _depth, [2, 3, 4], '1', **solid)
    draw_cube(_ax, (2, 5.5), 1, _depth, [2, 3], '1', **solid)
    draw_cube(_ax, (3, 5.5), 1, _depth, [2, 3, 7, 10], '1', **solid)
    draw_cube(_ax, (6, 7.5), 1, _depth, [1, 2, 3, 4, 5, 6, 9], '0', **solid)
    draw_cube(_ax, (7, 7.5), 1, _depth, [1, 2, 3, 6, 9], '1', **solid)
    draw_cube(_ax, (8, 7.5), 1, _depth, [1, 2, 3, 6, 7, 9, 10], '2', **solid)
    draw_cube(_ax, (6, 6.5), 1, _depth, range(2, 13), '0', **dotted)
    draw_cube(_ax, (7, 6.5), 1, _depth, [2, 3, 6, 7, 9, 10, 11], '1', **dotted)
    draw_cube(_ax, (8, 6.5), 1, _depth, [2, 3, 6, 7, 9, 10, 11], '2', **dotted)
    draw_cube(_ax, (6, 5.5), 1, _depth, [2, 3, 4, 7, 8, 10, 11, 12], '0', **dotted)
    draw_cube(_ax, (7, 5.5), 1, _depth, [2, 3, 7, 10, 11], '1', **dotted)
    draw_cube(_ax, (8, 5.5), 1, _depth, [2, 3, 7, 10, 11], '2', **dotted)
    draw_cube(_ax, (12, 7.5), 1, _depth, [1, 2, 3, 4, 5, 6, 9], '1', **solid)
    draw_cube(_ax, (13, 7.5), 1, _depth, [1, 2, 3, 6, 9], '2', **solid)
    draw_cube(_ax, (14, 7.5), 1, _depth, [1, 2, 3, 6, 7, 9, 10], '3', **solid)
    draw_cube(_ax, (12, 6.5), 1, _depth, [2, 3, 4], '1', **solid)
    draw_cube(_ax, (13, 6.5), 1, _depth, [2, 3], '2', **solid)
    draw_cube(_ax, (14, 6.5), 1, _depth, [2, 3, 7, 10], '3', **solid)
    draw_cube(_ax, (12, 5.5), 1, _depth, [2, 3, 4], '1', **solid)
    draw_cube(_ax, (13, 5.5), 1, _depth, [2, 3], '2', **solid)
    draw_cube(_ax, (14, 5.5), 1, _depth, [2, 3, 7, 10], '3', **solid)
    _ax.text(5, 7.0, '+', size=12, ha='center', va='center')
    _ax.text(10.5, 7.0, '=', size=12, ha='center', va='center')
    _ax.text(1, 9.0, '${\\tt np.ones((3,\\, 3)) + np.arange(3)}$', size=12, ha='left', va='bottom')
    draw_cube(_ax, (1, 3), 1, _depth, [1, 2, 3, 4, 5, 6, 7, 9, 10], '0', **solid)
    draw_cube(_ax, (1, 2), 1, _depth, [2, 3, 4, 7, 10], '1', **solid)
    draw_cube(_ax, (1, 1), 1, _depth, [2, 3, 4, 7, 10], '2', **solid)
    draw_cube(_ax, (2, 3), 1, _depth, [1, 2, 3, 6, 7, 9, 10, 11], '0', **dotted)
    draw_cube(_ax, (2, 2), 1, _depth, [2, 3, 7, 10, 11], '1', **dotted)
    draw_cube(_ax, (2, 1), 1, _depth, [2, 3, 7, 10, 11], '2', **dotted)
    draw_cube(_ax, (3, 3), 1, _depth, [1, 2, 3, 6, 7, 9, 10, 11], '0', **dotted)
    draw_cube(_ax, (3, 2), 1, _depth, [2, 3, 7, 10, 11], '1', **dotted)
    draw_cube(_ax, (3, 1), 1, _depth, [2, 3, 7, 10, 11], '2', **dotted)
    draw_cube(_ax, (6, 3), 1, _depth, [1, 2, 3, 4, 5, 6, 9], '0', **solid)
    draw_cube(_ax, (7, 3), 1, _depth, [1, 2, 3, 6, 9], '1', **solid)
    draw_cube(_ax, (8, 3), 1, _depth, [1, 2, 3, 6, 7, 9, 10], '2', **solid)
    draw_cube(_ax, (6, 2), 1, _depth, range(2, 13), '0', **dotted)
    draw_cube(_ax, (7, 2), 1, _depth, [2, 3, 6, 7, 9, 10, 11], '1', **dotted)
    draw_cube(_ax, (8, 2), 1, _depth, [2, 3, 6, 7, 9, 10, 11], '2', **dotted)
    draw_cube(_ax, (6, 1), 1, _depth, [2, 3, 4, 7, 8, 10, 11, 12], '0', **dotted)
    draw_cube(_ax, (7, 1), 1, _depth, [2, 3, 7, 10, 11], '1', **dotted)
    draw_cube(_ax, (8, 1), 1, _depth, [2, 3, 7, 10, 11], '2', **dotted)
    draw_cube(_ax, (12, 3), 1, _depth, [1, 2, 3, 4, 5, 6, 9], '0', **solid)
    draw_cube(_ax, (13, 3), 1, _depth, [1, 2, 3, 6, 9], '1', **solid)
    draw_cube(_ax, (14, 3), 1, _depth, [1, 2, 3, 6, 7, 9, 10], '2', **solid)
    draw_cube(_ax, (12, 2), 1, _depth, [2, 3, 4], '1', **solid)
    draw_cube(_ax, (13, 2), 1, _depth, [2, 3], '2', **solid)
    draw_cube(_ax, (14, 2), 1, _depth, [2, 3, 7, 10], '3', **solid)
    draw_cube(_ax, (12, 1), 1, _depth, [2, 3, 4], '2', **solid)
    draw_cube(_ax, (13, 1), 1, _depth, [2, 3], '3', **solid)
    draw_cube(_ax, (14, 1), 1, _depth, [2, 3, 7, 10], '4', **solid)
    _ax.text(5, 2.5, '+', size=12, ha='center', va='center')
    _ax.text(10.5, 2.5, '=', size=12, ha='center', va='center')
    _ax.text(1, 4.5, '${\\tt np.arange(3).reshape((3,\\, 1)) + np.arange(3)}$', ha='left', size=12, va='bottom')
    _ax.set_xlim(0, 16)
    _ax.set_ylim(0.5, 12.5)
    _fig.savefig('images/02.05-broadcasting.png')
    return dotted, draw_cube, np, plt, solid


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Aggregation and Grouping

        Figures from the chapter on aggregation and grouping
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Split-Apply-Combine
        """
    )
    return


@app.cell
def _(plt):
    def draw_dataframe(df, loc=None, width=None, ax=None, linestyle=None, textstyle=None):
        loc = loc or [0, 0]
        width = width or 1
        x, y = loc
        if _ax is None:
            _ax = plt.gca()
        ncols = len(df.columns) + 1
        nrows = len(df.index) + 1
        dx = dy = width / ncols
        if linestyle is None:
            linestyle = {'color': 'black'}
        if textstyle is None:
            textstyle = {'size': 12}
        textstyle.update({'ha': 'center', 'va': 'center'})
        for _i in range(ncols + 1):
            plt.plot(2 * [x + _i * dx], [y, y + dy * nrows], **linestyle)
        for _i in range(nrows + 1):
            plt.plot([x, x + dx * ncols], 2 * [y + _i * dy], **linestyle)
        for _i in range(nrows - 1):
            plt.text(x + 0.5 * dx, y + (_i + 0.5) * dy, str(df.index[::-1][_i]), **textstyle)
        for _i in range(ncols - 1):
            plt.text(x + (_i + 1.5) * dx, y + (nrows - 0.5) * dy, str(df.columns[_i]), style='italic', **textstyle)
        if df.index.name:
            plt.text(x + 0.5 * dx, y + (nrows - 0.5) * dy, str(df.index.name), style='italic', **textstyle)
        for _i in range(nrows - 1):
            for j in range(ncols - 1):
                plt.text(x + (j + 1.5) * dx, y + (_i + 0.5) * dy, str(df.values[::-1][_i, j]), **textstyle)
    import pandas as pd
    df = pd.DataFrame({'data': [1, 2, 3, 4, 5, 6]}, index=['A', 'B', 'C', 'A', 'B', 'C'])
    df.index.name = 'key'
    _fig = plt.figure(figsize=(8, 6), facecolor='white')
    _ax = plt.axes([0, 0, 1, 1])
    _ax.axis('off')
    draw_dataframe(df, [0, 0])
    for y, _ind in zip([3, 1, -1], 'ABC'):
        split = df[df.index == _ind]
        draw_dataframe(split, [2, y])
        sum = pd.DataFrame(split.sum()).T
        sum.index = [_ind]
        sum.index.name = 'key'
        sum.columns = ['data']
        draw_dataframe(sum, [4, y + 0.25])
    result = df.groupby(df.index).sum()
    draw_dataframe(result, [6, 0.75])
    style = dict(fontsize=14, ha='center', weight='bold')
    plt.text(0.5, 3.6, 'Input', **style)
    plt.text(2.5, 4.6, 'Split', **style)
    plt.text(4.5, 4.35, 'Apply (sum)', **style)
    plt.text(6.5, 2.85, 'Combine', **style)
    arrowprops = dict(facecolor='black', width=1, headwidth=6)
    plt.annotate('', (1.8, 3.6), (1.2, 2.8), arrowprops=arrowprops)
    plt.annotate('', (1.8, 1.75), (1.2, 1.75), arrowprops=arrowprops)
    plt.annotate('', (1.8, -0.1), (1.2, 0.7), arrowprops=arrowprops)
    plt.annotate('', (3.8, 3.8), (3.2, 3.8), arrowprops=arrowprops)
    plt.annotate('', (3.8, 1.75), (3.2, 1.75), arrowprops=arrowprops)
    plt.annotate('', (3.8, -0.3), (3.2, -0.3), arrowprops=arrowprops)
    plt.annotate('', (5.8, 2.8), (5.2, 3.6), arrowprops=arrowprops)
    plt.annotate('', (5.8, 1.75), (5.2, 1.75), arrowprops=arrowprops)
    plt.annotate('', (5.8, 0.7), (5.2, -0.1), arrowprops=arrowprops)
    plt.axis('equal')
    plt.ylim(-1.5, 5)
    _fig.savefig('images/03.08-split-apply-combine.png')
    return arrowprops, df, draw_dataframe, pd, result, split, style, sum, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## What Is Machine Learning?
        """
    )
    return


@app.cell
def _(plt):
    def format_plot(ax, title):
        _ax.xaxis.set_major_formatter(plt.NullFormatter())
        _ax.yaxis.set_major_formatter(plt.NullFormatter())
        _ax.set_xlabel('feature 1', color='gray')
        _ax.set_ylabel('feature 2', color='gray')
        _ax.set_title(title, color='gray')
    return (format_plot,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Classification Example Figures

        [Figure context](05.01-What-Is-Machine-Learning.ipynb#Classification:-Predicting-Discrete-Labels)

        The following code generates the figures from the Classification section.
        """
    )
    return


@app.cell
def _():
    from sklearn.datasets import make_blobs
    from sklearn.svm import SVC
    X, y_1 = make_blobs(n_samples=50, centers=2, random_state=0, cluster_std=0.6)
    clf = SVC(kernel='linear')
    clf.fit(X, y_1)
    X2, _ = make_blobs(n_samples=80, centers=2, random_state=0, cluster_std=0.8)
    X2 = X2[50:]
    y2 = clf.predict(X2)
    return SVC, X, X2, clf, make_blobs, y2, y_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Classification Example Figure 1
        """
    )
    return


@app.cell
def _(X, format_plot, plt, y_1):
    _fig, _ax = plt.subplots(figsize=(8, 6))
    point_style = dict(cmap='Paired', s=50)
    _ax.scatter(X[:, 0], X[:, 1], c=y_1, **point_style)
    format_plot(_ax, 'Input Data')
    _ax.axis([-1, 4, -2, 7])
    _fig.savefig('images/05.01-classification-1.png')
    return (point_style,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Classification Example Figure 2
        """
    )
    return


@app.cell
def _(X, clf, format_plot, np, plt, point_style, y_1):
    _xx = np.linspace(-1, 4, 10)
    _yy = np.linspace(-2, 7, 10)
    xy1, xy2 = np.meshgrid(_xx, _yy)
    Z = np.array([clf.decision_function([t]) for t in zip(xy1.flat, xy2.flat)]).reshape(xy1.shape)
    _fig, _ax = plt.subplots(figsize=(8, 6))
    line_style = dict(levels=[-1.0, 0.0, 1.0], linestyles=['dashed', 'solid', 'dashed'], colors='gray', linewidths=1)
    _ax.scatter(X[:, 0], X[:, 1], c=y_1, **point_style)
    _ax.contour(xy1, xy2, Z, **line_style)
    format_plot(_ax, 'Model Learned from Input Data')
    _ax.axis([-1, 4, -2, 7])
    _fig.savefig('images/05.01-classification-2.png')
    return Z, line_style, xy1, xy2


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Classification Example Figure 3
        """
    )
    return


@app.cell
def _(X2, Z, format_plot, line_style, plt, point_style, xy1, xy2, y2):
    _fig, _ax = plt.subplots(1, 2, figsize=(16, 6))
    _fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)
    _ax[0].scatter(X2[:, 0], X2[:, 1], c='gray', **point_style)
    _ax[0].axis([-1, 4, -2, 7])
    _ax[1].scatter(X2[:, 0], X2[:, 1], c=y2, **point_style)
    _ax[1].contour(xy1, xy2, Z, **line_style)
    _ax[1].axis([-1, 4, -2, 7])
    format_plot(_ax[0], 'Unknown Data')
    format_plot(_ax[1], 'Predicted Labels')
    _fig.savefig('images/05.01-classification-3.png')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Regression Example Figures

        [Figure Context](05.01-What-Is-Machine-Learning.ipynb#Regression:-Predicting-Continuous-Labels)

        The following code generates the figures from the regression section.
        """
    )
    return


@app.cell
def _(np):
    from sklearn.linear_model import LinearRegression
    _rng = np.random.RandomState(1)
    X_1 = _rng.randn(200, 2)
    y_2 = np.dot(X_1, [-2, 1]) + 0.1 * _rng.randn(X_1.shape[0])
    model = LinearRegression()
    model.fit(X_1, y_2)
    X2_1 = _rng.randn(100, 2)
    y2_1 = model.predict(X2_1)
    return LinearRegression, X2_1, X_1, model, y2_1, y_2


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Regression Example Figure 1
        """
    )
    return


@app.cell
def _(X_1, format_plot, plt, y_2):
    _fig, _ax = plt.subplots()
    _points = _ax.scatter(X_1[:, 0], X_1[:, 1], c=y_2, s=50, cmap='viridis')
    format_plot(_ax, 'Input Data')
    _ax.axis([-4, 4, -3, 3])
    _fig.savefig('images/05.01-regression-1.png')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Regression Example Figure 2
        """
    )
    return


@app.cell
def _(X_1, np, plt, y_2):
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    _points = np.hstack([X_1, y_2[:, None]]).reshape(-1, 1, 3)
    segments = np.hstack([_points, _points])
    segments[:, 0, 2] = -8
    _fig = plt.figure(figsize=(8, 6))
    _ax = _fig.add_subplot(111, projection='3d')
    _ax.scatter(X_1[:, 0], X_1[:, 1], y_2, c=y_2, s=35, cmap='viridis')
    _ax.add_collection3d(Line3DCollection(segments, colors='gray', alpha=0.2))
    _ax.scatter(X_1[:, 0], X_1[:, 1], -8 + np.zeros(X_1.shape[0]), c=y_2, s=10, cmap='viridis')
    _ax.patch.set_facecolor('white')
    _ax.view_init(elev=20, azim=-70)
    _ax.set_zlim3d(-8, 8)
    _ax.xaxis.set_major_formatter(plt.NullFormatter())
    _ax.yaxis.set_major_formatter(plt.NullFormatter())
    _ax.zaxis.set_major_formatter(plt.NullFormatter())
    _ax.set(xlabel='feature 1', ylabel='feature 2', zlabel='label')
    _ax.w_xaxis.line.set_visible(False)
    _ax.w_yaxis.line.set_visible(False)
    _ax.w_zaxis.line.set_visible(False)
    for tick in _ax.w_xaxis.get_ticklines():
        tick.set_visible(False)
    for tick in _ax.w_yaxis.get_ticklines():
        tick.set_visible(False)
    for tick in _ax.w_zaxis.get_ticklines():
        tick.set_visible(False)
    _ax.grid(False)
    _fig.savefig('images/05.01-regression-2.png')
    return Line3DCollection, segments, tick


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Regression Example Figure 3
        """
    )
    return


@app.cell
def _(X_1, format_plot, model, np, plt, y_2):
    from matplotlib.collections import LineCollection
    _fig, _ax = plt.subplots()
    pts = _ax.scatter(X_1[:, 0], X_1[:, 1], c=y_2, s=50, cmap='viridis', zorder=2)
    _xx, _yy = np.meshgrid(np.linspace(-4, 4), np.linspace(-3, 3))
    _Xfit = np.vstack([_xx.ravel(), _yy.ravel()]).T
    _yfit = model.predict(_Xfit)
    zz = _yfit.reshape(_xx.shape)
    _ax.pcolorfast([-4, 4], [-3, 3], zz, alpha=0.5, cmap='viridis', norm=pts.norm, zorder=1)
    format_plot(_ax, 'Input Data with Linear Fit')
    _ax.axis([-4, 4, -3, 3])
    _fig.savefig('images/05.01-regression-3.png')
    return LineCollection, pts, zz


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Regression Example Figure 4
        """
    )
    return


@app.cell
def _(X2_1, format_plot, plt, pts, y2_1):
    _fig, _ax = plt.subplots(1, 2, figsize=(16, 6))
    _fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)
    _ax[0].scatter(X2_1[:, 0], X2_1[:, 1], c='gray', s=50)
    _ax[0].axis([-4, 4, -3, 3])
    _ax[1].scatter(X2_1[:, 0], X2_1[:, 1], c=y2_1, s=50, cmap='viridis', norm=pts.norm)
    _ax[1].axis([-4, 4, -3, 3])
    format_plot(_ax[0], 'Unknown Data')
    format_plot(_ax[1], 'Predicted Labels')
    _fig.savefig('images/05.01-regression-4.png')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Clustering Example Figures

        [Figure context](#Clustering:-Inferring-Labels-on-Unlabeled-Data)

        The following code generates the figures from the clustering section.
        """
    )
    return


@app.cell
def _(make_blobs):
    from sklearn.cluster import KMeans
    X_2, y_3 = make_blobs(n_samples=100, centers=4, random_state=42, cluster_std=1.5)
    model_1 = KMeans(4, random_state=0)
    y_3 = model_1.fit_predict(X_2)
    return KMeans, X_2, model_1, y_3


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Clustering Example Figure 1
        """
    )
    return


@app.cell
def _(X_2, format_plot, plt):
    _fig, _ax = plt.subplots(figsize=(8, 6))
    _ax.scatter(X_2[:, 0], X_2[:, 1], s=50, color='gray')
    format_plot(_ax, 'Input Data')
    _fig.savefig('images/05.01-clustering-1.png')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Clustering Example Figure 2
        """
    )
    return


@app.cell
def _(X_2, format_plot, plt, y_3):
    _fig, _ax = plt.subplots(figsize=(8, 6))
    _ax.scatter(X_2[:, 0], X_2[:, 1], s=50, c=y_3, cmap='viridis')
    format_plot(_ax, 'Learned Cluster Labels')
    _fig.savefig('images/05.01-clustering-2.png')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Dimensionality Reduction Example Figures

        [Figure context](05.01-What-Is-Machine-Learning.ipynb#Dimensionality-Reduction:-Inferring-Structure-of-Unlabeled-Data)

        The following code generates the figures from the dimensionality reduction section.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Dimensionality Reduction Example Figure 1
        """
    )
    return


@app.cell
def _(format_plot, plt):
    from sklearn.datasets import make_swiss_roll
    X_3, y_4 = make_swiss_roll(200, noise=0.5, random_state=42)
    X_3 = X_3[:, [0, 2]]
    _fig, _ax = plt.subplots()
    _ax.scatter(X_3[:, 0], X_3[:, 1], color='gray', s=30)
    format_plot(_ax, 'Input Data')
    _fig.savefig('images/05.01-dimesionality-1.png')
    return X_3, make_swiss_roll, y_4


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Dimensionality Reduction Example Figure 2
        """
    )
    return


@app.cell
def _(X_3, format_plot, plt):
    from sklearn.manifold import Isomap
    model_2 = Isomap(n_neighbors=8, n_components=1)
    y_fit = model_2.fit_transform(X_3).ravel()
    _fig, _ax = plt.subplots()
    pts_1 = _ax.scatter(X_3[:, 0], X_3[:, 1], c=y_fit, cmap='viridis', s=30)
    cb = _fig.colorbar(pts_1, ax=_ax)
    format_plot(_ax, 'Learned Latent Parameter')
    cb.set_ticks([])
    cb.set_label('Latent Variable', color='gray')
    _fig.savefig('images/05.01-dimesionality-2.png')
    return Isomap, cb, model_2, pts_1, y_fit


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Introducing Scikit-Learn
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Features and Labels Grid

        The following is the code generating the diagram showing the features matrix and target array.
        """
    )
    return


@app.cell
def _(plt):
    _fig = plt.figure(figsize=(6, 4))
    _ax = _fig.add_axes([0, 0, 1, 1])
    _ax.axis('off')
    _ax.axis('equal')
    _ax.vlines(range(6), ymin=0, ymax=9, lw=1, color='black')
    _ax.hlines(range(10), xmin=0, xmax=5, lw=1, color='black')
    font_prop = dict(size=12, family='monospace')
    _ax.text(-1, -1, 'Feature Matrix ($X$)', size=14)
    _ax.text(0.1, -0.3, 'n_features $\\longrightarrow$', **font_prop)
    _ax.text(-0.1, 0.1, '$\\longleftarrow$ n_samples', rotation=90, va='top', ha='right', **font_prop)
    _ax.vlines(range(8, 10), ymin=0, ymax=9, lw=1, color='black')
    _ax.hlines(range(10), xmin=8, xmax=9, lw=1, color='black')
    _ax.text(7, -1, 'Target Vector ($y$)', size=14)
    _ax.text(7.9, 0.1, '$\\longleftarrow$ n_samples', rotation=90, va='top', ha='right', **font_prop)
    _ax.set_ylim(10, -2)
    _fig.savefig('images/05.02-samples-features.png')
    return (font_prop,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Hyperparameters and Model Validation
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Cross-Validation Figures
        """
    )
    return


@app.cell
def _(plt):
    def draw_rects(N, ax, textprop={}):
        for _i in range(N):
            _ax.add_patch(plt.Rectangle((0, _i), 5, 0.7, fc='white', ec='lightgray'))
            _ax.add_patch(plt.Rectangle((5.0 * _i / N, _i), 5.0 / N, 0.7, fc='lightgray'))
            _ax.text(5.0 * (_i + 0.5) / N, _i + 0.35, 'validation\nset', ha='center', va='center', **textprop)
            _ax.text(0, _i + 0.35, 'trial {0}'.format(N - _i), ha='right', va='center', rotation=90, **textprop)
        _ax.set_xlim(-1, 6)
        _ax.set_ylim(-0.2, N + 0.2)
    return (draw_rects,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### 2-Fold Cross-Validation
        """
    )
    return


@app.cell
def _(draw_rects, plt):
    _fig = plt.figure()
    _ax = _fig.add_axes([0, 0, 1, 1])
    _ax.axis('off')
    draw_rects(2, _ax, textprop=dict(size=14))
    _fig.savefig('images/05.03-2-fold-CV.png')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### 5-Fold Cross-Validation
        """
    )
    return


@app.cell
def _(draw_rects, plt):
    _fig = plt.figure(figsize=(8, 5))
    _ax = _fig.add_axes([0, 0, 1, 1])
    _ax.axis('off')
    draw_rects(5, _ax, textprop=dict(size=10))
    _fig.savefig('images/05.03-5-fold-CV.png')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Overfitting and Underfitting
        """
    )
    return


@app.cell
def _(np):
    def make_data(N=30, err=0.8, rseed=1):
        _rng = np.random.RandomState(rseed)
        X = _rng.rand(N, 1) ** 2
        y = 10 - 1.0 / (X.ravel() + 0.1)
        if err > 0:
            y = y + err * _rng.randn(N)
        return (X, y)
    return (make_data,)


@app.cell
def _(LinearRegression):
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import make_pipeline

    def PolynomialRegression(degree=2, **kwargs):
        return make_pipeline(PolynomialFeatures(degree),
                             LinearRegression(**kwargs))
    return PolynomialFeatures, PolynomialRegression, make_pipeline


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Bias-Variance Tradeoff
        """
    )
    return


@app.cell
def _(PolynomialRegression, make_data, np, plt):
    X_4, y_5 = make_data()
    xfit = np.linspace(-0.1, 1.0, 1000)[:, None]
    model1 = PolynomialRegression(1).fit(X_4, y_5)
    model20 = PolynomialRegression(20).fit(X_4, y_5)
    _fig, _ax = plt.subplots(1, 2, figsize=(16, 6))
    _fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)
    _ax[0].scatter(X_4.ravel(), y_5, s=40)
    _ax[0].plot(xfit.ravel(), model1.predict(xfit), color='gray')
    _ax[0].axis([-0.1, 1.0, -2, 14])
    _ax[0].set_title('High-bias model: Underfits the data', size=14)
    _ax[1].scatter(X_4.ravel(), y_5, s=40)
    _ax[1].plot(xfit.ravel(), model20.predict(xfit), color='gray')
    _ax[1].axis([-0.1, 1.0, -2, 14])
    _ax[1].set_title('High-variance model: Overfits the data', size=14)
    _fig.savefig('images/05.03-bias-variance.png')
    return X_4, model1, model20, xfit, y_5


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Bias-Variance Tradeoff Metrics
        """
    )
    return


@app.cell
def _(X_4, make_data, model1, model20, plt, xfit, y_5):
    _fig, _ax = plt.subplots(1, 2, figsize=(16, 6))
    _fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)
    X2_2, y2_2 = make_data(10, rseed=42)
    _ax[0].scatter(X_4.ravel(), y_5, s=40, c='blue')
    _ax[0].plot(xfit.ravel(), model1.predict(xfit), color='gray')
    _ax[0].axis([-0.1, 1.0, -2, 14])
    _ax[0].set_title('High-bias model: Underfits the data', size=14)
    _ax[0].scatter(X2_2.ravel(), y2_2, s=40, c='red')
    _ax[0].text(0.02, 0.98, 'training score: $R^2$ = {0:.2f}'.format(model1.score(X_4, y_5)), ha='left', va='top', transform=_ax[0].transAxes, size=14, color='blue')
    _ax[0].text(0.02, 0.91, 'validation score: $R^2$ = {0:.2f}'.format(model1.score(X2_2, y2_2)), ha='left', va='top', transform=_ax[0].transAxes, size=14, color='red')
    _ax[1].scatter(X_4.ravel(), y_5, s=40, c='blue')
    _ax[1].plot(xfit.ravel(), model20.predict(xfit), color='gray')
    _ax[1].axis([-0.1, 1.0, -2, 14])
    _ax[1].set_title('High-variance model: Overfits the data', size=14)
    _ax[1].scatter(X2_2.ravel(), y2_2, s=40, c='red')
    _ax[1].text(0.02, 0.98, 'training score: $R^2$ = {0:.2g}'.format(model20.score(X_4, y_5)), ha='left', va='top', transform=_ax[1].transAxes, size=14, color='blue')
    _ax[1].text(0.02, 0.91, 'validation score: $R^2$ = {0:.2g}'.format(model20.score(X2_2, y2_2)), ha='left', va='top', transform=_ax[1].transAxes, size=14, color='red')
    _fig.savefig('images/05.03-bias-variance-2.png')
    return X2_2, y2_2


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Validation Curve
        """
    )
    return


@app.cell
def _(np, plt):
    x = np.linspace(0, 1, 1000)
    _y1 = -(x - 0.5) ** 2
    y2_3 = _y1 - 0.33 + np.exp(x - 1)
    _fig, _ax = plt.subplots(figsize=(8, 6))
    _ax.plot(x, y2_3, lw=10, alpha=0.5, color='blue')
    _ax.plot(x, _y1, lw=10, alpha=0.5, color='red')
    _ax.text(0.15, 0.05, 'training score', rotation=45, size=16, color='blue')
    _ax.text(0.2, -0.05, 'validation score', rotation=20, size=16, color='red')
    _ax.text(0.02, 0.1, '$\\longleftarrow$ High Bias', size=18, rotation=90, va='center')
    _ax.text(0.98, 0.1, '$\\longleftarrow$ High Variance $\\longrightarrow$', size=18, rotation=90, ha='right', va='center')
    _ax.text(0.48, -0.12, 'Best$\\longrightarrow$\nModel', size=18, rotation=90, va='center')
    _ax.set_xlim(0, 1)
    _ax.set_ylim(-0.3, 0.5)
    _ax.set_xlabel('model complexity $\\longrightarrow$', size=14)
    _ax.set_ylabel('model score $\\longrightarrow$', size=14)
    _ax.xaxis.set_major_formatter(plt.NullFormatter())
    _ax.yaxis.set_major_formatter(plt.NullFormatter())
    _ax.set_title('Validation Curve Schematic', size=16)
    _fig.savefig('images/05.03-validation-curve.png')
    return x, y2_3


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Learning Curve
        """
    )
    return


@app.cell
def _(np, plt, x):
    N = np.linspace(0, 1, 1000)
    _y1 = 0.75 + 0.2 * np.exp(-4 * N)
    y2_4 = 0.7 - 0.6 * np.exp(-4 * N)
    _fig, _ax = plt.subplots(figsize=(8, 6))
    _ax.plot(x, _y1, lw=10, alpha=0.5, color='blue')
    _ax.plot(x, y2_4, lw=10, alpha=0.5, color='red')
    _ax.text(0.2, 0.83, 'training score', rotation=-10, size=16, color='blue')
    _ax.text(0.2, 0.5, 'validation score', rotation=30, size=16, color='red')
    _ax.text(0.98, 0.45, 'Good Fit $\\longrightarrow$', size=18, rotation=90, ha='right', va='center')
    _ax.text(0.02, 0.57, '$\\longleftarrow$ High Variance $\\longrightarrow$', size=18, rotation=90, va='center')
    _ax.set_xlim(0, 1)
    _ax.set_ylim(0, 1)
    _ax.set_xlabel('training set size $\\longrightarrow$', size=14)
    _ax.set_ylabel('model score $\\longrightarrow$', size=14)
    _ax.xaxis.set_major_formatter(plt.NullFormatter())
    _ax.yaxis.set_major_formatter(plt.NullFormatter())
    _ax.set_title('Learning Curve Schematic', size=16)
    _fig.savefig('images/05.03-learning-curve.png')
    return N, y2_4


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Gaussian Naive Bayes

        ### Gaussian Naive Bayes Example

        [Figure Context](05.05-Naive-Bayes.ipynb#Gaussian-Naive-Bayes)
        """
    )
    return


@app.cell
def _(make_blobs, np, plt):
    X_5, y_6 = make_blobs(100, 2, centers=2, random_state=2, cluster_std=1.5)
    _fig, _ax = plt.subplots()
    _ax.scatter(X_5[:, 0], X_5[:, 1], c=y_6, s=50, cmap='RdBu')
    _ax.set_title('Naive Bayes Model', size=14)
    xlim = (-8, 8)
    ylim = (-15, 5)
    xg = np.linspace(xlim[0], xlim[1], 60)
    yg = np.linspace(ylim[0], ylim[1], 40)
    _xx, _yy = np.meshgrid(xg, yg)
    Xgrid = np.vstack([_xx.ravel(), _yy.ravel()]).T
    for label, color in enumerate(['red', 'blue']):
        mask = y_6 == label
        mu, std = (X_5[mask].mean(0), X_5[mask].std(0))
        P = np.exp(-0.5 * (Xgrid - mu) ** 2 / std ** 2).prod(1)
        Pm = np.ma.masked_array(P, P < 0.03)
        _ax.pcolorfast(xg, yg, Pm.reshape(_xx.shape), alpha=0.5, cmap=color.title() + 's')
        _ax.contour(_xx, _yy, P.reshape(_xx.shape), levels=[0.01, 0.1, 0.5, 0.9], colors=color, alpha=0.2)
    _ax.set(xlim=xlim, ylim=ylim)
    _fig.savefig('images/05.05-gaussian-NB.png')
    return (
        P,
        Pm,
        X_5,
        Xgrid,
        color,
        label,
        mask,
        mu,
        std,
        xg,
        xlim,
        y_6,
        yg,
        ylim,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Linear Regression

        ### Gaussian Basis Functions

        [Figure Context](05.06-Linear-Regression.ipynb#Gaussian-Basis-Functions)
        """
    )
    return


@app.cell
def _(LinearRegression, make_pipeline, np, plt):
    from sklearn.base import BaseEstimator, TransformerMixin

    class GaussianFeatures(BaseEstimator, TransformerMixin):
        """Uniformly-spaced Gaussian Features for 1D input"""

        def __init__(self, N, width_factor=2.0):
            self.N = N
            self.width_factor = width_factor

        @staticmethod
        def _gauss_basis(x, y, width, axis=None):
            arg = (x - y) / width
            return np.exp(-0.5 * np.sum(arg ** 2, axis))

        def fit(self, X, y=None):
            self.centers_ = np.linspace(X.min(), X.max(), self.N)
            self.width_ = self.width_factor * (self.centers_[1] - self.centers_[0])
            return self

        def transform(self, X):
            return self._gauss_basis(X[:, :, np.newaxis], self.centers_, self.width_, axis=1)
    _rng = np.random.RandomState(1)
    x_1 = 10 * _rng.rand(50)
    y_7 = np.sin(x_1) + 0.1 * _rng.randn(50)
    xfit_1 = np.linspace(0, 10, 1000)
    gauss_model = make_pipeline(GaussianFeatures(10, 1.0), LinearRegression())
    gauss_model.fit(x_1[:, np.newaxis], y_7)
    _yfit = gauss_model.predict(xfit_1[:, np.newaxis])
    gf = gauss_model.named_steps['gaussianfeatures']
    lm = gauss_model.named_steps['linearregression']
    _fig, _ax = plt.subplots()
    for _i in range(10):
        selector = np.zeros(10)
        selector[_i] = 1
        _Xfit = gf.transform(xfit_1[:, None]) * selector
        _yfit = lm.predict(_Xfit)
        _ax.fill_between(xfit_1, _yfit.min(), _yfit, color='gray', alpha=0.2)
    _ax.scatter(x_1, y_7)
    _ax.plot(xfit_1, gauss_model.predict(xfit_1[:, np.newaxis]))
    _ax.set_xlim(0, 10)
    _ax.set_ylim(_yfit.min(), 1.5)
    _fig.savefig('images/05.06-gaussian-basis.png')
    return (
        BaseEstimator,
        GaussianFeatures,
        TransformerMixin,
        gauss_model,
        gf,
        lm,
        selector,
        x_1,
        xfit_1,
        y_7,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Random Forests
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Helper Code

        The following will create a module ``helpers_05_08.py`` which contains some tools used in [In-Depth: Decision Trees and Random Forests](05.08-Random-Forests.ipynb).
        """
    )
    return


@app.cell
def _():
    # magic command not supported in marimo; please file an issue to add support
    # %%file helpers_05_08.py
    # 
    # import numpy as np
    # import matplotlib.pyplot as plt
    # from sklearn.tree import DecisionTreeClassifier
    # from ipywidgets import interact
    # 
    # 
    # def visualize_tree(estimator, X, y, boundaries=True,
    #                    xlim=None, ylim=None, ax=None):
    #     ax = ax or plt.gca()
    #     
    #     # Plot the training points
    #     ax.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap='viridis',
    #                clim=(y.min(), y.max()), zorder=3)
    #     ax.axis('tight')
    #     ax.axis('off')
    #     if xlim is None:
    #         xlim = ax.get_xlim()
    #     if ylim is None:
    #         ylim = ax.get_ylim()
    #     
    #     # fit the estimator
    #     estimator.fit(X, y)
    #     xx, yy = np.meshgrid(np.linspace(*xlim, num=200),
    #                          np.linspace(*ylim, num=200))
    #     Z = estimator.predict(np.c_[xx.ravel(), yy.ravel()])
    # 
    #     # Put the result into a color plot
    #     n_classes = len(np.unique(y))
    #     Z = Z.reshape(xx.shape)
    #     contours = ax.contourf(xx, yy, Z, alpha=0.3,
    #                            levels=np.arange(n_classes + 1) - 0.5,
    #                            cmap='viridis', zorder=1)
    # 
    #     ax.set(xlim=xlim, ylim=ylim)
    #     
    #     # Plot the decision boundaries
    #     def plot_boundaries(i, xlim, ylim):
    #         if i >= 0:
    #             tree = estimator.tree_
    #         
    #             if tree.feature[i] == 0:
    #                 ax.plot([tree.threshold[i], tree.threshold[i]], ylim, '-k', zorder=2)
    #                 plot_boundaries(tree.children_left[i],
    #                                 [xlim[0], tree.threshold[i]], ylim)
    #                 plot_boundaries(tree.children_right[i],
    #                                 [tree.threshold[i], xlim[1]], ylim)
    #         
    #             elif tree.feature[i] == 1:
    #                 ax.plot(xlim, [tree.threshold[i], tree.threshold[i]], '-k', zorder=2)
    #                 plot_boundaries(tree.children_left[i], xlim,
    #                                 [ylim[0], tree.threshold[i]])
    #                 plot_boundaries(tree.children_right[i], xlim,
    #                                 [tree.threshold[i], ylim[1]])
    #             
    #     if boundaries:
    #         plot_boundaries(0, xlim, ylim)
    # 
    # 
    # def plot_tree_interactive(X, y):
    #     def interactive_tree(depth=5):
    #         clf = DecisionTreeClassifier(max_depth=depth, random_state=0)
    #         visualize_tree(clf, X, y)
    # 
    #     return interact(interactive_tree, depth=(1, 5))
    # 
    # 
    # def randomized_tree_interactive(X, y):
    #     N = int(0.75 * X.shape[0])
    #     
    #     xlim = (X[:, 0].min(), X[:, 0].max())
    #     ylim = (X[:, 1].min(), X[:, 1].max())
    #     
    #     def fit_randomized_tree(random_state=0):
    #         clf = DecisionTreeClassifier(max_depth=15)
    #         i = np.arange(len(y))
    #         rng = np.random.RandomState(random_state)
    #         rng.shuffle(i)
    #         visualize_tree(clf, X[i[:N]], y[i[:N]], boundaries=False,
    #                        xlim=xlim, ylim=ylim)
    #     
    #     interact(fit_randomized_tree, random_state=(0, 100));
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Decision Tree Example
        """
    )
    return


@app.cell
def _(plt):
    _fig = plt.figure(figsize=(10, 4))
    _ax = _fig.add_axes([0, 0, 0.8, 1], frameon=False, xticks=[], yticks=[])
    _ax.set_title('Example Decision Tree: Animal Classification', size=24)

    def text(ax, x, y, t, size=20, **kwargs):
        _ax.text(x, y, t, ha='center', va='center', size=size, bbox=dict(boxstyle='round', ec='k', fc='w'), **kwargs)
    text(_ax, 0.5, 0.9, 'How big is\nthe animal?', 20)
    text(_ax, 0.3, 0.6, 'Does the animal\nhave horns?', 18)
    text(_ax, 0.7, 0.6, 'Does the animal\nhave two legs?', 18)
    text(_ax, 0.12, 0.3, 'Are the horns\nlonger than 10cm?', 14)
    text(_ax, 0.38, 0.3, 'Is the animal\nwearing a collar?', 14)
    text(_ax, 0.62, 0.3, 'Does the animal\nhave wings?', 14)
    text(_ax, 0.88, 0.3, 'Does the animal\nhave a tail?', 14)
    text(_ax, 0.4, 0.75, '> 1m', 12, alpha=0.6)
    text(_ax, 0.6, 0.75, '< 1m', 12, alpha=0.6)
    text(_ax, 0.21, 0.45, 'yes', 12, alpha=0.6)
    text(_ax, 0.34, 0.45, 'no', 12, alpha=0.6)
    text(_ax, 0.66, 0.45, 'yes', 12, alpha=0.6)
    text(_ax, 0.79, 0.45, 'no', 12, alpha=0.6)
    _ax.plot([0.3, 0.5, 0.7], [0.6, 0.9, 0.6], '-k')
    _ax.plot([0.12, 0.3, 0.38], [0.3, 0.6, 0.3], '-k')
    _ax.plot([0.62, 0.7, 0.88], [0.3, 0.6, 0.3], '-k')
    _ax.plot([0.0, 0.12, 0.2], [0.0, 0.3, 0.0], '--k')
    _ax.plot([0.28, 0.38, 0.48], [0.0, 0.3, 0.0], '--k')
    _ax.plot([0.52, 0.62, 0.72], [0.0, 0.3, 0.0], '--k')
    _ax.plot([0.8, 0.88, 1.0], [0.0, 0.3, 0.0], '--k')
    _ax.axis([0, 1, 0, 1])
    _fig.savefig('images/05.08-decision-tree.png')
    return (text,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Decision Tree Levels
        """
    )
    return


@app.cell
def _(make_blobs, plt):
    from helpers_05_08 import visualize_tree
    from sklearn.tree import DecisionTreeClassifier
    _fig, _ax = plt.subplots(1, 4, figsize=(16, 3))
    _fig.subplots_adjust(left=0.02, right=0.98, wspace=0.1)
    X_6, y_8 = make_blobs(n_samples=300, centers=4, random_state=0, cluster_std=1.0)
    for _axi, _depth in zip(_ax, range(1, 5)):
        model_3 = DecisionTreeClassifier(max_depth=_depth)
        visualize_tree(model_3, X_6, y_8, ax=_axi)
        _axi.set_title('depth = {0}'.format(_depth))
    _fig.savefig('images/05.08-decision-tree-levels.png')
    return DecisionTreeClassifier, X_6, model_3, visualize_tree, y_8


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Decision Tree Overfitting
        """
    )
    return


@app.cell
def _(DecisionTreeClassifier, X_6, plt, visualize_tree, y_8):
    model_4 = DecisionTreeClassifier()
    _fig, _ax = plt.subplots(1, 2, figsize=(16, 6))
    _fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)
    visualize_tree(model_4, X_6[::2], y_8[::2], boundaries=False, ax=_ax[0])
    visualize_tree(model_4, X_6[1::2], y_8[1::2], boundaries=False, ax=_ax[1])
    _fig.savefig('images/05.08-decision-tree-overfitting.png')
    return (model_4,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Principal Component Analysis
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Principal Components Rotation
        """
    )
    return


@app.cell
def _():
    from sklearn.decomposition import PCA
    return (PCA,)


@app.cell
def _(plt):
    def draw_vector(v0, v1, ax=None):
        _ax = _ax or plt.gca()
        arrowprops = dict(arrowstyle='->', linewidth=2, shrinkA=0, shrinkB=0)
        _ax.annotate('', v1, v0, arrowprops=arrowprops)
    return (draw_vector,)


@app.cell
def _(PCA, draw_vector, np, plt):
    _rng = np.random.RandomState(1)
    X_7 = np.dot(_rng.rand(2, 2), _rng.randn(2, 200)).T
    _pca = PCA(n_components=2, whiten=True)
    _pca.fit(X_7)
    _fig, _ax = plt.subplots(1, 2, figsize=(16, 6))
    _fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)
    _ax[0].scatter(X_7[:, 0], X_7[:, 1], alpha=0.2)
    for length, vector in zip(_pca.explained_variance_, _pca.components_):
        v = vector * 3 * np.sqrt(length)
        draw_vector(_pca.mean_, _pca.mean_ + v, ax=_ax[0])
    _ax[0].axis('equal')
    _ax[0].set(xlabel='x', ylabel='y', title='input')
    X_pca = _pca.transform(X_7)
    _ax[1].scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.2)
    draw_vector([0, 0], [0, 3], ax=_ax[1])
    draw_vector([0, 0], [3, 0], ax=_ax[1])
    _ax[1].axis('equal')
    _ax[1].set(xlabel='component 1', ylabel='component 2', title='principal components', xlim=(-5, 5), ylim=(-3, 3.1))
    _fig.savefig('images/05.09-PCA-rotation.png')
    return X_7, X_pca, length, v, vector


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Digits Pixel Components
        """
    )
    return


@app.cell
def _(np, plt):
    def plot_pca_components(x, coefficients=None, mean=0, components=None, imshape=(8, 8), n_components=8, fontsize=12, show_mean=True):
        if coefficients is None:
            coefficients = x
        if components is None:
            components = np.eye(len(coefficients), len(x))
        mean = np.zeros_like(x) + mean
        _fig = plt.figure(figsize=(1.2 * (5 + n_components), 1.2 * 2))
        g = plt.GridSpec(2, 4 + bool(show_mean) + n_components, hspace=0.3)

        def show(i, j, x, title=None):
            _ax = _fig.add_subplot(g[_i, j], xticks=[], yticks=[])
            _ax.imshow(x.reshape(imshape), interpolation='nearest', cmap='binary')
            if title:
                _ax.set_title(title, fontsize=fontsize)
        show(slice(2), slice(2), x, 'True')
        approx = mean.copy()
        counter = 2
        if show_mean:
            show(0, 2, np.zeros_like(x) + mean, '$\\mu$')
            show(1, 2, approx, '$1 \\cdot \\mu$')
            counter = counter + 1
        for _i in range(n_components):
            approx = approx + coefficients[_i] * components[_i]
            show(0, _i + counter, components[_i], '$c_{0}$'.format(_i + 1))
            show(1, _i + counter, approx, '${0:.2f} \\cdot c_{1}$'.format(coefficients[_i], _i + 1))
            if show_mean or _i > 0:
                plt.gca().text(0, 1.05, '$+$', ha='right', va='bottom', transform=plt.gca().transAxes, fontsize=fontsize)
        show(slice(2), slice(-2, None), approx, 'Approx')
        return _fig
    return (plot_pca_components,)


@app.cell
def _(plot_pca_components, sns):
    from sklearn.datasets import load_digits
    digits = load_digits()
    sns.set_style('white')
    _fig = plot_pca_components(digits.data[10], show_mean=False)
    _fig.savefig('images/05.09-digits-pixel-components.png')
    return digits, load_digits


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Digits PCA Components
        """
    )
    return


@app.cell
def _(PCA, digits, plot_pca_components, sns):
    _pca = PCA(n_components=8)
    Xproj = _pca.fit_transform(digits.data)
    sns.set_style('white')
    _fig = plot_pca_components(digits.data[10], Xproj[10], _pca.mean_, _pca.components_)
    _fig.savefig('images/05.09-digits-pca-components.png')
    return (Xproj,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Manifold Learning
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### LLE vs MDS Linkages
        """
    )
    return


@app.cell
def _(np, plt):
    def make_hello(N=1000, rseed=42):
        _fig, _ax = plt.subplots(figsize=(4, 1))
        _fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        _ax.axis('off')
        _ax.text(0.5, 0.4, 'HELLO', va='center', ha='center', weight='bold', size=85)
        _fig.savefig('hello.png')
        plt.close(_fig)
        from matplotlib.image import imread
        data = imread('hello.png')[::-1, :, 0].T
        _rng = np.random.RandomState(rseed)
        X = _rng.rand(4 * N, 2)
        _i, j = (X * data.shape).astype(int).T
        mask = data[_i, j] < 1
        X = X[mask]
        X[:, 0] = X[:, 0] * (data.shape[0] / data.shape[1])
        X = X[:N]
        return X[np.argsort(X[:, 0])]
    return (make_hello,)


@app.cell
def _(make_hello, np, plt):
    def make_hello_s_curve(X):
        t = (X[:, 0] - 2) * 0.75 * np.pi
        x = np.sin(t)
        y = X[:, 1]
        z = np.sign(t) * (np.cos(t) - 1)
        return np.vstack((x, y, z)).T
    X_8 = make_hello(1000)
    XS = make_hello_s_curve(X_8)
    colorize = dict(c=X_8[:, 0], cmap=plt.cm.get_cmap('rainbow', 5))
    return XS, X_8, colorize, make_hello_s_curve


@app.cell
def _(Line3DCollection, XS, X_8, colorize, np, plt):
    from sklearn.neighbors import NearestNeighbors
    _rng = np.random.RandomState(42)
    _ind = _rng.permutation(len(X_8))
    lines_MDS = [(XS[_i], XS[j]) for _i in _ind[:100] for j in _ind[100:200]]
    nbrs = NearestNeighbors(n_neighbors=100).fit(XS).kneighbors(XS[_ind[:100]])[1]
    lines_LLE = [(XS[_ind[_i]], XS[j]) for _i in range(100) for j in nbrs[_i]]
    titles = ['MDS Linkages', 'LLE Linkages (100 NN)']
    _fig, _ax = plt.subplots(1, 2, figsize=(16, 6), subplot_kw=dict(projection='3d'))
    _fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0, wspace=0)
    for _axi, title, lines in zip(_ax, titles, [lines_MDS, lines_LLE]):
        _axi.scatter3D(XS[:, 0], XS[:, 1], XS[:, 2], **colorize)
        _axi.add_collection(Line3DCollection(lines, lw=1, color='black', alpha=0.05))
        _axi.view_init(elev=10, azim=-80)
        _axi.set_title(title, size=18)
    _fig.savefig('images/05.10-LLE-vs-MDS.png')
    return (
        NearestNeighbors,
        lines,
        lines_LLE,
        lines_MDS,
        nbrs,
        title,
        titles,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## K-Means
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Expectation-Maximization

        [Figure Context](05.11-K-Means.ipynb#K-Means-Algorithm:-Expectation-Maximization)

        The following figure shows a visual depiction of the Expectation-Maximization approach to K Means:
        """
    )
    return


@app.cell
def _(make_blobs, np, plt):
    from sklearn.metrics import pairwise_distances_argmin
    X_9, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=0)
    _rng = np.random.RandomState(42)
    centers = [0, 4] + _rng.randn(4, 2)

    def draw_points(ax, c, factor=1):
        _ax.scatter(X_9[:, 0], X_9[:, 1], c=c, cmap='viridis', s=50 * factor, alpha=0.3)

    def draw_centers(ax, centers, factor=1, alpha=1.0):
        _ax.scatter(centers[:, 0], centers[:, 1], c=np.arange(4), cmap='viridis', s=200 * factor, alpha=alpha)
        _ax.scatter(centers[:, 0], centers[:, 1], c='black', s=50 * factor, alpha=alpha)

    def make_ax(fig, gs):
        _ax = _fig.add_subplot(gs)
        _ax.xaxis.set_major_formatter(plt.NullFormatter())
        _ax.yaxis.set_major_formatter(plt.NullFormatter())
        return _ax
    _fig = plt.figure(figsize=(15, 4))
    gs = plt.GridSpec(4, 15, left=0.02, right=0.98, bottom=0.05, top=0.95, wspace=0.2, hspace=0.2)
    ax0 = make_ax(_fig, gs[:4, :4])
    ax0.text(0.98, 0.98, 'Random Initialization', transform=ax0.transAxes, ha='right', va='top', size=16)
    draw_points(ax0, 'gray', factor=2)
    draw_centers(ax0, centers, factor=2)
    for _i in range(3):
        ax1 = make_ax(_fig, gs[:2, 4 + 2 * _i:6 + 2 * _i])
        ax2 = make_ax(_fig, gs[2:, 5 + 2 * _i:7 + 2 * _i])
        y_pred = pairwise_distances_argmin(X_9, centers)
        draw_points(ax1, y_pred)
        draw_centers(ax1, centers)
        new_centers = np.array([X_9[y_pred == _i].mean(0) for _i in range(4)])
        draw_points(ax2, y_pred)
        draw_centers(ax2, centers, alpha=0.3)
        draw_centers(ax2, new_centers)
        for _i in range(4):
            ax2.annotate('', new_centers[_i], centers[_i], arrowprops=dict(arrowstyle='->', linewidth=1))
        centers = new_centers
        ax1.text(0.95, 0.95, 'E-Step', transform=ax1.transAxes, ha='right', va='top', size=14)
        ax2.text(0.95, 0.95, 'M-Step', transform=ax2.transAxes, ha='right', va='top', size=14)
    y_pred = pairwise_distances_argmin(X_9, centers)
    axf = make_ax(_fig, gs[:4, -4:])
    draw_points(axf, y_pred, factor=2)
    draw_centers(axf, centers, factor=2)
    axf.text(0.98, 0.98, 'Final Clustering', transform=axf.transAxes, ha='right', va='top', size=16)
    _fig.savefig('images/05.11-expectation-maximization.png')
    return (
        X_9,
        ax0,
        ax1,
        ax2,
        axf,
        centers,
        draw_centers,
        draw_points,
        gs,
        make_ax,
        new_centers,
        pairwise_distances_argmin,
        y_pred,
        y_true,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Interactive K-Means

        The following script uses IPython's interactive widgets to demonstrate the K-means algorithm interactively.
        Run this within the IPython notebook to explore the expectation maximization algorithm for computing K Means.
        """
    )
    return


@app.cell
def _():
    # '%matplotlib inline\nimport matplotlib.pyplot as plt\nimport numpy as np\n\nfrom ipywidgets import interact\nfrom sklearn.metrics import pairwise_distances_argmin\nfrom sklearn.datasets import make_blobs\n\ndef plot_kmeans_interactive(min_clusters=1, max_clusters=6):\n    X, y = make_blobs(n_samples=300, centers=4,\n                      random_state=0, cluster_std=0.60)\n        \n    def plot_points(X, labels, n_clusters):\n        plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap=\'viridis\',\n                    vmin=0, vmax=n_clusters - 1);\n            \n    def plot_centers(centers):\n        plt.scatter(centers[:, 0], centers[:, 1], marker=\'o\',\n                    c=np.arange(centers.shape[0]),\n                    s=200, cmap=\'viridis\')\n        plt.scatter(centers[:, 0], centers[:, 1], marker=\'o\',\n                    c=\'black\', s=50)\n            \n\n    def _kmeans_step(frame=0, n_clusters=4):\n        rng = np.random.RandomState(2)\n        labels = np.zeros(X.shape[0])\n        centers = rng.randn(n_clusters, 2)\n\n        nsteps = frame // 3\n\n        for i in range(nsteps + 1):\n            old_centers = centers\n            if i < nsteps or frame % 3 > 0:\n                labels = pairwise_distances_argmin(X, centers)\n\n            if i < nsteps or frame % 3 > 1:\n                centers = np.array([X[labels == j].mean(0)\n                                    for j in range(n_clusters)])\n                nans = np.isnan(centers)\n                centers[nans] = old_centers[nans]\n\n        # plot the data and cluster centers\n        plot_points(X, labels, n_clusters)\n        plot_centers(old_centers)\n\n        # plot new centers if third frame\n        if frame % 3 == 2:\n            for i in range(n_clusters):\n                plt.annotate(\'\', centers[i], old_centers[i], \n                             arrowprops=dict(arrowstyle=\'->\', linewidth=1))\n            plot_centers(centers)\n\n        plt.xlim(-4, 4)\n        plt.ylim(-2, 10)\n\n        if frame % 3 == 1:\n            plt.text(3.8, 9.5, "1. Reassign points to nearest centroid",\n                     ha=\'right\', va=\'top\', size=14)\n        elif frame % 3 == 2:\n            plt.text(3.8, 9.5, "2. Update centroids to cluster means",\n                     ha=\'right\', va=\'top\', size=14)\n    \n    return interact(_kmeans_step, frame=(0, 50),\n                    n_clusters=[min_clusters, max_clusters])\n\nplot_kmeans_interactive();' command supported automatically in marimo
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Gaussian Mixture Models
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Covariance Type

        [Figure Context](http://localhost:8888/notebooks/05.12-Gaussian-Mixtures.ipynb#Choosing-the-Covariance-Type)
        """
    )
    return


@app.cell
def _(np, plt):
    from sklearn.mixture import GaussianMixture
    from matplotlib.patches import Ellipse

    def draw_ellipse(position, covariance, ax=None, **kwargs):
        """Draw an ellipse with a given position and covariance"""
        _ax = _ax or plt.gca()
        if covariance.shape == (2, 2):
            U, s, Vt = np.linalg.svd(covariance)
            angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
            width, height = 2 * np.sqrt(s)
        elif covariance.shape == (2,):
            angle = 0
            width, height = 2 * np.sqrt(covariance)
        else:
            angle = 0
            width = height = 2 * np.sqrt(covariance)
        for nsig in range(1, 4):
            _ax.add_patch(Ellipse(position, nsig * width, nsig * height, angle, **kwargs))
    _fig, _ax = plt.subplots(1, 3, figsize=(14, 4))
    _fig.subplots_adjust(wspace=0.05)
    _rng = np.random.RandomState(5)
    X_10 = np.dot(_rng.randn(500, 2), _rng.randn(2, 2))
    for _i, cov_type in enumerate(['diag', 'spherical', 'full']):
        model_5 = GaussianMixture(1, covariance_type=cov_type).fit(X_10)
        _ax[_i].axis('equal')
        _ax[_i].scatter(X_10[:, 0], X_10[:, 1], alpha=0.5)
        _ax[_i].set_xlim(-3, 3)
        _ax[_i].set_title('covariance_type="{0}"'.format(cov_type), size=14, family='monospace')
        draw_ellipse(model_5.means_[0], model_5.covariances_[0], _ax[_i], alpha=0.2)
        _ax[_i].xaxis.set_major_formatter(plt.NullFormatter())
        _ax[_i].yaxis.set_major_formatter(plt.NullFormatter())
    _fig.savefig('images/05.12-covariance-type.png')
    return Ellipse, GaussianMixture, X_10, cov_type, draw_ellipse, model_5


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
