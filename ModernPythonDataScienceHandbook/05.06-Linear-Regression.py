import marimo

__generated_with = "0.10.15"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # In Depth: Linear Regression
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Just as naive Bayes (discussed in [In Depth: Naive Bayes Classification](05.05-Naive-Bayes.ipynb)) is a good starting point for classification tasks, linear regression models are a good starting point for regression tasks.
        Such models are popular because they can be fit quickly and are straightforward to interpret.
        You are already familiar with the simplest form of linear regression model (i.e., fitting a straight line to two-dimensional data), but such models can be extended to model more complicated data behavior.

        In this chapter we will start with a quick walkthrough of the mathematics behind this well-known problem, before moving on to see how linear models can be generalized to account for more complicated patterns in data.

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
        ## Simple Linear Regression

        We will start with the most familiar linear regression, a straight-line fit to data.
        A straight-line fit is a model of the form:
        $$
        y = ax + b
        $$
        where $a$ is commonly known as the *slope*, and $b$ is commonly known as the *intercept*.

        Consider the following data, which is scattered about a line with a slope of 2 and an intercept of –5 (see the following figure):
        """
    )
    return


@app.cell
def _(np, plt):
    _rng = np.random.RandomState(1)
    x = 10 * _rng.rand(50)
    y = 2 * x - 5 + _rng.randn(50)
    plt.scatter(x, y)
    return x, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We can use Scikit-Learn's `LinearRegression` estimator to fit this data and construct the best-fit line, as shown in the following figure:
        """
    )
    return


@app.cell
def _(np, plt, x, y):
    from sklearn.linear_model import LinearRegression
    model = LinearRegression(fit_intercept=True)
    model.fit(x[:, np.newaxis], y)
    xfit = np.linspace(0, 10, 1000)
    _yfit = model.predict(xfit[:, np.newaxis])
    plt.scatter(x, y)
    plt.plot(xfit, _yfit)
    return LinearRegression, model, xfit


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The slope and intercept of the data are contained in the model's fit parameters, which in Scikit-Learn are always marked by a trailing underscore.
        Here the relevant parameters are `coef_` and `intercept_`:
        """
    )
    return


@app.cell
def _(model):
    print("Model slope:    ", model.coef_[0])
    print("Model intercept:", model.intercept_)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We see that the results are very close to the values used to generate the data, as we might hope.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The `LinearRegression` estimator is much more capable than this, however—in addition to simple straight-line fits, it can also handle multidimensional linear models of the form:
        $$
        y = a_0 + a_1 x_1 + a_2 x_2 + \cdots
        $$
        where there are multiple $x$ values.
        Geometrically, this is akin to fitting a plane to points in three dimensions, or fitting a hyperplane to points in higher dimensions.

        The multidimensional nature of such regressions makes them more difficult to visualize, but we can see one of these fits in action by building some example data, using NumPy's matrix multiplication operator:
        """
    )
    return


@app.cell
def _(model, np):
    _rng = np.random.RandomState(1)
    X = 10 * _rng.rand(100, 3)
    y_1 = 0.5 + np.dot(X, [1.5, -2.0, 1.0])
    model.fit(X, y_1)
    print(model.intercept_)
    print(model.coef_)
    return X, y_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Here the $y$ data is constructed from a linear combination of three random $x$ values, and the linear regression recovers the coefficients used to construct the data.

        In this way, we can use the single `LinearRegression` estimator to fit lines, planes, or hyperplanes to our data.
        It still appears that this approach would be limited to strictly linear relationships between variables, but it turns out we can relax this as well.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Basis Function Regression

        One trick you can use to adapt linear regression to nonlinear relationships between variables is to transform the data according to *basis functions*.
        We have seen one version of this before, in the `PolynomialRegression` pipeline used in [Hyperparameters and Model Validation](05.03-Hyperparameters-and-Model-Validation.ipynb) and [Feature Engineering](05.04-Feature-Engineering.ipynb).
        The idea is to take our multidimensional linear model:
        $$
        y = a_0 + a_1 x_1 + a_2 x_2 + a_3 x_3 + \cdots
        $$
        and build the $x_1, x_2, x_3,$ and so on from our single-dimensional input $x$.
        That is, we let $x_n = f_n(x)$, where $f_n()$ is some function that transforms our data.

        For example, if $f_n(x) = x^n$, our model becomes a polynomial regression:
        $$
        y = a_0 + a_1 x + a_2 x^2 + a_3 x^3 + \cdots
        $$
        Notice that this is *still a linear model*—the linearity refers to the fact that the coefficients $a_n$ never multiply or divide each other.
        What we have effectively done is taken our one-dimensional $x$ values and projected them into a higher dimension, so that a linear fit can fit more complicated relationships between $x$ and $y$.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Polynomial Basis Functions

        This polynomial projection is useful enough that it is built into Scikit-Learn, using the `PolynomialFeatures` transformer:
        """
    )
    return


@app.cell
def _(np):
    from sklearn.preprocessing import PolynomialFeatures
    x_1 = np.array([2, 3, 4])
    poly = PolynomialFeatures(3, include_bias=False)
    poly.fit_transform(x_1[:, None])
    return PolynomialFeatures, poly, x_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We see here that the transformer has converted our one-dimensional array into a three-dimensional array, where each column contains the exponentiated value.
        This new, higher-dimensional data representation can then be plugged into a linear regression.

        As we saw in [Feature Engineering](05.04-Feature-Engineering.ipynb), the cleanest way to accomplish this is to use a pipeline.
        Let's make a 7th-degree polynomial model in this way:
        """
    )
    return


@app.cell
def _(LinearRegression, PolynomialFeatures):
    from sklearn.pipeline import make_pipeline
    poly_model = make_pipeline(PolynomialFeatures(7),
                               LinearRegression())
    return make_pipeline, poly_model


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        With this transform in place, we can use the linear model to fit much more complicated relationships between $x$ and $y$. 
        For example, here is a sine wave with noise (see the following figure):
        """
    )
    return


@app.cell
def _(np, plt, poly_model, xfit):
    _rng = np.random.RandomState(1)
    x_2 = 10 * _rng.rand(50)
    y_2 = np.sin(x_2) + 0.1 * _rng.randn(50)
    poly_model.fit(x_2[:, np.newaxis], y_2)
    _yfit = poly_model.predict(xfit[:, np.newaxis])
    plt.scatter(x_2, y_2)
    plt.plot(xfit, _yfit)
    return x_2, y_2


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Our linear model, through the use of seventh-order polynomial basis functions, can provide an excellent fit to this nonlinear data!
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Gaussian Basis Functions

        Of course, other basis functions are possible.
        For example, one useful pattern is to fit a model that is not a sum of polynomial bases, but a sum of Gaussian bases.
        The result might look something like the following figure:
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ![](images/05.06-gaussian-basis.png)

        [figure source in Appendix](https://github.com/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/06.00-Figure-Code.ipynb)
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The shaded regions in the plot are the scaled basis functions, and when added together they reproduce the smooth curve through the data.
        These Gaussian basis functions are not built into Scikit-Learn, but we can write a custom transformer that will create them, as shown here and illustrated in the following figure (Scikit-Learn transformers are implemented as Python classes; reading Scikit-Learn's source is a good way to see how they can be created):
        """
    )
    return


@app.cell
def _(LinearRegression, make_pipeline, np, plt, x_2, xfit, y_2):
    from sklearn.base import BaseEstimator, TransformerMixin

    class GaussianFeatures(BaseEstimator, TransformerMixin):
        """Uniformly spaced Gaussian features for one-dimensional input"""

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
    gauss_model = make_pipeline(GaussianFeatures(20), LinearRegression())
    gauss_model.fit(x_2[:, np.newaxis], y_2)
    _yfit = gauss_model.predict(xfit[:, np.newaxis])
    plt.scatter(x_2, y_2)
    plt.plot(xfit, _yfit)
    plt.xlim(0, 10)
    return BaseEstimator, GaussianFeatures, TransformerMixin, gauss_model


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        I've included this example just to make clear that there is nothing magic about polynomial basis functions: if you have some sort of intuition into the generating process of your data that makes you think one basis or another might be appropriate, you can use that instead.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Regularization

        The introduction of basis functions into our linear regression makes the model much more flexible, but it also can very quickly lead to overfitting (refer back to [Hyperparameters and Model Validation](05.03-Hyperparameters-and-Model-Validation.ipynb) for a discussion of this).
        For example, the following figure shows what happens if we use a large number of Gaussian basis functions:
        """
    )
    return


@app.cell
def _(
    GaussianFeatures,
    LinearRegression,
    make_pipeline,
    np,
    plt,
    x_2,
    xfit,
    y_2,
):
    model_1 = make_pipeline(GaussianFeatures(30), LinearRegression())
    model_1.fit(x_2[:, np.newaxis], y_2)
    plt.scatter(x_2, y_2)
    plt.plot(xfit, model_1.predict(xfit[:, np.newaxis]))
    plt.xlim(0, 10)
    plt.ylim(-1.5, 1.5)
    return (model_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        With the data projected to the 30-dimensional basis, the model has far too much flexibility and goes to extreme values between locations where it is constrained by data.
        We can see the reason for this if we plot the coefficients of the Gaussian bases with respect to their locations, as shown in the following figure:
        """
    )
    return


@app.cell
def _(
    GaussianFeatures,
    LinearRegression,
    make_pipeline,
    np,
    plt,
    x_2,
    xfit,
    y_2,
):
    def basis_plot(model, title=None):
        fig, ax = plt.subplots(2, sharex=True)
        model.fit(x_2[:, np.newaxis], y_2)
        ax[0].scatter(x_2, y_2)
        ax[0].plot(xfit, model.predict(xfit[:, np.newaxis]))
        ax[0].set(xlabel='x', ylabel='y', ylim=(-1.5, 1.5))
        if title:
            ax[0].set_title(title)
        ax[1].plot(model.steps[0][1].centers_, model.steps[1][1].coef_)
        ax[1].set(xlabel='basis location', ylabel='coefficient', xlim=(0, 10))
    model_2 = make_pipeline(GaussianFeatures(30), LinearRegression())
    basis_plot(model_2)
    return basis_plot, model_2


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The lower panel of this figure shows the amplitude of the basis function at each location.
        This is typical overfitting behavior when basis functions overlap: the coefficients of adjacent basis functions blow up and cancel each other out.
        We know that such behavior is problematic, and it would be nice if we could limit such spikes explicitly in the model by penalizing large values of the model parameters.
        Such a penalty is known as *regularization*, and comes in several forms.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Ridge Regression ($L_2$ Regularization)

        Perhaps the most common form of regularization is known as *ridge regression* or $L_2$ *regularization* (sometimes also called *Tikhonov regularization*).
        This proceeds by penalizing the sum of squares (2-norms) of the model coefficients $\theta_n$. In this case, the penalty on the model fit would be: 
        $$
        P = \alpha\sum_{n=1}^N \theta_n^2
        $$
        where $\alpha$ is a free parameter that controls the strength of the penalty.
        This type of penalized model is built into Scikit-Learn with the `Ridge` estimator (see the following figure):
        """
    )
    return


@app.cell
def _(GaussianFeatures, basis_plot, make_pipeline):
    from sklearn.linear_model import Ridge
    model_3 = make_pipeline(GaussianFeatures(30), Ridge(alpha=0.1))
    basis_plot(model_3, title='Ridge Regression')
    return Ridge, model_3


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The $\alpha$ parameter is essentially a knob controlling the complexity of the resulting model.
        In the limit $\alpha \to 0$, we recover the standard linear regression result; in the limit $\alpha \to \infty$, all model responses will be suppressed.
        One advantage of ridge regression in particular is that it can be computed very efficiently—at hardly more computational cost than the original linear regression model.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Lasso Regression ($L_1$ Regularization)

        Another common type of regularization is known as *lasso regression* or *L~1~ regularization* involves penalizing the sum of absolute values (1-norms) of regression coefficients:
        $$
        P = \alpha\sum_{n=1}^N |\theta_n|
        $$
        Though this is conceptually very similar to ridge regression, the results can differ surprisingly. For example, due to its construction, lasso regression tends to favor *sparse models* where possible: that is, it preferentially sets many model coefficients to exactly zero.

        We can see this behavior if we duplicate the previous example using L1-normalized coefficients (see the following figure):
        """
    )
    return


@app.cell
def _(GaussianFeatures, basis_plot, make_pipeline):
    from sklearn.linear_model import Lasso
    model_4 = make_pipeline(GaussianFeatures(30), Lasso(alpha=0.001, max_iter=2000))
    basis_plot(model_4, title='Lasso Regression')
    return Lasso, model_4


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        With the lasso regression penalty, the majority of the coefficients are exactly zero, with the functional behavior being modeled by a small subset of the available basis functions.
        As with ridge regularization, the $\alpha$ parameter tunes the strength of the penalty and should be determined via, for example, cross-validation (refer back to [Hyperparameters and Model Validation](05.03-Hyperparameters-and-Model-Validation.ipynb) for a discussion of this).
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Example: Predicting Bicycle Traffic
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        As an example, let's take a look at whether we can predict the number of bicycle trips across Seattle's Fremont Bridge based on weather, season, and other factors.
        We already saw this data in [Working With Time Series](03.11-Working-with-Time-Series.ipynb), but here we will join the bike data with another dataset and try to determine the extent to which weather and seasonal factors—temperature, precipitation, and daylight hours—affect the volume of bicycle traffic through this corridor.
        Fortunately, the National Oceanic and Atmospheric Administration (NOAA) makes its daily [weather station data](http://www.ncdc.noaa.gov/cdo-web/search?datasetid=GHCND) available—I used station ID USW00024233—and we can easily use Pandas to join the two data sources.
        We will perform a simple linear regression to relate weather and other information to bicycle counts, in order to estimate how a change in any one of these parameters affects the number of riders on a given day.

        In particular, this is an example of how the tools of Scikit-Learn can be used in a statistical modeling framework, in which the parameters of the model are assumed to have interpretable meaning.
        As discussed previously, this is not a standard approach within machine learning, but such interpretation is possible for some models.

        Let's start by loading the two datasets, indexing by date:
        """
    )
    return


@app.cell
def _():
    # url = 'https://raw.githubusercontent.com/jakevdp/bicycle-data/main'
    # !curl -O {url}/FremontBridge.csv
    # !curl -O {url}/SeattleWeather.csv
    return


@app.cell
def _():
    import pandas as pd
    counts = pd.read_csv('FremontBridge.csv',
                         index_col='Date', parse_dates=True)
    weather = pd.read_csv('SeattleWeather.csv',
                          index_col='DATE', parse_dates=True)
    return counts, pd, weather


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        For simplicity, let's look at data prior to 2020 in order to avoid the effects of the COVID-19 pandemic, which significantly affected commuting patterns in Seattle:
        """
    )
    return


@app.cell
def _(counts, weather):
    counts_1 = counts[counts.index < '2020-01-01']
    weather_1 = weather[weather.index < '2020-01-01']
    return counts_1, weather_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Next we will compute the total daily bicycle traffic, and put this in its own `DataFrame`:
        """
    )
    return


@app.cell
def _(counts_1):
    daily = counts_1.resample('d').sum()
    daily['Total'] = daily.sum(axis=1)
    daily = daily[['Total']]
    return (daily,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We saw previously that the patterns of use generally vary from day to day. Let's account for this in our data by adding binary columns that indicate the day of the week:
        """
    )
    return


@app.cell
def _(daily):
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    for i in range(7):
        daily[days[i]] = (daily.index.dayofweek == i).astype(float)
    return days, i


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Similarly, we might expect riders to behave differently on holidays; let's add an indicator of this as well:
        """
    )
    return


@app.cell
def _(daily, pd):
    from pandas.tseries.holiday import USFederalHolidayCalendar
    cal = USFederalHolidayCalendar()
    holidays = cal.holidays('2012', '2020')
    daily_1 = daily.join(pd.Series(1, index=holidays, name='holiday'))
    daily_1['holiday'].fillna(0, inplace=True)
    return USFederalHolidayCalendar, cal, daily_1, holidays


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We also might suspect that the hours of daylight would affect how many people ride. Let's use the standard astronomical calculation to add this information (see the following figure):
        """
    )
    return


@app.cell
def _(daily_1, np, pd, plt):
    def hours_of_daylight(date, axis=23.44, latitude=47.61):
        """Compute the hours of daylight for the given date"""
        days = (date - pd.datetime(2000, 12, 21)).days
        m = 1.0 - np.tan(np.radians(latitude)) * np.tan(np.radians(axis) * np.cos(days * 2 * np.pi / 365.25))
        return 24.0 * np.degrees(np.arccos(1 - np.clip(m, 0, 2))) / 180.0
    daily_1['daylight_hrs'] = list(map(hours_of_daylight, daily_1.index))
    daily_1[['daylight_hrs']].plot()
    plt.ylim(8, 17)
    return (hours_of_daylight,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We can also add the average temperature and total precipitation to the data.
        In addition to the inches of precipitation, let's add a flag that indicates whether a day is dry (has zero precipitation):
        """
    )
    return


@app.cell
def _(daily_1, weather_1):
    weather_1['Temp (F)'] = 0.5 * (weather_1['TMIN'] + weather_1['TMAX'])
    weather_1['Rainfall (in)'] = weather_1['PRCP']
    weather_1['dry day'] = (weather_1['PRCP'] == 0).astype(int)
    daily_2 = daily_1.join(weather_1[['Rainfall (in)', 'Temp (F)', 'dry day']])
    return (daily_2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Finally, let's add a counter that increases from day 1, and measures how many years have passed.
        This will let us measure any observed annual increase or decrease in daily crossings:
        """
    )
    return


@app.cell
def _(daily_2):
    daily_2['annual'] = (daily_2.index - daily_2.index[0]).days / 365.0
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Now our data is in order, and we can take a look at it:
        """
    )
    return


@app.cell
def _(daily_2):
    daily_2.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        With this in place, we can choose the columns to use, and fit a linear regression model to our data.
        We will set `fit_intercept=False`, because the daily flags essentially operate as their own day-specific intercepts:
        """
    )
    return


@app.cell
def _(LinearRegression, daily_2):
    daily_2.dropna(axis=0, how='any', inplace=True)
    column_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun', 'holiday', 'daylight_hrs', 'Rainfall (in)', 'dry day', 'Temp (F)', 'annual']
    X_1 = daily_2[column_names]
    y_3 = daily_2['Total']
    model_5 = LinearRegression(fit_intercept=False)
    model_5.fit(X_1, y_3)
    daily_2['predicted'] = model_5.predict(X_1)
    return X_1, column_names, model_5, y_3


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Finally, we can compare the total and predicted bicycle traffic visually (see the following figure):
        """
    )
    return


@app.cell
def _(daily_2):
    daily_2[['Total', 'predicted']].plot(alpha=0.5)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        From the fact that the data and model predictions don't line up exactly, it is evident that we have missed some key features.
        Either our features are not complete (i.e., people decide whether to ride to work based on more than just these features), or there are some nonlinear relationships that we have failed to take into account (e.g., perhaps people ride less at both high and low temperatures).
        Nevertheless, our rough approximation is enough to give us some insights, and we can take a look at the coefficients of the linear model to estimate how much each feature contributes to the daily bicycle count:
        """
    )
    return


@app.cell
def _(X_1, model_5, pd):
    params = pd.Series(model_5.coef_, index=X_1.columns)
    params
    return (params,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        These numbers are difficult to interpret without some measure of their uncertainty.
        We can compute these uncertainties quickly using bootstrap resamplings of the data:
        """
    )
    return


@app.cell
def _(X_1, model_5, np, y_3):
    from sklearn.utils import resample
    np.random.seed(1)
    err = np.std([model_5.fit(*resample(X_1, y_3)).coef_ for i in range(1000)], 0)
    return err, resample


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        With these errors estimated, let's again look at the results:
        """
    )
    return


@app.cell
def _(err, params, pd):
    print(pd.DataFrame({'effect': params.round(0),
                        'uncertainty': err.round(0)}))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The `effect` column here, roughly speaking, shows how the number of riders is affected by a change of the feature in question.
        For example, there is a clear divide when it comes to the day of the week: there are thousands fewer riders on weekends than on weekdays.
        We also see that for each additional hour of daylight, 409 ± 26 more people choose to ride; a temperature increase of one degree Fahrenheit encourages 179 ± 7 people to grab their bicycle; a dry day means an average of 2,111 ± 101 more riders,
        and every inch of rainfall leads 2,790 ± 186 riders to choose another mode of transport.
        Once all these effects are accounted for, we see a modest increase of 324 ± 22 new daily riders each year.

        Our simple model is almost certainly missing some relevant information. For example, as mentioned earlier, nonlinear effects (such as effects of precipitation *and* cold temperature) and nonlinear trends within each variable (such as disinclination to ride at very cold and very hot temperatures) cannot be accounted for in a simple linear model.
        Additionally, we have thrown away some of the finer-grained information (such as the difference between a rainy morning and a rainy afternoon), and we have ignored correlations between days (such as the possible effect of a rainy Tuesday on Wednesday's numbers, or the effect of an unexpected sunny day after a streak of rainy days).
        These are all potentially interesting effects, and you now have the tools to begin exploring them if you wish!
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
