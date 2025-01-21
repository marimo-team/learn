import marimo

__generated_with = "0.10.15"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Vectorized String Operations
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        One strength of Python is its relative ease in handling and manipulating string data.
        Pandas builds on this and provides a comprehensive set of *vectorized string operations* that are an important part of the type of munging required when working with (read: cleaning up) real-world data.
        In this chapter, we'll walk through some of the Pandas string operations, and then take a look at using them to partially clean up a very messy dataset of recipes collected from the internet.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Introducing Pandas String Operations

        We saw in previous chapters how tools like NumPy and Pandas generalize arithmetic operations so that we can easily and quickly perform the same operation on many array elements. For example:
        """
    )
    return


@app.cell
def _():
    import numpy as np
    x = np.array([2, 3, 5, 7, 11, 13])
    x * 2
    return np, x


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        This *vectorization* of operations simplifies the syntax of operating on arrays of data: we no longer have to worry about the size or shape of the array, but just about what operation we want done.
        For arrays of strings, NumPy does not provide such simple access, and thus you're stuck using a more verbose loop syntax:
        """
    )
    return


@app.cell
def _():
    data = ['peter', 'Paul', 'MARY', 'gUIDO']
    [s.capitalize() for s in data]
    return (data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        This is perhaps sufficient to work with some data, but it will break if there are any missing values, so this approach requires putting in extra checks:
        """
    )
    return


@app.cell
def _():
    data_1 = ['peter', 'Paul', None, 'MARY', 'gUIDO']
    [s if s is None else s.capitalize() for s in data_1]
    return (data_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        This kind of manual approach is not only verbose and inconvenient, it can be error-prone.

        Pandas includes features to address both this need for vectorized string operations and the need for correctly handling missing data via the `str` attribute of Pandas `Series` and `Index` objects containing strings.
        So, for example, if we create a Pandas `Series` with this data we can directly call the `str.capitalize` method, which has missing value handling built in:
        """
    )
    return


@app.cell
def _(data_1):
    import pandas as pd
    names = pd.Series(data_1)
    names.str.capitalize()
    return names, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Tables of Pandas String Methods

        If you have a good understanding of string manipulation in Python, most of the Pandas string syntax is intuitive enough that it's probably sufficient to just list the available methods. We'll start with that here, before diving deeper into a few of the subtleties.
        The examples in this section use the following `Series` object:
        """
    )
    return


@app.cell
def _(pd):
    monte = pd.Series(['Graham Chapman', 'John Cleese', 'Terry Gilliam',
                       'Eric Idle', 'Terry Jones', 'Michael Palin'])
    return (monte,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Methods Similar to Python String Methods

        Nearly all of Python's built-in string methods are mirrored by a Pandas vectorized string method. Here is a list of Pandas `str` methods that mirror Python string methods:

        |           |                |                |                |
        |-----------|----------------|----------------|----------------|
        |`len()`    | `lower()`      | `translate()`  | `islower()`    | 
        |`ljust()`  | `upper()`      | `startswith()` | `isupper()`    | 
        |`rjust()`  | `find()`       | `endswith()`   | `isnumeric()`  | 
        |`center()` | `rfind()`      | `isalnum()`    | `isdecimal()`  | 
        |`zfill()`  | `index()`      | `isalpha()`    | `split()`      | 
        |`strip()`  | `rindex()`     | `isdigit()`    | `rsplit()`     | 
        |`rstrip()` | `capitalize()` | `isspace()`    | `partition()`  | 
        |`lstrip()` | `swapcase()`   | `istitle()`    | `rpartition()` |

        Notice that these have various return values. Some, like `lower`, return a series of strings:
        """
    )
    return


@app.cell
def _(monte):
    monte.str.lower()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        But some others return numbers:
        """
    )
    return


@app.cell
def _(monte):
    monte.str.len()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Or Boolean values:
        """
    )
    return


@app.cell
def _(monte):
    monte.str.startswith('T')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Still others return lists or other compound values for each element:
        """
    )
    return


@app.cell
def _(monte):
    monte.str.split()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We'll see further manipulations of this kind of series-of-lists object as we continue our discussion.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Methods Using Regular Expressions

        In addition, there are several methods that accept regular expressions (regexps) to examine the content of each string element, and follow some of the API conventions of Python's built-in `re` module:

        | Method    | Description |
        |-----------|-------------|
        | `match`   | Calls `re.match` on each element, returning a Boolean. |
        | `extract` | Calls `re.match` on each element, returning matched groups as strings.|
        | `findall` | Calls `re.findall` on each element |
        | `replace` | Replaces occurrences of pattern with some other string|
        | `contains`| Calls `re.search` on each element, returning a boolean |
        | `count`   | Counts occurrences of pattern|
        | `split`   | Equivalent to `str.split`, but accepts regexps |
        | `rsplit`  | Equivalent to `str.rsplit`, but accepts regexps |
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        With these, we can do a wide range of operations.
        For example, we can extract the first name from each element by asking for a contiguous group of characters at the beginning of each element:
        """
    )
    return


@app.cell
def _(monte):
    monte.str.extract('([A-Za-z]+)', expand=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Or we can do something more complicated, like finding all names that start and end with a consonant, making use of the start-of-string (`^`) and end-of-string (`$`) regular expression characters:
        """
    )
    return


@app.cell
def _(monte):
    monte.str.findall(r'^[^AEIOU].*[^aeiou]$')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The ability to concisely apply regular expressions across `Series` or `DataFrame` entries opens up many possibilities for analysis and cleaning of data.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Miscellaneous Methods
        Finally, there are some miscellaneous methods that enable other convenient operations:

        | Method | Description |
        |--------|-------------|
        | `get` | Indexes each element |
        | `slice` | Slices each element|
        | `slice_replace` | Replaces slice in each element with the passed value|
        | `cat`      | Concatenates strings|
        | `repeat` | Repeats values |
        | `normalize` | Returns Unicode form of strings |
        | `pad` | Adds whitespace to left, right, or both sides of strings|
        | `wrap` | Splits long strings into lines with length less than a given width|
        | `join` | Joins strings in each element of the `Series` with the passed separator|
        | `get_dummies` | Extracts dummy variables as a `DataFrame` |
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Vectorized item access and slicing

        The `get` and `slice` operations, in particular, enable vectorized element access from each array.
        For example, we can get a slice of the first three characters of each array using `str.slice(0, 3)`.
        Note that this behavior is also available through Python's normal indexing syntax; for example, `df.str.slice(0, 3)` is equivalent to `df.str[0:3]`:
        """
    )
    return


@app.cell
def _(monte):
    monte.str[0:3]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Indexing via `df.str.get(i)` and `df.str[i]` are likewise similar.

        These indexing methods also let you access elements of arrays returned by `split`.
        For example, to extract the last name of each entry, we can combine `split` with `str` indexing:
        """
    )
    return


@app.cell
def _(monte):
    monte.str.split().str[-1]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Indicator variables

        Another method that requires a bit of extra explanation is the `get_dummies` method.
        This is useful when your data has a column containing some sort of coded indicator.
        For example, we might have a dataset that contains information in the form of codes, such as A = "born in America," B = "born in the United Kingdom," C = "likes cheese," D = "likes spam":
        """
    )
    return


@app.cell
def _(monte, pd):
    full_monte = pd.DataFrame({'name': monte,
                               'info': ['B|C|D', 'B|D', 'A|C',
                                        'B|D', 'B|C', 'B|C|D']})
    full_monte
    return (full_monte,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The `get_dummies` routine lets us split out these indicator variables into a `DataFrame`:
        """
    )
    return


@app.cell
def _(full_monte):
    full_monte['info'].str.get_dummies('|')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        With these operations as building blocks, you can construct an endless range of string processing procedures when cleaning your data.

        We won't dive further into these methods here, but I encourage you to read through ["Working with Text Data"](https://pandas.pydata.org/pandas-docs/stable/user_guide/text.html) in the Pandas online documentation, or to refer to the resources listed in [Further Resources](03.13-Further-Resources.ipynb).
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Example: Recipe Database

        These vectorized string operations become most useful in the process of cleaning up messy, real-world data.
        Here I'll walk through an example of that, using an open recipe database compiled from various sources on the web.
        Our goal will be to parse the recipe data into ingredient lists, so we can quickly find a recipe based on some ingredients we have on hand. The scripts used to compile this can be found at https://github.com/fictivekin/openrecipes, and the link to the most recent version of the database is found there as well.

        This database is about 30 MB, and can be downloaded and unzipped with these commands:
        """
    )
    return


@app.cell
def _():
    # repo = "https://raw.githubusercontent.com/jakevdp/open-recipe-data/master"
    # !cd data && curl -O {repo}/recipeitems.json.gz
    # !gunzip data/recipeitems.json.gz
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The database is in JSON format, so we will use `pd.read_json` to read it (`lines=True` is required for this dataset because each line of the file is a JSON entry):
        """
    )
    return


@app.cell
def _(pd):
    recipes = pd.read_json('data/recipeitems.json', lines=True)
    recipes.shape
    return (recipes,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We see there are nearly 175,000 recipes, and 17 columns.
        Let's take a look at one row to see what we have:
        """
    )
    return


@app.cell
def _(recipes):
    recipes.iloc[0]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        There is a lot of information there, but much of it is in a very messy form, as is typical of data scraped from the web.
        In particular, the ingredient list is in string format; we're going to have to carefully extract the information we're interested in.
        Let's start by taking a closer look at the ingredients:
        """
    )
    return


@app.cell
def _(recipes):
    recipes.ingredients.str.len().describe()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The ingredient lists average 250 characters long, with a minimum of 0 and a maximum of nearly 10,000 characters!

        Just out of curiosity, let's see which recipe has the longest ingredient list:
        """
    )
    return


@app.cell
def _(np, recipes):
    recipes.name[np.argmax(recipes.ingredients.str.len())]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We can do other aggregate explorations; for example, we can see how many of the recipes are for breakfast foods (using regular expression syntax to match both lowercase and capital letters):
        """
    )
    return


@app.cell
def _(recipes):
    recipes.description.str.contains('[Bb]reakfast').sum()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Or how many of the recipes list cinnamon as an ingredient:
        """
    )
    return


@app.cell
def _(recipes):
    recipes.ingredients.str.contains('[Cc]innamon').sum()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We could even look to see whether any recipes misspell the ingredient as "cinamon":
        """
    )
    return


@app.cell
def _(recipes):
    recipes.ingredients.str.contains('[Cc]inamon').sum()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        This is the type of data exploration that is possible with Pandas string tools.
        It is data munging like this that Python really excels at.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### A Simple Recipe Recommender

        Let's go a bit further, and start working on a simple recipe recommendation system: given a list of ingredients, we want to find any recipes that use all those ingredients.
        While conceptually straightforward, the task is complicated by the heterogeneity of the data: there is no easy operation, for example, to extract a clean list of ingredients from each row.
        So, we will cheat a bit: we'll start with a list of common ingredients, and simply search to see whether they are in each recipe's ingredient list.
        For simplicity, let's just stick with herbs and spices for the time being:
        """
    )
    return


@app.cell
def _():
    spice_list = ['salt', 'pepper', 'oregano', 'sage', 'parsley',
                  'rosemary', 'tarragon', 'thyme', 'paprika', 'cumin']
    return (spice_list,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We can then build a Boolean `DataFrame` consisting of `True` and `False` values, indicating whether each ingredient appears in the list:
        """
    )
    return


@app.cell
def _(pd, recipes, spice_list):
    import re
    spice_df = pd.DataFrame({
        spice: recipes.ingredients.str.contains(spice, re.IGNORECASE)
        for spice in spice_list})
    spice_df.head()
    return re, spice_df


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Now, as an example, let's say we'd like to find a recipe that uses parsley, paprika, and tarragon.
        We can compute this very quickly using the `query` method of ``DataFrame``s, discussed further in [High-Performance Pandas: `eval()` and `query()`](03.12-Performance-Eval-and-Query.ipynb):
        """
    )
    return


@app.cell
def _(spice_df):
    selection = spice_df.query('parsley & paprika & tarragon')
    len(selection)
    return (selection,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We find only 10 recipes with this combination. Let's use the index returned by this selection to discover the names of those recipes:
        """
    )
    return


@app.cell
def _(recipes, selection):
    recipes.name[selection.index]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Now that we have narrowed down our recipe selection from 175,000 to 10, we are in a position to make a more informed decision about what we'd like to cook for dinner.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Going Further with Recipes

        Hopefully this example has given you a bit of a flavor (heh) of the types of data cleaning operations that are efficiently enabled by Pandas string methods.
        Of course, building a robust recipe recommendation system would require a *lot* more work!
        Extracting full ingredient lists from each recipe would be an important piece of the task; unfortunately, the wide variety of formats used makes this a relatively time-consuming process.
        This points to the truism that in data science, cleaning and munging of real-world data often comprises the majority of the work—and Pandas provides the tools that can help you do this efficiently.
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
