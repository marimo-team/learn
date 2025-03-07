# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "altair==5.5.0",
#     "marimo",
#     "numpy==2.2.3",
#     "polars==1.24.0",
# ]
# ///

import marimo

__generated_with = "0.11.17"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
        # Strings

        _By [Péter Ferenc Gyarmati](http://github.com/peter-gy)_.

        In this chapter we're going to dig into string manipulation. For a fun twist, we'll be mostly playing around with a dataset that every Polars user has bumped into without really thinking about it—the source code of the `polars` module itself. More precisely, we'll use a dataframe that pulls together all the Polars expressions and their docstrings, giving us a cool, hands-on way to explore the expression API in a truly data-driven manner.

        We'll cover parsing, length calculation, case conversion, and much more, with practical examples and visualizations. Finally, we will combine various techniques you learned in prior chapters to build a fully interactive playground in which you can execute the official code examples of Polars expressions.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ##  🛠️  Parsing & Conversion

        Let's warm up with one of the most frequent use cases: parsing raw strings into various formats.
        We'll take a tiny dataframe with metadata about Python packages represented as raw JSON strings and we'll use Polars string expressions to parse the attributes into their true data types.
        """
    )
    return


@app.cell(hide_code=True)
def _(pl):
    pip_metadata_raw_df = pl.DataFrame(
        [
            '{"package": "polars", "version": "1.24.0", "released_at": "2025-03-02T20:31:12+0000", "size_mb": "30.9"}',
            '{"package": "marimo", "version": "0.11.14", "released_at": "2025-03-04T00:28:57+0000", "size_mb": "10.7"}',
        ],
        schema={"raw_json": pl.String},
    )
    pip_metadata_raw_df
    return (pip_metadata_raw_df,)


@app.cell
def _(mo):
    mo.md(r"""We can use the [`json_decode`](https://docs.pola.rs/api/python/stable/reference/series/api/polars.Series.str.json_decode.html) expression to parse the raw JSON strings into Polars-native structs and we can use the [unnest](https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.unnest.html) dataframe operation to have a dedicated column per parsed attribute.""")
    return


@app.cell
def _(pip_metadata_raw_df, pl):
    pip_metadata_df = pip_metadata_raw_df.select(json=pl.col('raw_json').str.json_decode()).unnest('json')
    pip_metadata_df
    return (pip_metadata_df,)


@app.cell
def _(mo):
    mo.md(r"""This is already a much friendlier representation of the data we started out with, but note that since the JSON entries had only string attributes, all values are strings, even the temporal `released_at` and numerical `size_mb` columns.""")
    return


@app.cell
def _(mo):
    mo.md(r"""As we know that the `size_mb` column should have a decimal representation, we go ahead and use [`to_decimal`](https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.Expr.str.to_decimal.html#polars.Expr.str.to_decimal) to perform the conversion.""")
    return


@app.cell
def _(pip_metadata_df, pl):
    pip_metadata_df.select(
        'package',
        'version',
        pl.col('size_mb').str.to_decimal(),
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        Moving on to the `released_at` attribute which indicates the exact time when a given Python package got released, we have a bit more options to consider. We can convert to `Date`, `DateTime`, and `Time` types based on the desired temporal granularity. The [`to_date`](https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.Expr.str.to_date.html), [`to_datetime`](https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.Expr.str.to_datetime.html), and [`to_time`](https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.Expr.str.to_time.html) expressions are here to help us with the conversion, all we need is to provide the desired format string.

        Since Polars uses Rust under the hood to implement all its expressions, we need to consult the [`chrono::format`](https://docs.rs/chrono/latest/chrono/format/strftime/index.html) reference to come up with appropriate format strings.

        Here's a quick reference:

        | Specifier | Meaning            |
        |-----------|--------------------|
        | `%Y`      | Year (e.g., 2025) |
        | `%m`      | Month (01-12)     |
        | `%d`      | Day (01-31)       |
        | `%H`      | Hour (00-23)      |
        | `%z`      | UTC offset        |

        The raw strings we are working with look like `"2025-03-02T20:31:12+0000"`. We can match this using the `"%Y-%m-%dT%H:%M:%S%z"` format string.
        """
    )
    return


@app.cell
def _(pip_metadata_df, pl):
    pip_metadata_df.select(
        'package',
        'version',
        release_date=pl.col('released_at').str.to_date('%Y-%m-%dT%H:%M:%S%z'),
        release_datetime=pl.col('released_at').str.to_datetime('%Y-%m-%dT%H:%M:%S%z'),
        release_time=pl.col('released_at').str.to_time('%Y-%m-%dT%H:%M:%S%z'),
    )
    return


@app.cell
def _(mo):
    mo.md(r"""Alternatively, instead of using three different functions to perform the conversion to date, we can use a single one, [`strptime`](https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.Expr.str.strptime.html) which takes the desired temporal data type as its first parameter.""")
    return


@app.cell
def _(pip_metadata_df, pl):
    pip_metadata_df.select(
        'package',
        'version',
        release_date=pl.col('released_at').str.strptime(pl.Date, '%Y-%m-%dT%H:%M:%S%z'),
        release_datetime=pl.col('released_at').str.strptime(pl.Datetime, '%Y-%m-%dT%H:%M:%S%z'),
        release_time=pl.col('released_at').str.strptime(pl.Time, '%Y-%m-%dT%H:%M:%S%z'),
    )
    return


@app.cell
def _(mo):
    mo.md(r"""And to wrap up this section on parsing and conversion, let's consider a final scenario. What if we don't want to parse the entire raw JSON string, because we only need a subset of its attributes? Well, in this case we can leverage the [`json_path_match`](https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.Expr.str.json_path_match.html) expression to extract only the desired attributes using standard [JSONPath](https://goessner.net/articles/JsonPath/) syntax.""")
    return


@app.cell
def _(pip_metadata_raw_df, pl):
    pip_metadata_raw_df.select(
        package=pl.col("raw_json").str.json_path_match("$.package"),
        version=pl.col("raw_json").str.json_path_match("$.version"),
        release_date=pl.col("raw_json")
        .str.json_path_match("$.size_mb")
        .str.to_decimal(),
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 📊 Dataset Overview

        Now that we got our hands dirty, let's consider a somewhat wilder dataset for the subsequent sections: a dataframe of metadata about every single expression in your current Polars module.

        At the risk of stating the obvious, in the previous section, when we typed `pl.col('raw_json').str.json_decode()`, we accessed the `json_decode` member of the `str` expression namespace through the `pl.col('raw_json')` expression *instance*. Under the hood, deep inside the Polars source code, there is a corresponding `def json_decode(...)` method with a carefully authored docstring explaining the purpose and signature of the member.

        Since Python makes module introspection simple, we can easily enumerate all Polars expressions and organize their metadata in `expressions_df`, to be used for all the upcoming string manipulation examples.
        """
    )
    return


@app.cell(hide_code=True)
def _(pl):
    def list_members(expr, namespace) -> list[dict]:
        """Iterates through the attributes of `expr` and returns their metadata"""
        members = []
        for attrname in expr.__dir__():
            is_namespace = attrname in pl.Expr._accessors
            is_private = attrname.startswith("_")
            if is_namespace or is_private:
                continue

            attr = getattr(expr, attrname)
            members.append(
                {
                    "namespace": namespace,
                    "member": attrname,
                    "docstring": attr.__doc__,
                }
            )
        return members


    def list_expr_meta() -> list[dict]:
        # Dummy expression instance to 'crawl'
        expr = pl.lit("")
        root_members = list_members(expr, "root")
        namespaced_members: list[list[dict]] = [
            list_members(getattr(expr, namespace), namespace)
            for namespace in pl.Expr._accessors
        ]
        return sum(namespaced_members, root_members)


    expressions_df = pl.from_dicts(list_expr_meta(), infer_schema_length=None).sort('namespace', 'member')
    expressions_df
    return expressions_df, list_expr_meta, list_members


@app.cell
def _(mo):
    mo.md(r"""As the following visualization shows, `str` is one of the richest Polars expression namespaces with multiple dozens of functions in it.""")
    return


@app.cell(hide_code=True)
def _(alt, expressions_df):
    expressions_df.plot.bar(
        x=alt.X("count(member):Q", title='Count of Expressions'),
        y=alt.Y("namespace:N", title='Namespace').sort("-x"),
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 📏 Length Calculation

        A common use case is to compute the length of a string. Most people associate string length exclusively with the number of characters the said string consists of; however, in certain scenarios it is useful to also know how much memory is required for storing, so how many bytes are required to represent the textual data.

        The expressions [`len_chars`](https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.Expr.str.len_chars.html) and [`len_bytes`](https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.Expr.str.len_bytes.html) are here to help us with these calculations.

        Below, we compute `docstring_len_chars` and `docstring_len_bytes` columns to see how many characters and bytes the documentation of each expression is made up of.
        """
    )
    return


@app.cell
def _(expressions_df, pl):
    docstring_length_df = expressions_df.with_columns(
        docstring_len_chars=pl.col("docstring").str.len_chars(),
        docstring_len_bytes=pl.col("docstring").str.len_bytes(),
    )
    docstring_length_df
    return (docstring_length_df,)


@app.cell
def _(mo):
    mo.md(r"""As the dataframe preview above and the scatterplot below show, the docstring length measured in bytes is almost always bigger than the length expressed in characters. This is due to the fact that the docstrings include characters which require more than a single byte to represent, such as "╞" for displaying dataframe header and body separators.""")
    return


@app.cell
def _(alt, docstring_length_df):
    docstring_length_df.plot.point(
        x=alt.X('docstring_len_chars', title='Docstring Length (Chars)'),
        y=alt.Y('docstring_len_bytes', title='Docstring Length (Bytes)'),
        tooltip=['namespace', 'member', 'docstring_len_chars', 'docstring_len_bytes'],
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 🔠 Case Conversion

        Another frequent string transformation is lowercasing, uppercasing, and titlecasing. We can use [`to_lowercase`](https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.Expr.str.to_lowercase.html), [`to_uppercase`](https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.Expr.str.to_lowercase.html) and [`to_titlecase`](https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.Expr.str.to_titlecase.html) for doing so.
        """
    )
    return


@app.cell
def _(expressions_df, pl):
    expressions_df.select(
        member_lower=pl.col('member').str.to_lowercase(),
        member_upper=pl.col('member').str.to_uppercase(),
        member_title=pl.col('member').str.to_titlecase(),
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## ➕ Padding

        Sometimes we need to ensure that strings have a fixed-size character length. [`pad_start`](https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.Expr.str.pad_start.html) and [`pad_end`](https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.Expr.str.pad_end.html) can be used to fill the "front" or "back" of a string with a supplied character, while [`zfill`](https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.Expr.str.zfill.html) is a utility for padding the start of a string with `"0"` until it reaches a particular length. In other words, `zfill` is a more specific version of `pad_start`, where the `fill_char` parameter is explicitly set to `"0"`.

        In the example below we take the unique Polars expression namespaces and pad them so that they have a uniform length which you can control via a slider.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    padding = mo.ui.slider(0, 16, step=1, value=8, label="Padding Size")
    return (padding,)


@app.cell
def _(expressions_df, padding, pl):
    padded_df = expressions_df.select("namespace").unique().select(
        "namespace",
        namespace_front_padded=pl.col("namespace").str.pad_start(padding.value, "_"),
        namespace_back_padded=pl.col("namespace").str.pad_end(padding.value, "_"),
        namespace_zfilled=pl.col("namespace").str.zfill(padding.value),
    )
    return (padded_df,)


@app.cell(hide_code=True)
def _(mo, padded_df, padding):
    mo.vstack([
        padding,
        padded_df,
    ])
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 🔄 Replacing

        Let's say we want to convert from `snake_case` API member names to `kebab-case`, that is, we need to replace the underscore character with a hyphen. For operations like that, we can use [`replace`](https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.Expr.str.replace.html) and [`replace_all`](https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.Expr.str.replace_all.html).

        As the example below demonstrates, `replace` stops after the first occurrence of the to-be-replaced pattern, while `replace_all` goes all the way through and changes all underscores to hyphens resulting in the `kebab-case` representation we were looking for.
        """
    )
    return


@app.cell
def _(expressions_df, pl):
    expressions_df.select(
        "member",
        member_kebab_case_partial=pl.col("member").str.replace("_", "-"),
        member_kebab_case=pl.col("member").str.replace_all("_", "-"),
    ).sort(pl.col("member").str.len_chars(), descending=True)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        A related expression is [`replace_many`](https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.Expr.str.replace_many.html), which accepts *many* pairs of to-be-matched patterns and corresponding replacements and uses the [Aho–Corasick algorithm](https://en.wikipedia.org/wiki/Aho%E2%80%93Corasick_algorithm) to carry out the operation with great performance.

        In the example below we replace all instances of `"min"` with `"minimum"` and `"max"` with `"maximum"` using a single expression.
        """
    )
    return


@app.cell
def _(expressions_df, pl):
    expressions_df.select(
        "member",
        member_modified=pl.col("member").str.replace_many(
            {
                "min": "minimum",
                "max": "maximum",
            }
        ),
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 🔍 Searching & Matching

        A common need when working with strings is to determine whether their content satisfies some condition: whether it starts or ends with a particular substring or contains a certain pattern.

        Let's suppose we want to determine whether a member of the Polars expression API is a "converter", such as `to_decimal`, identified by its `"to_"` prefix. We can use [`starts_with`](https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.Expr.str.starts_with.html) to perform this check.
        """
    )
    return


@app.cell
def _(expressions_df, pl):
    expressions_df.select(
        "namespace",
        "member",
        is_converter=pl.col("member").str.starts_with("to_"),
    ).sort(-pl.col("is_converter").cast(pl.Int8))
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        Throughout this course as you have gained familiarity with the expression API you might have noticed that some members end with an underscore such as `or_`, since their "body" is a reserved Python keyword.

        Let's use [`ends_with`](https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.Expr.str.ends_with.html) to find all the members which are named after such keywords.
        """
    )
    return


@app.cell
def _(expressions_df, pl):
    expressions_df.select(
        "namespace",
        "member",
        is_escaped_keyword=pl.col("member").str.ends_with("_"),
    ).sort(-pl.col("is_escaped_keyword").cast(pl.Int8))
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        Now let's move on to analyzing the docstrings in a bit more detail. Based on their content we can determine whether a member is deprecated, accepts parameters, comes with examples, or references external URL(s) & related members.

        As demonstrated below, we can compute all these boolean attributes using [`contains`](https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.Expr.str.contains.html) to check whether the docstring includes a particular substring.
        """
    )
    return


@app.cell
def _(expressions_df, pl):
    expressions_df.with_columns(
        is_deprecated=pl.col('docstring').str.contains('.. deprecated', literal=True),
        has_parameters=pl.col('docstring').str.contains('Parameters'),
        has_examples=pl.col('docstring').str.contains('Examples'),
        has_related_members=pl.col('docstring').str.contains('See Also'),
        has_url=pl.col('docstring').str.contains('https?://'),
    )
    return


@app.cell
def _(mo):
    mo.md(r"""For scenarios where we want to combine multiple substrings to check for, we can use the [`contains`](https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.Expr.str.contains.html) expression to check for the presence of various patterns.""")
    return


@app.cell
def _(expressions_df, pl):
    expressions_df.with_columns(
        has_reference=pl.col('docstring').str.contains_any(['See Also', 'https://'])
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        From the above analysis we could see that almost all the members come with code examples. It would be interesting to know how many variable assignments are going on within each of these examples, right? That's not as simple as checking for a pre-defined literal string containment though, because variables can have arbitrary names - any valid Python identifier is allowed. While the `contains` function supports checking for regular expressions instead of literal strings too, it would not suffice for this exercise because it only tells us whether there is at least a single occurrence of the sought pattern rather than telling us the exact number of matches.

        Fortunately, we can take advantage of [`count_matches`](https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.Expr.str.count_matches.html) to achieve exactly what we want. We specify the regular expression `r'[a-zA-Z_][a-zA-Z0-9_]* = '` according to the [`regex` Rust crate](https://docs.rs/regex/latest/regex/) to match Python identifiers and we leave the rest to Polars.

        In `count_matches(r'[a-zA-Z_][a-zA-Z0-9_]* = ')`:

        - `[a-zA-Z_]` matches a letter or underscore (start of a Python identifier).
        - `[a-zA-Z0-9_]*` matches zero or more letters, digits, or underscores.
        - ` = ` matches a space, equals sign, and space (indicating assignment).

        This finds variable assignments like `x = ` or `df_result = ` in docstrings.
        """
    )
    return


@app.cell
def _(expressions_df, pl):
    expressions_df.with_columns(
        variable_assignment_count=pl.col('docstring').str.count_matches(r'[a-zA-Z_][a-zA-Z0-9_]* = '),
    )
    return


@app.cell
def _(mo):
    mo.md(r"""A related application example is to *find* the first index where a particular pattern is present, so that it can be used for downstream processing such as slicing. Below we use the [`find`](https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.Expr.str.find.html) expression to determine the index at which a code example starts in the docstring - identified by the Python shell substring `">>>"`.""")
    return


@app.cell
def _(expressions_df, pl):
    expressions_df.with_columns(
        code_example_start=pl.col('docstring').str.find('>>>'),
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## ✂️ Slicing and Substrings

        Sometimes we are only interested in a particular substring. We can use [`head`](https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.Expr.str.head.html), [`tail`](https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.Expr.str.tail.html) and [`slice`](https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.Expr.str.slice.html) to extract a substring from the start, end, or between arbitrary indices.
        """
    )
    return


@app.cell
def _(mo):
    slice = mo.ui.slider(1, 50, step=1, value=25, label="Slice Size")
    return (slice,)


@app.cell
def _(expressions_df, pl, slice):
    sliced_df = expressions_df.select(
        # Original string
        "docstring",
        # First 25 chars
        docstring_head=pl.col("docstring").str.head(slice.value),
        # 50 chars after the first 25 chars
        docstring_slice=pl.col("docstring").str.slice(slice.value, 2*slice.value),
        # Last 25 chars
        docstring_tail=pl.col("docstring").str.tail(slice.value),
    )
    return (sliced_df,)


@app.cell
def _(mo, slice, sliced_df):
    mo.vstack([
        slice,
        sliced_df,
    ])
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## ➗ Splitting

        Certain strings follow a well-defined structure and we might be only interested in some parts of them. For example, when dealing with `snake_cased_expression` member names we might be curious to get only the first, second, or $n^{\text{th}}$ word before an underscore. We would need to *split* the string at a particular pattern for downstream processing.

        The [`split`](https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.Expr.str.split.html), [`split_exact`](https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.Expr.str.split_exact.html) and [`splitn`](https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.Expr.str.splitn.html) expressions enable us to achieve this.

        The primary difference between these string splitting utilities is that `split` produces a list of variadic length based on the number of resulting segments, `splitn` returns a struct with at least `0` and at most `n` fields while `split_exact` returns a struct of exactly `n` fields.
        """
    )
    return


@app.cell
def _(expressions_df, pl):
    expressions_df.select(
        'member',
        member_name_parts=pl.col('member').str.split('_'),
        member_name_parts_n=pl.col('member').str.splitn('_', n=2),
        member_name_parts_exact=pl.col('member').str.split_exact('_', n=2),
    )
    return


@app.cell
def _(mo):
    mo.md(r"""As a more practical example, we can use the `split` expression with some aggregation to count the number of times a particular word occurs in member names across all namespaces. This enables us to create a word cloud of the API members' constituents!""")
    return


@app.cell(hide_code=True)
def _(mo, wordcloud, wordcloud_height, wordcloud_width):
    mo.vstack([
        wordcloud_width,
        wordcloud_height,
        wordcloud,
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    wordcloud_width = mo.ui.slider(0, 64, step=1, value=32, label="Word Cloud Width")
    wordcloud_height = mo.ui.slider(0, 32, step=1, value=16, label="Word Cloud Height")
    return wordcloud_height, wordcloud_width


@app.cell(hide_code=True)
def _(alt, expressions_df, pl, random, wordcloud_height, wordcloud_width):
    wordcloud_df = (
        expressions_df.select(pl.col("member").str.split("_"))
        .explode("member")
        .group_by("member")
        .agg(pl.len())
        # Generating random x and y coordinates to distribute the words in the 2D space
        .with_columns(
            x=pl.col("member").map_elements(
                lambda e: random.randint(0, wordcloud_width.value),
                return_dtype=pl.UInt8,
            ),
            y=pl.col("member").map_elements(
                lambda e: random.randint(0, wordcloud_height.value),
                return_dtype=pl.UInt8,
            ),
        )
    )

    wordcloud = alt.Chart(wordcloud_df).mark_text(baseline="middle").encode(
        x=alt.X("x:O", axis=None),
        y=alt.Y("y:O", axis=None),
        text="member:N",
        color=alt.Color("len:Q", scale=alt.Scale(scheme="bluepurple")),
        size=alt.Size("len:Q", legend=None),
        tooltip=["member", "len"],
    ).configure_view(strokeWidth=0)
    return wordcloud, wordcloud_df


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 🔗 Concatenation & Joining

        Often we would like to create longer strings from strings we already have. We might want to create a formatted, sentence-like string or join multiple existing strings in our dataframe into a single one.

        The top-level [`concat_str`](https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.concat_str.html) expression enables us to combine strings *horizontally* in a dataframe. As the example below shows, we can take the `member` and `namespace` column of each row and construct a `description` column in which each row will correspond to the value ``f"- Expression `{member}` belongs to namespace `{namespace}`"``.
        """
    )
    return


@app.cell
def _(expressions_df, pl):
    descriptions_df = expressions_df.select(
        description=pl.concat_str(
            [
                pl.lit("- Expression "),
                pl.lit("`"),
                "member",
                pl.lit("`"),
                pl.lit(" belongs to namespace "),
                pl.lit("`"),
                "namespace",
                pl.lit("`"),
            ],
        )
    )
    descriptions_df
    return (descriptions_df,)


@app.cell
def _(mo):
    mo.md(
        r"""
        Now that we have constructed these bullet points through *horizontal* concatenation of strings, we can perform a *vertical* one so that we end up with a single string in which we have a bullet point on each line.

        We will use the [`join`](https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.join.html) expression to do so.
        """
    )
    return


@app.cell
def _(descriptions_df, pl):
    descriptions_df.select(pl.col('description').str.join('\n'))
    return


@app.cell(hide_code=True)
def _(descriptions_df, mo, pl):
    mo.md(f"""In fact, since the string we constructed dynamically is valid markdown, we can display it dynamically using Marimo's `mo.md` utility!

    ---

    {descriptions_df.select(pl.col('description').str.join('\n')).to_numpy().squeeze().tolist()}
    """)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 🔍 Pattern-based Extraction

        In the vast majority of the cases, when dealing with unstructured text data, all we really want is to extract something structured from it. A common use case is to extract URLs from text to get a better understanding of related content.

        In the example below that's exactly what we do. We scan the `docstring` of each API member and extract URLs from them using [`extract`](https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.Expr.str.extract.html) and [`extract_all`](https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.Expr.str.extract_all.html) using a simple regular expression to match http and https URLs.

        Note that `extract` stops after a first match and returns a scalar result (or `null` if there was no match) while `extract_all` returns a - potentially empty - list of matches.
        """
    )
    return


@app.cell
def _(expressions_df, pl):
    url_pattern = r'(https?://[^\s>]+)'
    expressions_df.with_columns(
        "docstring",
        url_match=pl.col('docstring').str.extract(url_pattern),
        url_matches=pl.col('docstring').str.extract_all(url_pattern),
    ).filter(pl.col('url_match').is_not_null())
    return (url_pattern,)


@app.cell
def _(mo):
    mo.md(
        r"""
        Note that in each `docstring` where a code example involving dataframes is present, we will see an output such as "shape: (5, 2)" indicating the number of rows and columns of the dataframe produced by the sample code. Let's say we would like to *capture* this information in a structured way.

        [`extract_groups`](https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.Expr.str.extract_groups.html) is a really powerful expression allowing us to achieve exactly that.

        Below we define the regular expression `r"shape:\s*\((?<height>\S+),\s*(?<width>\S+)\)"` with two capture groups, named `height` and `width` and pass it as the parameter of `extract_groups`. After execution, for each `docstring`, we end up with fully structured data we can further process downstream!
        """
    )
    return


@app.cell
def _(expressions_df, pl):
    expressions_df.with_columns(
        example_df_shape=pl.col('docstring').str.extract_groups(r"shape:\s*\((?<height>\S+),\s*(?<width>\S+)\)"),
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 🧹 Stripping

        Strings might require some cleaning before further processing, such as the removal of some characters from the beginning or end of the text. [`strip_chars_start`](https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.Expr.str.strip_chars_start.html), [`strip_chars_end`](https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.Expr.str.strip_chars_end.html) and [`strip_chars`](https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.Expr.str.strip_chars.html) are here to facilitate this.

        All we need to do is to specify a set of characters we would like to get rid of and Polars handles the rest for us.
        """
    )
    return


@app.cell
def _(expressions_df, pl):
    expressions_df.select(
        "member",
        member_front_stripped=pl.col("member").str.strip_chars_start("a"),
        member_back_stripped=pl.col("member").str.strip_chars_end("n"),
        member_fully_stripped=pl.col("member").str.strip_chars("na"),
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        Note that when using the above expressions, the specified characters do not need to form a sequence; they are handled as a set. However, in certain use cases we only want to strip complete substrings, so we would need our input to be strictly treated as a sequence rather than as a set.

        That's exactly the rationale behind [`strip_prefix`](https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.Expr.str.strip_prefix.html) and [`strip_suffix`](https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.Expr.str.strip_suffix.html).

        Below we use these to remove the `"to_"` prefixes and `"_with"` suffixes from each member name.
        """
    )
    return


@app.cell
def _(expressions_df, pl):
    expressions_df.select(
        "member",
        member_prefix_stripped=pl.col("member").str.strip_prefix("to_"),
        member_suffix_stripped=pl.col("member").str.strip_suffix("_with"),
    ).slice(20)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 🔑 Encoding & Decoding

        Should you find yourself in the need of encoding your strings into [base64](https://en.wikipedia.org/wiki/Base64) or [hexadecimal](https://en.wikipedia.org/wiki/Hexadecimal) format, then Polars has your back with its [`encode`](https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.Expr.str.encode.html) expression.
        """
    )
    return


@app.cell
def _(expressions_df, pl):
    encoded_df = expressions_df.select(
        "member",
        member_base64=pl.col('member').str.encode('base64'),
        member_hex=pl.col('member').str.encode('hex'),
    )
    encoded_df
    return (encoded_df,)


@app.cell
def _(mo):
    mo.md(r"""And of course, you can convert back into a human-readable representation using the [`decode`](https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.Expr.str.decode.html) expression.""")
    return


@app.cell
def _(encoded_df, pl):
    encoded_df.with_columns(
        member_base64_decoded=pl.col('member_base64').str.decode('base64').cast(pl.String),
        member_hex_decoded=pl.col('member_hex').str.decode('hex').cast(pl.String),
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 🚀 Application: Dynamic Execution of Polars Examples

        Now that we are familiar with string expressions, we can combine them with other Polars operations to build a fully interactive playground where code examples of Polars expressions can be explored.

        We make use of string expressions to extract the raw Python source code of examples from the docstrings and we leverage the interactive Marimo environment to enable the selection of expressions via a searchable dropdown and a fully functional code editor whose output is rendered with Marimo's rich display utilities.

        In other words, we will use Polars to execute Polars. ❄️ How cool is that?
        """
    )
    return


@app.cell(hide_code=True)
def _(
    example_editor,
    execution_result,
    expression,
    expression_description,
    expression_docs_link,
    mo,
):
    mo.vstack(
        [
            expression,
            mo.hstack([expression_description, expression_docs_link]),
            example_editor,
            execution_result,
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo, selected_expression_record):
    expression_description = mo.md(selected_expression_record["description"])
    expression_docs_link = mo.md(
        f"🐻‍❄️ [Official Docs](https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.Expr.{selected_expression_record['expr']}.html)"
    )
    return expression_description, expression_docs_link


@app.cell(hide_code=True)
def _(example_editor, execute_code):
    execution_result = execute_code(example_editor.value)
    return (execution_result,)


@app.cell(hide_code=True)
def _(code_df, mo):
    expression = mo.ui.dropdown(code_df.get_column('expr'), value='arr.all', searchable=True)
    return (expression,)


@app.cell(hide_code=True)
def _(code_df, expression):
    selected_expression_record = code_df.filter(expr=expression.value).to_dicts()[0]
    return (selected_expression_record,)


@app.cell(hide_code=True)
def _(mo, selected_expression_record):
    example_editor = mo.ui.code_editor(value=selected_expression_record["code"])
    return (example_editor,)


@app.cell(hide_code=True)
def _(expressions_df, pl):
    code_df = (
        expressions_df.select(
            expr=pl.when(pl.col("namespace") == "root")
            .then("member")
            .otherwise(pl.concat_str(["namespace", "member"], separator=".")),
            description=pl.col("docstring")
            .str.split("\n\n")
            .list.get(0)
            .str.slice(9),
            docstring_lines=pl.col("docstring").str.split("\n"),
        )
        .with_row_index()
        .explode("docstring_lines")
        .rename({"docstring_lines": "docstring_line"})
        .with_columns(pl.col("docstring_line").str.strip_chars(" "))
        .filter(pl.col("docstring_line").str.contains_any([">>> ", "... "]))
        .with_columns(pl.col("docstring_line").str.slice(4))
        .group_by(pl.exclude("docstring_line"), maintain_order=True)
        .agg(code=pl.col("docstring_line").str.join("\n"))
        .drop("index")
    )
    return (code_df,)


@app.cell(hide_code=True)
def _():
    def execute_code(code: str):
        import ast

        # Create a new local namespace for execution
        local_namespace = {}

        # Parse the code into an AST to identify the last expression
        parsed_code = ast.parse(code)

        # Check if there's at least one statement
        if not parsed_code.body:
            return None

        # If the last statement is an expression, we'll need to get its value
        last_is_expr = isinstance(parsed_code.body[-1], ast.Expr)

        if last_is_expr:
            # Split the code: everything except the last statement, and the last statement
            last_expr = ast.Expression(parsed_code.body[-1].value)

            # Remove the last statement from the parsed code
            parsed_code.body = parsed_code.body[:-1]

            # Execute everything except the last statement
            if parsed_code.body:
                exec(
                    compile(parsed_code, "<string>", "exec"),
                    globals(),
                    local_namespace,
                )

            # Execute the last statement and get its value
            result = eval(
                compile(last_expr, "<string>", "eval"), globals(), local_namespace
            )
            return result
        else:
            # If the last statement is not an expression (e.g., an assignment),
            # execute the entire code and return None
            exec(code, globals(), local_namespace)
            return None
    return (execute_code,)


@app.cell(hide_code=True)
def _():
    import polars as pl
    import marimo as mo
    import altair as alt
    import random

    random.seed(42)
    return alt, mo, pl, random


if __name__ == "__main__":
    app.run()
