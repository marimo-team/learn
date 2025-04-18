# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "plotly.express",
#     "plotly==6.0.1",
#     "duckdb==1.2.1",
#     "sqlglot==26.11.1",
#     "pyarrow==19.0.1",
#     "polars==1.27.1",
# ]
# ///

import marimo

__generated_with = "0.12.10"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#Loading CSVs with DuckDB""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        <p> I remember when I first learnt about DuckDB, it was a gamechanger — I used to load the data I wanted to work on to a database software like MS SQL Server, and then build a bridge to an IDE with the language I wanted to use like Python, or R; it was quite the hassle. DuckDB changed my whole world — now I could just import the data file into the IDE, or notebook, make a duckdb connection, and there we go! But then, I realized I didn't even need the step of first importing the file using python. I could just query the csv file directly using SQL through a DuckDB connection.</p> 

        ##Introduction
        <p> I found this dataset on the evolution of AI research by discipline from <a href= "https://oecd.ai/en/data?selectedArea=ai-research&selectedVisualization=16731"> OECD</a>, and it piqued my interest. I feel like publications in natural language processing drastically jumped in the mid 2010s, and I'm excited to find out if that's the case. </p> 

        <p> In this notebook, we'll: </p>
        <ul>
            <li> Import the CSV file into the notebook</li>
            <li> Create another table within the database based on the CSV</li>
            <li> Dig into publications on natural language processing have evolved over the years</li>
        </ul>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""##Load the CSV""")
    return


@app.cell
def _(mo):
    _df = mo.sql(
        f"""
        /* Another way to load the CSV could be 
        SELECT * 
        FROM read_csv('https://github.com/Mustjaab/Loading_CSVs_in_DuckDB/blob/main/AI_Research_Data.csv')
        */
        SELECT * 
        FROM "https://raw.githubusercontent.com/Mustjaab/Loading_CSVs_in_DuckDB/refs/heads/main/AI_Research_Data.csv"
        LIMIT 5;
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""##Create Another Table""")
    return


@app.cell
def _(mo):
    Discipline_Analysis = mo.sql(
        f"""
        -- Build a table based on the CSV where it just contains the specified columns
        CREATE TABLE Domain_Analysis AS
            SELECT Year, Concept, publications FROM "https://raw.githubusercontent.com/Mustjaab/Loading_CSVs_in_DuckDB/refs/heads/main/AI_Research_Data.csv"
        """
    )
    return Discipline_Analysis, Domain_Analysis


@app.cell
def _(Domain_Analysis, mo):
    Analysis = mo.sql(
        f"""
        SELECT * 
        FROM Domain_Analysis
        GROUP BY Concept, Year, publications
        ORDER BY Year
        """
    )
    return (Analysis,)


@app.cell
def _(Domain_Analysis, mo):
    _df = mo.sql(
        f"""
        SELECT 
            AVG(CASE WHEN Year < 2020 THEN publications END) AS avg_pre_2020,
            AVG(CASE WHEN Year >= 2020 THEN publications END) AS avg_2020_onward
        FROM Domain_Analysis
        WHERE Concept = 'Natural language processing';
        """
    )
    return


@app.cell
def _(Domain_Analysis, mo):
    NLP_Analysis = mo.sql(
        f"""
        SELECT 
            publications, 
            CASE 
                WHEN Year < 2020 THEN 'Pre-2020' 
                ELSE '2020-onward' 
            END AS period
        FROM Domain_Analysis
        WHERE Year >= 2000 
        AND Concept = 'Natural language processing';
        """,
        output=False
    )
    return (NLP_Analysis,)


@app.cell
def _(NLP_Analysis, px):
    px.box(NLP_Analysis, x='period', y='publications', color='period')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""<p> We can see there's a significant increase in NLP publications 2020 and onwards which definitely makes sense provided the rapid emergence of commercial large language models, and AI assistants. </p>""")

@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ##Conclusion
        <p> In this notebook, we learned how to:</p> 
        <ul>
            <li> Load a CSV into DuckDB </li>
            <li> Create other tables using the imported CSV </li>
            <li> Seamlessly analyze and visualize data between SQL, and Python cells</li>
        </ul>
        """
    )
    return


@app.cell
def _():
    import pyarrow
    import polars
    return polars, pyarrow


@app.cell
def _():
    import marimo as mo
    import plotly.express as px
    return mo, px


if __name__ == "__main__":
    app.run()
