# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "pandas",
#     "matplotlib",
# ]
# ///

import marimo

__generated_with = "0.13.13-dev18"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
     # 📈 Unconventional Charts with Plotly

        _By [Aarshia Gupta](https://github.com/aarshiagupta)._


        This notebook demonstrates some of Plotly's more unconventional yet insightful chart types, using the **Palmer Penguins** dataset.

        We'll cover:

        - 🧼 Waterfall charts for cumulative comparisons
        - 🧲 Sunburst charts for hierarchical exploration
        - 🔗 Sankey diagrams for flow-based analysis
        - 🔁 Parallel categories for multi-dimensional relationships

        Each section includes widgets to let you explore interactivity and insights visually.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ##🐧 Dataset Preview

        This is a snapshot of the **Palmer Penguins dataset** — a clean, fun, and rich dataset to explore species, size, and sex distributions.

    - Numerical: `flipper_length_mm`, `bill_length_mm`, `bill_depth_mm`, `body_mass_g`
    
    - Categorical: `species`, `island`, `sex`


      You can:

    - scroll across the columns,
    
    - filter by clicking on column histograms,
    
     - and download the full CSV.
    """
    )
    return


@app.cell
def _(df):
    df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 📊 Waterfall Chart: Compare Body Mass Across Species

    This **waterfall chart** lets us compare the **average body mass of each penguin species**.

    Unlike a bar chart, which shows absolute values, a waterfall chart shows **change from a baseline**—ideal for cumulative comparisons or highlighting differences in a sequence.
    """
    )
    return


@app.cell(hide_code=True)
def _(df, mo):
    species_options = df["species"].unique().tolist()
    waterfall_dropdown = mo.ui.dropdown(
        options=species_options,
        value=species_options[0],
        label="Select reference species for waterfall chart"
    )
    return (waterfall_dropdown,)


@app.cell(hide_code=True)
def _(df, go, mo, waterfall_dropdown):
    mo.md("""
    ## 🧼 Waterfall Chart: Compare Body Mass Across Species

    A **waterfall chart** is useful when you want to show how values cumulatively add up or change from a starting point.  
    In this case, we start with the average body mass of one reference species (chosen using the dropdown) and show how the other species compare.

    Use the dropdown above to choose the reference point.
    """)

    ref_species = waterfall_dropdown.value
    means = df.groupby("species")["body_mass_g"].mean().reset_index()
    means = means.sort_values("species")
    ref_mass = means[means["species"] == ref_species]["body_mass_g"].iloc[0]

    fig = go.Figure(go.Waterfall(
        x=means["species"],
        y=means["body_mass_g"] - ref_mass,
        base=ref_mass,
        measure=["relative"] * (len(means) - 1) + ["total"],
        connector={"line": {"color": "gray"}},
        increasing={"marker": {"color": "#00CC96"}},
        decreasing={"marker": {"color": "#EF553B"}},
        totals={"marker": {"color": "#636EFA"}},
    ))
    fig.update_layout(title=f"Change in Avg Body Mass Relative to {ref_species}", title_x=0.5)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### 📉 What This Chart Tells Us

    This waterfall chart compares the **average body mass** of *Chinstrap* and *Gentoo* penguins relative to *Adelie*.

    - The chart starts with **Adelie** at baseline (0), since it’s selected as the reference.
    - The **green bar for Chinstrap** shows a **positive difference**, meaning Chinstrap penguins weigh **more on average than Adelie**.
    - The **blue bar for Gentoo** shows an even **larger positive jump**, confirming Gentoo penguins are **heaviest overall**.

    💡 **Insight**:
    > - **Adelie**: ~3709 grams (baseline)
    > - **Chinstrap**: ~3733 grams → about **24 grams heavier** than Adelie
    > - **Gentoo**: ~3738 grams → about **29 grams heavier** than Adelie

    This tells us that while **Chinstrap and Gentoo both outweigh Adelie**, the differences are not uniform—Gentoo penguins have the **highest body mass**, likely due to differences in habitat, diet, or species physiology.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🔁 Parallel Categories: Species × Island × Sex

    This chart is powerful for visualizing **combinations of categorical variables**.

    We use a **parallel categories plot** to explore how the species are distributed across islands and sexes.

    - Each band connects values between columns (e.g., species → island → sex)
    - Thicker bands = more penguins

    This helps answer:
    - Are certain species only found on certain islands?
    - Are male and female penguins distributed evenly?
    """
    )
    return


@app.cell(hide_code=True)
def _(px):
    def parallel(df, mo):
        mo.md("## 🔁 Parallel Categories: Species × Island × Sex")
        fig = px.parallel_categories(df, dimensions=["species", "island", "sex"], color_continuous_scale=px.colors.sequential.Inferno)
        fig.update_layout(title="Parallel Categories: Categorical Relationships", title_x=0.5)
        return mo.ui.plotly(fig)
    return (parallel,)


@app.cell(hide_code=True)
def _(df, mo, parallel):
    parallel(df, mo)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### 📊 What This Chart Reveals

    This **Parallel Categories** chart reveals several important insights about the Palmer Penguins dataset:

    - 🟢 **Species-Island Associations**:
      - **Adelie penguins** are the most evenly distributed, appearing across **Torgersen**, **Biscoe**, and **Dream** islands.
      - **Chinstrap penguins** are found **only on Dream island**, indicating a strong species-island dependency.
      - **Gentoo penguins** are found **only on Biscoe island**, showing a similar isolated distribution.

    - 🔵 **Sex Distribution Across Islands**:
      - Both **male and female penguins** appear in all species groups, suggesting **balanced sampling** by sex.
      - However, some islands and species combinations show thicker connections to one sex over the other, hinting at **uneven sample counts**—e.g., **more female Chinstraps** on Dream.

    - 🧠 **Why This Matters**:
      - This chart helps us **quickly identify categorical dependencies**. For instance, if we are studying habitat preferences or sex-based behavior, this plot immediately shows where those categories are **confined** or **overlapping**.
      - It also suggests potential sampling bias (e.g., if certain groups are overrepresented), which is important when doing any **statistical comparisons or modeling**.

    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""

    ## 🌞 Sunburst Chart: Island → Species → Sex (by Body Mass)

    This hierarchical **sunburst chart** visualizes how **body mass is distributed** across categories:

    - **Center**: Islands
    - **Middle**: Species
    - **Outer**: Sex

    Each segment’s size represents the **sum of body mass**, allowing us to trace mass distribution **top-down**.

    Useful to:
    - Compare total mass of species on each island
    - See whether one sex dominates a species in size
    """
    )
    return


@app.cell(hide_code=True)
def _(px):
    def sunburst(df, mo):
        mo.md("## 🧲 Sunburst Chart: Island → Species → Sex")
        fig = px.sunburst(df, path=["island", "species", "sex"], values="body_mass_g", color="island", color_discrete_sequence=px.colors.qualitative.Bold)
        fig.update_layout(title="Sunburst: Hierarchical Category Breakdown by Body Mass", title_x=0.5)
        return mo.ui.plotly(fig)
    return (sunburst,)


@app.cell(hide_code=True)
def _(df, mo, sunburst):
    sunburst(df, mo)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### 🌞 What This Chart Reveals

    This **Sunburst chart** provides a hierarchical view of how total **body mass** is distributed across islands, species, and sex:

    - 🏝️ **Islands (Center)**:
          - **Biscoe** contributes the **largest total body mass**, mostly due to **Gentoo penguins**.
          - **Dream** and **Torgersen** show smaller overall masses, dominated by **Chinstrap** and **Adelie**, respectively.

    - 🐧 **Species (Middle Layer)**:
          - **Gentoo penguins** dominate in size—both physically and in summed body mass—making them the largest contributors on Biscoe Island.
          - **Adelie penguins** appear on multiple islands but with smaller slices, showing that while they are numerous, their individual body masses are smaller.

    - 🚻 **Sex (Outer Layer)**:
          - Within each species and island, we can see the **balance (or imbalance)** between **male and female** penguins.
          - For instance, Biscoe’s Gentoo penguins appear to be **evenly split** by sex in body mass, whereas some species on smaller islands may show **slight tilts**.

    #### 🔍 Why It Matters:
    - This chart answers **top-down mass distribution** questions: 
          - *Where is the most penguin biomass located?*
          - *Which island hosts the heaviest penguins?*
          - *Are sexes equally represented in body contribution?*

    It’s particularly helpful when comparing **overall ecological load**, understanding which locations have **heavier penguin populations**, and examining **sex-based mass variation** within each group.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🔗 Sankey Diagram: Flow from Island to Species

    Finally, we use a **Sankey diagram** to represent the **flow of penguin counts** from islands to species.

    - The **width** of each flow shows how many penguins transition from one category to another.
    - It’s perfect for illustrating **volume transitions** between two categorical variables.

    This helps answer:
    - Which species dominate each island?
    - Where are penguins most abundant?
    """
    )
    return


@app.cell(hide_code=True)
def _(go):
    def sankey(df, mo):
        mo.md("## 🔗 Sankey Diagram: Flow from Island to Species")
        from collections import defaultdict

        df_counts = df.groupby(["island", "species"]).size().reset_index(name="count")

        labels = list(set(df_counts["island"]).union(df_counts["species"]))
        label_to_index = {label: i for i, label in enumerate(labels)}

        sources = df_counts["island"].map(label_to_index).tolist()
        targets = df_counts["species"].map(label_to_index).tolist()
        values = df_counts["count"].tolist()

        fig = go.Figure(go.Sankey(
            node=dict(label=labels),
            link=dict(source=sources, target=targets, value=values)
        ))
        fig.update_layout(title="Sankey: Number of Penguins from Island to Species", title_x=0.5)
        return mo.ui.plotly(fig)
    return (sankey,)


@app.cell(hide_code=True)
def _(df, mo, sankey):
    sankey(df, mo)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### 🧠 What This Chart Reveals

    This **Sankey diagram** highlights the **distribution of penguin counts** from each **island** to each **species** — not by mass, but by **individual frequency**.

    - 📦 **Width of each flow** indicates **how many penguins** of a given species were found on each island.
    - 💡 It’s useful for spotting **population hubs** and **island-specific species dominance**.

    #### Key Insights:
    - **Biscoe Island** has the **highest volume** of penguins overall, **almost entirely Gentoo**. This shows that Gentoo penguins are **concentrated** in one location.
    - **Dream Island** shows a high transition to **Chinstrap penguins**, confirming it’s the main habitat for that species.
    - **Torgersen Island** only connects to **Adelie penguins**, suggesting they are the **only species found** there.

    This chart complements the Sunburst by emphasizing **frequency-based distribution** (how many penguins live where), rather than **mass contribution**. Together, they offer a fuller ecological picture: *Gentoo may be largest in body mass, but Chinstrap dominates Dream in sheer number.*
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### 🎉 Thanks for Exploring

        In this notebook, we explored unconventional interactive visualizations with Plotly — perfect for multi-dimensional data storytelling.

        Feel free to adapt these charts for your own data workflows in Marimo!
    """
    )
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    return go, mo, pd, px


@app.cell(hide_code=True)
def _(pd):
    df = pd.read_csv("https://raw.githubusercontent.com/allisonhorst/palmerpenguins/master/inst/extdata/penguins.csv").dropna()
    return (df,)


if __name__ == "__main__":
    app.run()
