# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "numpy==2.2.5",
#     "pandas==2.2.3",
#     "plotly==6.0.1",
# ]
# ///

import marimo

__generated_with = "0.13.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go
    return go, mo, np, px


@app.cell
def _(mo):
    mo.md(
        r"""
    # Statistical Charts

    Welcome to this lesson on statistical charts with Plotly. In this lesson, you will learn to create and interact with four key chart types:

    - **Box Plots:** Understand data spread, median, and outliers.  
    - **Violin Plots:** Explore distribution shapes and density estimates.  
    - **Histograms:** Analyze frequency distributions and experiment with different bin sizes.  
    - **Error Bars:** Visualize measurement uncertainties and variability.

    We will use the built-in tipstips dataset from Plotly for our examples. This dataset contains real-world information about restaurant bills, tips, and customer attributes, making it perfect for exploring statistical visualizations. Here are the first few rows of this dataset displayed below:
    """
    )
    return


@app.cell
def _(px):
    tips = px.data.tips()
    tips.head()
    return (tips,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ---
    ## Box Plots

    Box plots help you visualize the five-number summary (minimum, first quartile, median, third quartile, and maximum) along with potential outliers. In this example, you’ll explore how the total bill varies by different grouping variables like day, time, or sex.
    """
    )
    return


@app.cell
def _(mo, px, tips):
    def update_boxplot(groupby_var='day'):
        fig = px.box(tips, 
                     x=groupby_var, 
                     y='total_bill',
                     title=f'Box Plot: Total Bill by {groupby_var.capitalize()}',
                     labels={groupby_var:groupby_var.capitalize(), 'total_bill':"Total Bill"})
        return mo.ui.plotly(fig)

    dropdown = mo.ui.dropdown(
        options=['day', 'time', 'sex'],
        value='day',
    )
    return dropdown, update_boxplot


@app.cell
def _(dropdown, mo):
    mo.hstack([dropdown, mo.md(f"Selected Variable: {dropdown.value}")])
    return


@app.cell
def _(dropdown, update_boxplot):
    update_boxplot(dropdown.value)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    Explore how customer spending changes using this interactive box plot of total bills from the Tips dataset. By default, you're seeing how bills vary by day of the week—notice how weekends like Saturday and Sunday tend to have higher medians and more outliers, suggesting busier and possibly more expensive dining days. Use the dropdown menu to switch the grouping to time of day or customer sex. This lets you uncover new patterns, like whether dinner tends to be pricier than lunch, or if there are differences in spending between male and female diners. As you explore, pay attention to the shape and spread of each box—each one tells a story about how total bills shift across different categories.

    ---
    ## Violin Plots

    Violin plots complement box plots by showing the full distribution via kernel density estimation. In this section, you can toggle a box overlay on the violin plot to compare both representations.
    """
    )
    return


@app.cell
def _(mo, px, tips):
    def update_violinplot(overlay_box=True):
        fig = px.violin(tips, 
                        y="total_bill", 
                        box=overlay_box,  # Show/hide box plot inside the violin
                        points="all",     # Show all individual points
                        title="Violin Plot of Total Bill",
                        labels={"total_bill": "Total Bill"})
        return mo.ui.plotly(fig)

    checkbox = mo.ui.checkbox(label="Include box plot overlay")
    return checkbox, update_violinplot


@app.cell
def _(checkbox, mo):
    mo.hstack([checkbox, mo.md(f"Box Plot Overlay: {checkbox.value}")])
    return


@app.cell
def _(checkbox, update_violinplot):
    update_violinplot(checkbox.value)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    Dive into the distribution of total bills with this interactive violin plot, which combines a density plot and scatter of individual data points to show both spread and concentration. You have the option to enhance the visualization further by checking the box to include a box plot overlay. This adds a clear summary of the median, quartiles, and potential outliers on top of the smooth violin shape. The result is a powerful blend of summary statistics and detailed distribution, letting you explore where most bills fall and how varied the spending can be. Toggle the box plot on or off to switch between a cleaner look and a more informative breakdown.

    ---

    ## Histograms

    Histograms illustrate the frequency distribution of a continuous variable. Adjusting the bin size can change how the distribution appears. In the example below, you will experiment with different bin sizes for the total bill.
    """
    )
    return


@app.cell
def _(mo, px, tips):
    def update_histogram(bin_size=5):
        # Calculate number of bins based on the chosen bin size
        min_val = tips['total_bill'].min()
        max_val = tips['total_bill'].max()
        nbins = int((max_val - min_val) / bin_size)

        fig = px.histogram(tips,
                           x="total_bill",
                           nbins=nbins,
                           title="Histogram of Total Bill",
                           labels={"total_bill": "Total Bill"})
        return mo.ui.plotly(fig)

    slider = mo.ui.slider(start=0.25, stop=15, label="Slider", value=3, step=0.25)
    return slider, update_histogram


@app.cell
def _(mo, slider):
    mo.hstack([slider, mo.md(f"Bin Size: {slider.value}")], justify='space-around')
    return


@app.cell
def _(slider, update_histogram):
    update_histogram(slider.value)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Interactively explore how total bills are distributed using this adjustable histogram. Use the slider at the top to control the bin size—the level of detail you see in the data. A smaller bin size gives you a finer view of how bill amounts vary, revealing subtle patterns and fluctuations. A larger bin size smooths out the distribution, helping you spot broader trends more easily. As you move the slider, notice how the shape of the histogram changes, giving you different perspectives on how often certain bill amounts occur. This tool puts you in control of the granularity, helping you uncover insights at both the detailed and summary levels.

    ---

    ## Error Bars

    Error bars display uncertainty in data measurements (like the standard error or confidence intervals). In this example, we group the tips data by day, calculate the mean total bill along with its standard error, and display these values using error bars on a bar chart.
    """
    )
    return


@app.cell
def _(go, mo, np, tips):
    grouped = tips.groupby('day').agg({'total_bill': ['mean', 'std', 'count']})
    grouped.columns = ['mean_total_bill', 'std_total_bill', 'count']
    grouped = grouped.reset_index()
    grouped['sem'] = grouped['std_total_bill'] / np.sqrt(grouped['count'])

    def update_error_bars(error_scale=1.0):
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=grouped['day'],
                y=grouped['mean_total_bill'],
                error_y=dict(
                    type='data',
                    array=grouped['sem'] * error_scale,
                    visible=True
                ),
                name='Mean Total Bill'
            )
        )
        fig.update_layout(title=f'Mean Total Bill by Day with Error Bars (Scale: {error_scale})',
                          xaxis_title="Day",
                          yaxis_title="Mean Total Bill")
        return mo.ui.plotly(fig)

    slider2 = mo.ui.slider(start=0.25, stop=9, label="Slider", value=3, step=0.25)
    return slider2, update_error_bars


@app.cell
def _(mo, slider2):
    mo.hstack([slider2, mo.md(f"Error Scale: {slider2.value}")], justify='space-around')
    return


@app.cell
def _(slider2, update_error_bars):
    update_error_bars(slider2.value)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    This interactive bar chart lets you explore the average total bill by day, with adjustable error bars to reflect uncertainty or variability in the data. Use the slider to change the error scale—making the error bars longer or shorter depending on how much variability you want to emphasize. A higher scale exaggerates the range of possible values, helping highlight potential overlap or spread in the data, while a smaller scale offers a tighter, more focused view around the mean. This tool is especially useful for understanding not just central tendencies, but also the confidence or variability behind those averages, giving you a clearer sense of which differences might be meaningful.

    ---
    ## Conclusion

    You’ve now gained hands‑on experience creating and interpreting box plots, violin plots, histograms, and error‑bar charts with Plotly. By manipulating sliders, checkboxes, and dropdowns, you’ve seen firsthand how visual parameters—like bin size or error‑bar scale—can transform your understanding of data distributions and variability.
    """
    )
    return



if __name__ == "__main__":
    app.run()
