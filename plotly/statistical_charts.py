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

__generated_with = "0.12.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Statistical Charts
        By [Bryan Zhang](https://github.com/BryanZhang938)

        Welcome to this lesson on statistical charts with Plotly. In this lesson, you will learn to create and interact with four key chart types:

        - **Box Plots:** Understand data spread, median, and outliers.  
        - **Violin Plots:** Explore distribution shapes and density estimates.  
        - **Histograms:** Analyze frequency distributions and experiment with different bin sizes.  
        - **Error Bars:** Visualize measurement uncertainties and variability.

        We will use the built-in [tips dataset](https://github.com/plotly/datasets/blob/master/tips.csv) from Plotly for our examples. This dataset contains real-world information about restaurant bills, tips, and customer attributes, making it perfect for exploring statistical visualizations. Here are the first few rows of this dataset displayed below:
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
        ## Box Plots

        Box plots help you visualize the five-number summary (minimum, first quartile, median, third quartile, and maximum) along with potential outliers. In this example, you’ll explore how the total bill varies by different grouping variables like day, time, or sex.
        """
    )
    return


@app.cell
def _(mo):
    dropdown = mo.ui.dropdown(
        options=['day', 'time', 'sex'],
        value='day',
    )
    return (dropdown,)


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
        Explore how customer spending changes using this interactive box plot of total bills. By default, you're seeing how bills vary by day of the week. This box plot gives a clear look at how total bill amounts vary by day of the week. It’s immediately noticeable that Saturday and Sunday have higher medians and more extreme outliers compared to weekdays like Thursday and Friday. This makes intuitive sense as weekends are peak times for dining out, leading to larger group sizes, longer stays, and potentially more expensive meals, which contribute to higher bills.

        A couple of interesting insights:

        * Sunday shows a noticeably wider spread, hinting that people might either go for cheap brunches or big family dinners.
        * Friday seems relatively modest in terms of both spread and outliers. Perhaps people are grabbing quicker meals or just transitioning into the weekend.
        * Thursday is visibly lower, indicating a quieter dining day with fewer large bills.

        If you switch the dropdown to group by time of day or sex, you might notice:

        * Dinner bills tend to be higher than lunch.
        * Male diners often have slightly higher bills.

        This kind of visualization really helps tell the story of human behavior through data and their eating habits throughout the week.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Violin Plots

        Violin plots complement box plots by showing the full distribution via kernel density estimation. In this section, you can toggle a box overlay on the violin plot to compare both representations.
        """
    )
    return


@app.cell
def _(mo):
    checkbox = mo.ui.checkbox(label="Include box plot overlay")
    return (checkbox,)


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
        Explore how customer spending changes using this interactive violin plot of total bills. By default, you’re seeing the full distribution of total bill values using a combination of kernel density estimation and individual data points. This violin plot gives a rich view of where most bills fall and how frequently certain bill ranges appear. The plot reveals that the majority of total bills cluster between $10 and $20, with a dense concentration around $15–20, which makes sense with one or two diners.

        A couple of interesting insights:

        * The distribution thins out gradually past $30 but extends up to $50, indicating a smaller population of high spenders.
        * There’s a slight dip around $25–30, which could suggest fewer people are medium spenders.

        If you enable the box plot overlay checkbox, you might notice:

        * A right-skewed box confirming that while most bills are moderate, some outliers go much higher.
        * The median aligns with the densest part of the violin.
        * The outliers highlight the distribution of big spenders.

        This kind of visualization is powerful for understanding how different spending behaviors occur. It brings to light both the routine choices of the average diner and the bigger spenders, helping you intuitively grasp the shape of customer spending habits.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Histograms

        Histograms illustrate the frequency distribution of a continuous variable. Adjusting the bin size can change how the distribution appears. In the example below, you will experiment with different bin sizes for the total bill.
        """
    )
    return


@app.cell
def _(mo):
    slider = mo.ui.slider(start=0.25, stop=15, label="Slider", value=3, step=0.25)
    return (slider,)


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
        Explore how customer spending is distributed using this interactive histogram of total bills. This histogram shows how often different total bill amounts occur, grouped into bins whose size you can adjust using the slider. With a bin size of 3.0, the plot reveals that most total bills fall between $10 and $25, with a steep drop-off after that. The shape of the histogram is right-skewed, reflecting that while high bills are possible, they’re much less common.

        A couple of interesting insights:

        * The highest frequency is seen in the $15–20 range, reinforcing the idea that this is the most common amount of customer spendings.
        * Very few bills exceed $40, though there are a handful, which could reflect large parties or customers ordering premium items.
        * The first bin (just above \$0) is nearly empty, showing that it’s rare for someone to spend less than $5–6, which might indicate a dine-in minimum or typical meal base cost.

        If you change the bin size using the slider, you might notice:

        * Smaller bins expose more granular variation with small spikes or gaps that could suggest specific pricing patterns or menu clustering.
        * Larger bins smooth the curve but hide finer trends, which are useful for seeing the big picture but not small-scale quirks.

        This visualization is especially helpful for understanding the distribution shape and concentration of typical spending. It not only confirms where most customers land in terms of cost, but also highlights the rarity and scale of outliers, in particular the higher spending cases.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Error Bars

        Error bars display uncertainty in data measurements (like the standard error or confidence intervals). In this example, we group the tips data by day, calculate the mean total bill along with its standard error, and display these values using error bars on a bar chart.
        """
    )
    return


@app.cell
def _(mo, np, tips):
    grouped = tips.groupby('day').agg({'total_bill': ['mean', 'std', 'count']})
    grouped.columns = ['mean_total_bill', 'std_total_bill', 'count']
    grouped = grouped.reset_index()
    grouped['sem'] = grouped['std_total_bill'] / np.sqrt(grouped['count'])

    slider2 = mo.ui.slider(start=0.25, stop=9, label="Slider", value=3, step=0.25)
    return grouped, slider2


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
        Explore how average customer spending varies by day of the week using this interactive bar plot with error bars. This chart shows the mean total bill for each day, with vertical lines representing variability around the mean. At a default error scale of 3.0, it’s clear that Saturday and Sunday have higher average bills compared to Thursday and Friday, reinforcing the trend that weekends tend to see more spending.

        A couple of interesting insights:

        * Sunday has the highest mean bill, followed closely by Saturday, suggesting weekend diners spend more, possibly due to larger groups or longer stay times.
        * Thursday has the lowest mean, which aligns with earlier visualizations that showed it as a quieter dining day.
        * Friday, despite being the start of the weekend, shows a lower mean compared to Saturday and Sunday, indicating that people could just be grabbing a quick meal.

        If you adjust the error bar scale using the slider, you might notice:

        * Larger scales exaggerate the error bars, making variability more prominent and helping to visually emphasize overlap between days.
        * Smaller scales make the bars tighter, better for seeing the central trend without as much distraction from the noise.

        This visualization is useful for comparing averages while accounting for uncertainty or variability in the data. It provides a more nuanced view than plain bar plots by hinting at how consistent or spread out each day’s spending behavior might be.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ---

        ## When to Use Box Plots, Violin Plots, Histograms, and Error Bars

        Each of these visualizations has its strengths depending on what you want to learn from the data. Here's a breakdown of when and why you'd choose each one:

        ### Box Plots

        Use box plots when you want a concise summary of the distribution, such as the median, quartiles, and presence of outliers. They’re ideal for comparing distributions across categories, like total bill amounts by day of the week.

        Use when:

        * You want to identify outliers and central tendency quickly.
        * You need to compare the spread and skew across multiple groups.

        Avoid when:

        * You're just analyzing a single variable and want to see detailed distribution shape.

        ---

        ### Violin Plots

        Use violin plots when you want to visualize the full shape of the distribution. Violin plots combine the summary features of box plots with a smoothed density estimate, making them great for exploring multimodal or skewed distributions.

        Use when:

        * You want to see not just the median and spread but also where values concentrate.
        * You’re curious about the overall shape of the distribution.
        * You’re comparing categories but want more nuance than box plots offer.

        Avoid when:

        * You're focusing on simplicy of the visualizations, as violin plots can be difficult to interpret at a glance.

        ---

        ### Histograms

        Use histograms to explore the distribution of a single quantitative variable, especially when the raw shape of the data matters. Histograms are useful for spotting skew, gaps, or clusters in the data.

        Use when:

        * You’re analyzing the frequency of different value ranges for one variable.
        * You want to identify the most common spending amounts.

        Avoid when:

        * Comparing multiple categories at the same time.

        ---

        ### Error Bars

        Use error bars to show variability or uncertainty around average values (like the mean). They are useful for communicating confidence intervals or standard deviation visually.

        Use when:

        * You want to compare averages across groups while acknowledging data variability.
        * You're presenting statistical summaries where reliability matters.
        * You need to visually assess whether differences between groups are likely significant.

        Avoid when:

        * The full distribution or individual data points are important.
        * You don’t have a meaningful way to quantify variability, like with a confidence interval.

        ---

        By combining these plots thoughtfully, you can get deep insights into your dataset.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Conclusion

        This lesson provides a deep dive into customer spending behavior using the Tips dataset. Each visualization offers a unique perspective on how total bills vary across different conditions like day of the week, time of day, or customer demographics.

        Box plots helped us quickly compare spending patterns across days and detect outliers. Violin plots revealed the full shape of those distributions, highlighting where most customers tend to spend. Histograms showed us how spending is distributed overall, letting us fine-tune our view using adjustable bin sizes. Error bars allowed us to visualize average spending while accounting for variability, helping us interpret the statistical reliability of those comparisons.

        Together, these tools tell a more complete story by capturing both average trends and unusual behaviors. By interacting with the visualizations, users can uncover insights about how, when, and how much people tend to spend, which can inform decisions in fields ranging from restaurant management to customer behavior analysis.
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go
    return go, mo, np, pd, px


@app.cell
def _(mo, px, tips):
    def update_boxplot(groupby_var='day'):
        fig = px.box(tips, 
                     x=groupby_var, 
                     y='total_bill',
                     title=f'Box Plot: Total Bill by {groupby_var.capitalize()}',
                     labels={groupby_var:groupby_var.capitalize(), 'total_bill':"Total Bill"})
        return mo.ui.plotly(fig)
    return (update_boxplot,)


@app.cell
def _(go, grouped, mo):
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
    return (update_error_bars,)


@app.cell
def _(mo, px, tips):
    def update_histogram(bin_size=5):
        min_val = tips['total_bill'].min()
        max_val = tips['total_bill'].max()
        nbins = int((max_val - min_val) / bin_size)

        fig = px.histogram(tips,
                           x="total_bill",
                           nbins=nbins,
                           title="Histogram of Total Bill",
                           labels={"total_bill": "Total Bill"})
        return mo.ui.plotly(fig)
    return (update_histogram,)


@app.cell
def _():
    return


@app.cell
def _(mo, px, tips):
    def update_violinplot(overlay_box=True):
        fig = px.violin(tips, 
                        y="total_bill", 
                        box=overlay_box, 
                        points="all",    
                        title="Violin Plot of Total Bill",
                        labels={"total_bill": "Total Bill"})
        return mo.ui.plotly(fig)
    return (update_violinplot,)


if __name__ == "__main__":
    app.run()
