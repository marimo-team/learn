# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "numpy==2.3.1",
#     "pyarrow==20.0.0",
#     "pandas==2.3.0",
#     "plotly==6.1.2",
# ]
# ///

import marimo

__generated_with = "0.14.7"
app = marimo.App(width="medium", app_title="Plotly Basic Charts")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    import pyarrow
    return go, mo, np, pd, px


@app.cell(hide_code=True)
def _(mo, np, pd):
    np.random.seed(42)
    data = (np.random.rand(100, 10)) * 10
    df = pd.DataFrame(data, columns=[f"Col-{i + 1}" for i in range(10)])

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].round(3)

    dates = pd.date_range(start="2022-01-01", periods=len(df), freq="D")
    df_time = pd.DataFrame(
        {
            "Date": dates,
            "Value-1": np.random.randn(100).cumsum() + 10,
            "Value-2": np.random.randn(100).cumsum() + 20,
        }
    )

    mo.ui.tabs(
        {
            "Regular DataFrame": df,
            "Time Series DataFrame": df_time,
        }
    )
    return df, df_time


@app.cell(hide_code=True)
def _(df, go, mo):
    # Create sliders for basic line chart customization
    line_width_slider = mo.ui.slider(1, 10, step=1, value=2, label="Line Width")
    marker_size_slider = mo.ui.slider(2, 15, step=1, value=6, label="Marker Size")


    # Simple Line Plot
    def basic_line_plot():
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["Col-1"],
                mode="lines",
                name="Line",
                line=dict(width=line_width_slider.value),
            )
        )
        fig.update_layout(title="Basic Line Plot")
        return fig


    # Multi-Line Plot
    def multi_line_plot():
        fig = go.Figure()
        for i, col in enumerate(["Col-1", "Col-2", "Col-3"]):
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[col],
                    mode="lines",
                    name=f"Series {i + 1}",
                    line=dict(width=line_width_slider.value),
                )
            )
        fig.update_layout(title="Multi-Line Plot")
        return fig


    # Line and Scatter Plot
    def line_scatter_plot():
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["Col-1"],
                mode="lines",
                name="Line",
                line=dict(width=line_width_slider.value),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["Col-2"],
                mode="markers",
                name="Markers",
                marker=dict(size=marker_size_slider.value),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["Col-3"],
                mode="lines+markers",
                name="Line + Markers",
                line=dict(width=line_width_slider.value),
                marker=dict(size=marker_size_slider.value),
            )
        )
        fig.update_layout(title="Line and Scatter Plot")
        return fig
    return (
        basic_line_plot,
        line_scatter_plot,
        line_width_slider,
        marker_size_slider,
        multi_line_plot,
    )


@app.cell(hide_code=True)
def _(
    basic_line_plot,
    line_scatter_plot,
    line_width_slider,
    marker_size_slider,
    mo,
    multi_line_plot,
):
    line_charts = mo.ui.tabs(
        {
            "Line Plot": mo.ui.plotly(basic_line_plot()),
            "Multi-Line Plot": mo.ui.plotly(multi_line_plot()),
            "Line and Scatter Plot": mo.ui.plotly(line_scatter_plot()),
        }
    )
    basic_line_charts = mo.vstack(
        [
            mo.md("Adjust the parameters below to customize the charts:"),
            mo.hstack(
                [line_width_slider, marker_size_slider],
                align="center",
                justify="start",
            ),
            line_charts,
        ],
        gap=2,
    )
    return (basic_line_charts,)


@app.cell(hide_code=True)
def _(mo):
    color_selector = mo.ui.dropdown(
        options={
            "firebrick": "Red",
            "royalblue": "Blue",
            "green": "Green",
            "purple": "Purple",
            "orange": "Orange",
            "black": "Black",
        },
        value="royalblue",
        label="Primary Line color",
    )

    dash_style = mo.ui.dropdown(
        options={
            "solid": "Solid",
            "dash": "Dashed",
            "dot": "Dotted",
            "dashdot": "Dash-Dot",
        },
        value="solid",
        label="Line Style",
    )

    marker_size = mo.ui.slider(2, 20, step=2, value=8, label="Marker Size")
    return color_selector, dash_style, marker_size


@app.cell(hide_code=True)
def _(color_selector, dash_style, df, go, line_width_slider, marker_size, mo):
    # Function to create styled line chart with live updates
    def create_styled_line():
        fig = go.Figure()

        # Add main styled line
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["Col-1"],
                mode="lines+markers",
                name="Primary Line",
                line=dict(
                    color=color_selector.value,
                    width=line_width_slider.value,
                    dash=dash_style.value,
                ),
                marker=dict(size=marker_size.value, line=dict(width=2)),
            )
        )

        # Add reference line
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["Col-2"],
                mode="lines",
                name="Reference Line",
                line=dict(color="gray", width=1, dash="dot"),
                opacity=0.7,
            )
        )

        fig.update_layout(
            title="Interactive Styled Line Chart",
            xaxis_title="X Axis",
            yaxis_title="Y Axis",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )
        return fig


    # Combine controls and chart
    styled_line_ui = mo.hstack(
        [
            mo.hstack(
                [
                    mo.md("#### Style Controls"),
                    color_selector,
                    line_width_slider,
                    dash_style,
                    marker_size,
                ],
                gap=2,
                justify="center",
            ),
            create_styled_line,
        ],
        align="center",
        justify="center",
        widths=[1, 3],
    )


    # Functions for other styled examples (non-interactive)
    def line_color_customization():
        color_custom = go.Figure()
        color_custom.add_trace(
            go.Scatter(
                x=df.index,
                y=df["Col-1"],
                line=dict(color="firebrick", width=2),
                name="Red Line",
            )
        )
        color_custom.add_trace(
            go.Scatter(
                x=df.index,
                y=df["Col-2"],
                line=dict(color="royalblue", width=2),
                name="Blue Line",
            )
        )
        color_custom.add_trace(
            go.Scatter(
                x=df.index,
                y=df["Col-3"],
                line=dict(color="green", width=2),
                name="Green Line",
            )
        )
        color_custom.update_layout(title="Line color Customization")
        return color_custom


    def line_dash_patterns():
        dash_patterns = go.Figure()
        dash_patterns.add_trace(
            go.Scatter(
                x=df.index, y=df["Col-1"], line=dict(dash="solid"), name="Solid"
            )
        )
        dash_patterns.add_trace(
            go.Scatter(
                x=df.index, y=df["Col-2"], line=dict(dash="dash"), name="Dash"
            )
        )
        dash_patterns.add_trace(
            go.Scatter(
                x=df.index, y=df["Col-3"], line=dict(dash="dot"), name="Dot"
            )
        )
        dash_patterns.add_trace(
            go.Scatter(
                x=df.index,
                y=df["Col-4"],
                line=dict(dash="dashdot"),
                name="Dash-Dot",
            )
        )
        dash_patterns.update_layout(title="Line Dash Patterns")
        return dash_patterns


    def annotations_example():
        annotations = go.Figure()
        annotations.add_trace(go.Scatter(x=df.index, y=df["Col-1"], name="Data"))
        annotations.add_annotation(
            x=50,
            y=df["Col-1"].iloc[50],
            text="Maximum Point",
            showarrow=True,
            arrowhead=1,
        )
        annotations.add_annotation(
            x=25,
            y=df["Col-1"].iloc[25],
            text="Interesting Pattern",
            showarrow=True,
            arrowhead=1,
        )
        annotations.update_layout(title="Annotations and Text Labels")
        return annotations
    return (
        annotations_example,
        line_color_customization,
        line_dash_patterns,
        styled_line_ui,
    )


@app.cell(hide_code=True)
def _(
    annotations_example,
    line_color_customization,
    line_dash_patterns,
    mo,
    styled_line_ui,
):
    # Create tabs for all styled examples
    styled_examples = mo.ui.tabs(
        {
            "Interactive Styling": styled_line_ui,
            "color Examples": mo.ui.plotly(line_color_customization()),
            "Line Dash Examples": mo.ui.plotly(line_dash_patterns()),
            "Annotations Example": mo.ui.plotly(annotations_example()),
        }
    )

    styled_line_charts = mo.vstack(
        [
            mo.md("### Styled Line Charts"),
            mo.md("Use the interactive controls or explore preset examples:"),
            styled_examples,
        ],
        gap=2,
    )
    return (styled_line_charts,)


@app.cell(hide_code=True)
def _(df_time, go, mo, px):
    # Time Series Line Charts

    # Date and Time on X-axis
    time_series = px.line(
        df_time, x="Date", y="Value-1", title="Time Series with Date on X-axis"
    )

    # Time interval analysis - resampled data
    df_monthly = df_time.set_index("Date").resample("ME").mean().reset_index()
    time_interval = px.line(
        df_monthly,
        x="Date",
        y="Value-1",
        title="Monthly Average (Time Interval Analysis)",
    )

    # Real-time data simulation
    rt_fig = go.Figure()
    rt_fig.add_trace(
        go.Scatter(
            x=df_time["Date"],
            y=df_time["Value-1"],
            name="Real-time Data",
            mode="lines",
        )
    )
    rt_fig.add_trace(
        go.Scatter(
            x=df_time["Date"].iloc[-10:],
            y=df_time["Value-1"].iloc[-10:],
            line=dict(color="red", width=4),
            name="Latest Data",
        )
    )
    rt_fig.update_layout(title="Real-time Data Streaming Simulation")

    time_series_line_charts = mo.ui.tabs(
        {
            "Data and Time on the X-axis": mo.ui.plotly(time_series),
            "Time Interval Analysis": mo.ui.plotly(time_interval),
            "Real-time Data Streaming": mo.ui.plotly(rt_fig),
        }
    )
    return (time_series_line_charts,)


@app.cell(hide_code=True)
def _(df, go, mo):
    # Multi-Axis Line Charts

    # Dual Y-axis line plot
    dual_y = go.Figure()
    dual_y.add_trace(
        go.Scatter(x=df.index, y=df["Col-1"], name="Left Y-axis data")
    )
    dual_y.add_trace(
        go.Scatter(
            x=df.index, y=df["Col-5"] * 10, name="Right Y-axis data", yaxis="y2"
        )
    )
    dual_y.update_layout(
        title="Dual Y-axis Line Plot",
        yaxis=dict(title="Left Y-axis"),
        yaxis2=dict(title="Right Y-axis", overlaying="y", side="right"),
    )

    # Multiple Y-axes
    multi_y = go.Figure()
    multi_y.add_trace(go.Scatter(x=df.index, y=df["Col-1"], name="Y-axis 1"))
    multi_y.add_trace(
        go.Scatter(x=df.index, y=df["Col-2"] * 5, name="Y-axis 2", yaxis="y2")
    )
    multi_y.add_trace(
        go.Scatter(x=df.index, y=df["Col-3"] * 10, name="Y-axis 3", yaxis="y3")
    )

    multi_y.update_layout(
        title="Multiple Y-axes Line Plot",
        yaxis=dict(title="Y-axis 1"),
        yaxis2=dict(title="Y-axis 2", overlaying="y", side="right"),
        yaxis3=dict(
            title="Y-axis 3",
            overlaying="y",
            anchor="free",
            side="right",
            position=0.85,
        ),
    )

    multi_axis_line_charts = mo.ui.tabs(
        {
            "Dual Y-axis Line Plot": mo.ui.plotly(dual_y),
            "Multiple Y-axes": mo.ui.plotly(multi_y),
        }
    )
    return (multi_axis_line_charts,)


@app.cell(hide_code=True)
def _(df, go, mo):
    # Interactive Line Charts

    # Hover effects
    hover_fig = go.Figure()
    hover_fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Col-1"],
            name="Hover Data",
            hovertemplate="Index: %{x}<br>Value: %{y:.2f}<extra></extra>",
        )
    )
    hover_fig.update_layout(title="Enhanced Hover Effects")

    # Zoom and Pan
    zoom_fig = go.Figure()
    zoom_fig.add_trace(go.Scatter(x=df.index, y=df["Col-1"], name="Data 1"))
    zoom_fig.add_trace(go.Scatter(x=df.index, y=df["Col-2"], name="Data 2"))
    zoom_fig.update_layout(
        title="Zoom and Pan Features",
        xaxis=dict(rangeslider=dict(visible=True)),
        updatemenus=[
            dict(
                buttons=[
                    dict(
                        label="Reset Zoom",
                        method="relayout",
                        args=[{"xaxis.range": [None, None]}],
                    )
                ],
                direction="left",
                pad={"r": 10, "t": 10},
                showactive=False,
                type="buttons",
                x=0.1,
                xanchor="left",
                y=1.1,
                yanchor="top",
            )
        ],
    )

    # Click events placeholder - in a real app, this would have callbacks
    click_fig = go.Figure()
    click_fig.add_trace(
        go.Scatter(x=df.index, y=df["Col-1"], name="Clickable Data")
    )
    click_fig.update_layout(
        title="Click Events and Interactions (click points to see)",
        annotations=[
            dict(
                x=50,
                y=df["Col-1"].iloc[50],
                xref="x",
                yref="y",
                text="Click on points!",
                showarrow=True,
                arrowhead=2,
                ax=0,
                ay=-40,
            )
        ],
    )

    interactive_line_charts = mo.ui.tabs(
        {
            "Hover Effects": mo.ui.plotly(hover_fig),
            "Zoom and Pan": mo.ui.plotly(zoom_fig),
            "Click Events and Interactions": mo.ui.plotly(click_fig),
        }
    )
    return (interactive_line_charts,)


@app.cell(hide_code=True)
def _(df, go, mo, px):
    # Comparative Line Charts

    # Stacked line chart (area chart)
    stacked = px.area(
        df, x=df.index, y=["Col-1", "Col-2", "Col-3"], title="Stacked Line Chart"
    )

    # Grouped line chart
    grouped = go.Figure()
    # Group 1
    grouped.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Col-1"],
            name="Group 1 - Series 1",
            line=dict(color="blue"),
        )
    )
    grouped.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Col-2"],
            name="Group 1 - Series 2",
            line=dict(color="lightblue"),
        )
    )

    # Group 2
    grouped.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Col-5"],
            name="Group 2 - Series 1",
            line=dict(color="red"),
        )
    )
    grouped.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Col-6"],
            name="Group 2 - Series 2",
            line=dict(color="lightcoral"),
        )
    )
    grouped.update_layout(title="Grouped Line Chart")

    # Overlapping line chart
    overlap = go.Figure()
    overlap.add_trace(
        go.Scatter(
            x=df.index, y=df["Col-1"], name="Series 1", line=dict(color="blue")
        )
    )
    overlap.add_trace(
        go.Scatter(
            x=df.index, y=df["Col-2"], name="Series 2", line=dict(color="red")
        )
    )
    overlap.add_trace(
        go.Scatter(
            x=df.index, y=df["Col-3"], name="Series 3", line=dict(color="green")
        )
    )
    overlap.update_layout(title="Overlapping Line Chart")

    comparative_line_charts = mo.ui.tabs(
        {
            "Stacked Line Charts": mo.ui.plotly(stacked),
            "Grouped Line Charts": mo.ui.plotly(grouped),
            "Overlapping Line Charts": mo.ui.plotly(overlap),
        }
    )
    return (comparative_line_charts,)


@app.cell(hide_code=True)
def _(
    basic_line_charts,
    comparative_line_charts,
    interactive_line_charts,
    mo,
    multi_axis_line_charts,
    styled_line_charts,
    time_series_line_charts,
):
    main_categories = mo.ui.tabs(
        {
            "Basic Line Charts": basic_line_charts,
            "Styled Line Charts": styled_line_charts,
            "Time Series Line Charts": time_series_line_charts,
            "Multi-Axis Line Charts": multi_axis_line_charts,
            "Interactive Line Charts": interactive_line_charts,
            "Comparative Line Charts": comparative_line_charts,
        }
    )

    mo.vstack(
        [
            mo.md("""
        # Basic Line Charts

        This app demonstrates various categories of line charts using Plotly.
        Select a category below to explore different line chart types.
        """),
            main_categories,
        ],
        gap=2,
    )
    return


if __name__ == "__main__":
    app.run()
