# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "altair",
#     "asimpy",
#     "marimo",
#     "polars==1.24.0",
# ]
# ///

import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import random
    import statistics

    import altair as alt
    import polars as pl

    from asimpy import Environment, Process

    return Environment, Process, alt, mo, pl, random, statistics


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # The Inspector's Paradox

    ## *Why the Bus Is Always Late*

    Buses arrive at a stop with some average headway (gap between buses) of $\mu$ minutes. A passenger arrives at a uniformly random time and waits for the next bus. How long do they wait? The naive answer is $\mu / 2$: on average you land in the middle of a gap. The correct answer is almost always longer—sometimes much longer.

    The expected wait is not $\mu/2$ but:

    $$E[\text{wait}] = \frac{\mu}{2} + \frac{\sigma^2}{2\mu}$$

    where $\sigma^2 = \text{Var}[\text{headway}]$. The second term is always non-negative, so higher variance always means longer expected waits, even when the mean headway is unchanged.

    ### Three Bus Schedules with Mean Headway $\mu = 10$

    | Schedule    | $\sigma^2$ | Predicted wait | Naive wait |
    |-------------|-----------|----------------|-----------|
    | Regular     | 0         | 5.0            | 5.0       |
    | Exponential | 100       | 10.0           | 5.0       |
    | Clustered   | 64        | 8.2            | 5.0       |

    For exponentially distributed headways, $\sigma^2 = \mu^2$, so:

    $$E[\text{wait}] = \frac{\mu}{2} + \frac{\mu^2}{2\mu} = \mu$$

    A passenger waits on average for an *entire* mean headway — twice the naive expectation.

    ## Why This Happens: Length-Biased Sampling

    A passenger arriving at a random time is more likely to land inside a *long* gap than a short one, because long gaps occupy more time on the clock. This is called *length-biased sampling*. The interval containing your arrival is not a random headway: it is drawn from the length-biased distribution with density:

    $$f^*(h) = \frac{h \cdot f(h)}{\mu}$$

    The mean of this biased distribution is $\mu + \sigma^2/\mu$, and you arrive uniformly within it, giving expected wait $(\mu + \sigma^2/\mu)/2$.

    The same phenomenon explains why the average class size experienced by a student exceeds the average class size reported by the university (large classes have more students to report them).

    ## Why "Inspector's Paradox"?

    The name comes from quality control, where an inspector arrives at a random time to sample a production process and systematically encounters longer-than-average intervals. The paradox is that a random observer is more likely to land inside a long gap than a short one, so their experienced mean interval exceeds the true mean interval. It feels paradoxical because you'd expect a random arrival to see the average gap, but length-biased sampling guarantees they see worse-than-average gaps whenever there's any variance at all.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Implementation

    A `BusService` process generates buses under three headway distributions (regular, exponential, clustered bimodal) and records their arrival times. After the simulation, passenger wait times are estimated by sampling $N$ uniformly random arrival times and finding the next bus for each, without needing explicit `Passenger` processes.
    """)
    return


@app.cell
def _(mo):
    sim_time_slider = mo.ui.slider(
        start=0,
        stop=100_000,
        step=1_000,
        value=20_000,
        label="Simulation time",
    )

    mean_headway_slider = mo.ui.slider(
        start=5.0,
        stop=30.0,
        step=1.0,
        value=10.0,
        label="Mean headway",
    )

    seed_input = mo.ui.number(
        value=192,
        step=1,
        label="Random seed",
    )

    run_button = mo.ui.run_button(label="Run simulation")

    mo.vstack([
        sim_time_slider,
        mean_headway_slider,
        seed_input,
        run_button,
    ])
    return mean_headway_slider, seed_input, sim_time_slider


@app.cell
def _(mean_headway_slider, seed_input, sim_time_slider):
    SIM_TIME = int(sim_time_slider.value)
    MEAN_HEADWAY = float(mean_headway_slider.value)
    SEED = int(seed_input.value)
    N_PASSENGERS = 20_000
    return MEAN_HEADWAY, N_PASSENGERS, SEED, SIM_TIME


@app.cell
def _(MEAN_HEADWAY, Process, random):
    class BusService(Process):
        def init(self, mode, bus_arrivals):
            self.mode = mode
            self.bus_arrivals = bus_arrivals

        async def run(self):
            while True:
                if self.mode == "regular":
                    headway = MEAN_HEADWAY
                elif self.mode == "exponential":
                    headway = random.expovariate(1.0 / MEAN_HEADWAY)
                elif self.mode == "clustered":
                    headway = MEAN_HEADWAY * 0.2 if random.random() < 0.5 else MEAN_HEADWAY * 1.8
                else:
                    raise ValueError(f"Unknown mode: {self.mode}")
                await self.timeout(headway)
                self.bus_arrivals.append(self.now)

    return (BusService,)


@app.cell
def _(BusService, Environment, SIM_TIME):
    def collect_buses(mode):
        bus_arrivals = []
        env = Environment()
        BusService(env, mode, bus_arrivals)
        env.run(until=SIM_TIME)
        return bus_arrivals

    return (collect_buses,)


@app.cell
def _(N_PASSENGERS, random, statistics):
    def expected_wait(bus_arrivals, n=N_PASSENGERS):
        max_t = bus_arrivals[-1]
        waits = []
        for _ in range(n):
            t = random.uniform(0.0, max_t * 0.95)
            for b in bus_arrivals:
                if b > t:
                    waits.append(b - t)
                    break
        return statistics.mean(waits) if waits else 0.0

    return (expected_wait,)


@app.cell
def _(statistics):
    def headway_variance(bus_arrivals):
        headways = [b - a for a, b in zip(bus_arrivals, bus_arrivals[1:])]
        return statistics.variance(headways) if len(headways) > 1 else 0.0

    return (headway_variance,)


@app.cell(hide_code=True)
def _(MEAN_HEADWAY, mo):
    mu = MEAN_HEADWAY
    naive = MEAN_HEADWAY / 2.0
    var_exp = mu ** 2
    var_clustered = 0.5 * (mu * 0.2 - mu) ** 2 + 0.5 * (mu * 1.8 - mu) ** 2
    mo.md(f"""
    ## Results

    Mean headway: {MEAN_HEADWAY} → naive expected wait = {naive:.1f}

    - **Exponential** (Var ≈ {var_exp:.1f}): predicted = {mu / 2 + var_exp / (2 * mu):.1f} (= full mean headway!)
    - **Clustered** (Var ≈ {var_clustered:.1f}): predicted = {mu / 2 + var_clustered / (2 * mu):.1f}
    """)
    return (naive,)


@app.cell
def _(MEAN_HEADWAY, collect_buses, expected_wait, headway_variance, naive, pl):
    def run_models():
        rows = []
        for mode in ["regular", "exponential", "clustered"]:
            buses = collect_buses(mode)
            var_h = headway_variance(buses)
            mean_w = expected_wait(buses)
            rows.append({
                "mode": mode,
                "var_headway": round(var_h, 4),
                "mean_wait": round(mean_w, 4),
                "predicted": round(MEAN_HEADWAY / 2.0 + var_h / (2.0 * MEAN_HEADWAY), 4),
                "ratio": round(mean_w / naive, 4),
            })
        return pl.DataFrame(rows)

    return (run_models,)


@app.cell
def _(SEED, random, run_models):
    random.seed(SEED)
    df = run_models()
    df
    return (df,)


@app.cell
def _(alt, df, naive, pl):
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("mode:N", title="Bus schedule type"),
            y=alt.Y("mean_wait:Q", title="Mean passenger wait"),
            color=alt.Color("mode:N", legend=None),
            tooltip=["mode:N", "mean_wait:Q", "ratio:Q"],
        )
        .properties(title="Inspector's Paradox: Mean Wait by Schedule Type")
    )
    naive_line = (
        alt.Chart(pl.DataFrame({"naive": [naive]}))
        .mark_rule(strokeDash=[4, 4], color="gray")
        .encode(y="naive:Q")
    )
    (chart + naive_line)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Understanding the Math

    ### Length-biased sampling

    Suppose buses run on an irregular schedule where gaps between buses are either 2 minutes or 18 minutes, each with probability 1/2. The mean gap is $\mu = (2 + 18)/2 = 10$ minutes. Now ask: if you arrive at a completely random moment, which gap are you most likely to land inside?

    A 2-minute gap occupies only 2 minutes on the clock, but an 18-minute gap occupies 18. Out of every 20 minutes of clock time on average, 2 minutes belong to a short gap and 18 to a long one. So a random arrival lands in a short gap with probability $2/(2+18) = 1/10$ and in a long gap with probability $18/20 = 9/10$. The expected gap length you experience is:

    $$E[\text{gap experienced}] = \frac{1}{10} \cdot 2 + \frac{9}{10} \cdot 18 = 0.2 + 16.2 = 16.4 \text{ minutes}$$

    That is far above the mean gap of 10 minutes. You are disproportionately likely to land inside a long gap simply because it takes up more time.

    ### The wait formula

    Once you are inside a gap, you arrive uniformly within it, so on average you land in the middle. Your expected wait is half the gap length you experience. The full formula is:

    $$E[\text{wait}] = \frac{\mu}{2} + \frac{\sigma^2}{2\mu}$$

    Here $\mu$ is the mean gap and $\sigma^2 = \text{Var}[\text{gap}]$ is the variance of gap lengths. The first term, $\mu/2$, is what you would get if every gap were exactly $\mu$ (deterministic buses — arrive in the middle every time). The second term, $\sigma^2/(2\mu)$, is the extra waiting from length-biased sampling. It is always non-negative, so irregular buses always make you wait longer than regular buses with the same mean headway.

    ### Why variance matters

    The variance $\sigma^2$ measures how spread out the gap sizes are. A perfectly regular bus schedule has $\sigma^2 = 0$ and gives the naive answer $\mu/2$. An exponentially distributed schedule has $\sigma^2 = \mu^2$, which doubles the expected wait to $\mu$. More irregular buses, higher penalty.

    ### Connecting to expected values

    The formula arises from a standard result: the expected length of the gap containing a random arrival is $\mu + \sigma^2/\mu$. You can think of this as the mean gap plus a correction term proportional to the variance divided by the mean. Dividing by 2 (uniform arrival within the gap) gives the wait formula above.
    """)
    return


if __name__ == "__main__":
    app.run()
