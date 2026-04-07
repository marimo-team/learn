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

__generated_with = "0.22.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import math
    import random

    import altair as alt
    import polars as pl

    from asimpy import Environment, Process, Resource

    return Environment, Process, Resource, alt, math, mo, pl, random


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Basic Ideas in Queueing Theory

    ## *Arrivals, Servers, and Utilization*

    Three concepts underpin every queueing model. The first is *Poisson arrivals*: when customers arrive independently at a constant average rate $\lambda$, gaps between consecutive arrivals follow an *exponential* distribution with mean $1/\lambda$, and the count of arrivals in any window of width $t$ follows a Poisson distribution with mean $\lambda t$. These two descriptions are equivalent: if one holds, the other must too.

    The second key idea is that the exponential distribution is *memoryless*. Knowing you have already waited $s$ units gives no information about when the next arrival will come. This property makes the math simple, but means that the exponential distribution isn't a good fit for scenarios where events happen in bursts. Some of the later tutorials will explore models that handle this.

    The final concept is *server utilization*. A server completing work at rate $\mu$ has utilization $\rho = \lambda/\mu$, which is the long-run fraction of time it is busy. The system is stable only when $\rho < 1$. When $\rho \geq 1$, arrivals outpace service and the queue grows without bound. Even at exact balance ($\rho = 1$), randomness creates bursts that accumulate faster than the server recovers, so the expected queue length is infinite.

    The code below uses a discrete event simulation package called [asimpy](https://asimpy.readthedocs.io/) to model a simple job queue. The technical term for this kind of system is an *[M/M/1 queue](https://en.wikipedia.org/wiki/M/M/1_queue)*: memoryless (Poisson) arrivals, memoryless service times, and one server.
    """)
    return


@app.cell
def _(mo):
    sim_time_slider = mo.ui.slider(
        start=0,
        stop=100_000,
        step=1_000,
        value=20_000,
        label="Simulation time"
    )
    arrival_rate_slider = mo.ui.slider(
        start=1.0,
        stop=5.0,
        step=0.01,
        value=2.0,
        label="Arrival rate (λ)"
    )
    service_rate_slider = mo.ui.slider(
        start=1.0,
        stop=5.0,
        step=0.01,
        value=2.0,
        label="Service rate (μ)"
    )
    window_slider = mo.ui.slider(
        start=1.0,
        stop=5.0,
        step=0.01,
        value=1.0,
        label="Counting window"
    )
    seed_input = mo.ui.number(
        value=192,
        step=1,
        label="Random seed"
    )
    run_button = mo.ui.run_button(label="Run simulation")
    mo.vstack([sim_time_slider, arrival_rate_slider, service_rate_slider, window_slider, seed_input, run_button])
    return (
        arrival_rate_slider,
        seed_input,
        service_rate_slider,
        sim_time_slider,
        window_slider,
    )


@app.cell
def _(
    arrival_rate_slider,
    random,
    seed_input,
    service_rate_slider,
    sim_time_slider,
    window_slider,
):
    SIM_TIME = int(sim_time_slider.value)
    ARRIVAL_RATE = float(arrival_rate_slider.value)
    SERVICE_RATE = float(service_rate_slider.value)
    WINDOW = float(window_slider.value)
    SEED = int(seed_input.value)
    random.seed(SEED)
    return ARRIVAL_RATE, SERVICE_RATE, SIM_TIME, WINDOW


@app.cell
def _(Process, random):
    class ArrivalSource(Process):
        """Generates arrivals at a Poisson rate and records inter-arrival gaps."""

        def init(self, rate, gaps):
            self.rate = rate
            self.gaps = gaps

        async def run(self):
            while True:
                gap = random.expovariate(self.rate)
                await self.timeout(gap)
                self.gaps.append(gap)

    return (ArrivalSource,)


@app.cell
def _(Process, SERVICE_RATE, random):
    class Customer(Process):
        def init(self, server, total_service):
            self.server = server
            self.total_service = total_service

        async def run(self):
            async with self.server:
                svc = random.expovariate(SERVICE_RATE)
                await self.timeout(svc)
                self.total_service[0] += svc

    return (Customer,)


@app.cell
def _(Customer, Process, random):
    class Arrivals(Process):
        def init(self, rate, server, total_service):
            self.rate = rate
            self.server = server
            self.total_service = total_service

        async def run(self):
            while True:
                await self.timeout(random.expovariate(self.rate))
                Customer(self._env, self.server, self.total_service)

    return (Arrivals,)


@app.cell
def _(
    ARRIVAL_RATE,
    ArrivalSource,
    Environment,
    SIM_TIME,
    WINDOW,
    alt,
    math,
    pl,
):
    def comparison():
        gaps = []
        env = Environment()
        ArrivalSource(env, ARRIVAL_RATE, gaps)
        env.run(until=SIM_TIME)
        n = int(SIM_TIME / WINDOW)
        counts = [0] * n
        t = 0.0
        for g in gaps:
            t += g
            w = int(t / WINDOW)
            if w < n:
                counts[w] += 1
        freq = {}
        for c in counts:
            freq[c] = freq.get(c, 0) + 1
        lam_w = ARRIVAL_RATE * WINDOW
        return pl.DataFrame([
            {
                "k": k, 
                "observed": freq.get(k, 0) / n, 
                "theory": (lam_w**k) * math.exp(-lam_w) / math.factorial(k)
            }
            for k in range(max(counts) + 1)
        ])

    _df = comparison().unpivot(
        on=["observed", "theory"], index="k", variable_name="source", value_name="probability"
    )
    (
        alt.Chart(_df).mark_bar(opacity=0.8)
        .encode(
            x=alt.X("k:O", title=f"Arrivals per window (width {WINDOW})"),
            y=alt.Y("probability:Q", title="Probability"),
            color=alt.Color("source:N", title="Series"),
            xOffset="source:N",
            tooltip=["k:O", "source:N", "probability:Q"],
        )
        .properties(title=f"Arrival Counts: Observed vs. Poisson(λ={ARRIVAL_RATE})")
    )
    return


@app.cell
def _(Arrivals, Environment, Resource, SERVICE_RATE, SIM_TIME):
    def simulate(rho):
        env = Environment()
        server = Resource(env, capacity=1)
        total_service = [0.0]
        Arrivals(env, rho * SERVICE_RATE, server, total_service)
        env.run(until=SIM_TIME)
        busy = total_service[0] / SIM_TIME
        return {"rho_target": rho, "rho_observed": round(busy, 4), "idle_frac": round(1.0 - busy, 4)}

    return (simulate,)


@app.cell
def _(pl, simulate):
    df_sim = pl.DataFrame([simulate(rho) for rho in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]])
    df_sim
    return (df_sim,)


@app.cell
def _(alt, df_sim):
    _df_plot = df_sim.unpivot(
        on=["rho_observed", "idle_frac"],
        index="rho_target",
        variable_name="metric",
        value_name="fraction",
    )
    chart = (
        alt.Chart(_df_plot).mark_line(point=True)
        .encode(
            x=alt.X("rho_target:Q", title="Target utilization (ρ = λ/μ)"),
            y=alt.Y("fraction:Q", title="Fraction of time"),
            color=alt.Color("metric:N", title="Metric"),
            tooltip=["rho_target:Q", "metric:N", "fraction:Q"],
        )
        .properties(title="Server Utilization: Busy and Idle Fractions vs. ρ")
    )
    chart
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Understanding the Math

    ### Why inter-arrival gaps are exponential

    Divide $[0, t]$ into $n$ tiny slices of width $\Delta = t/n$. The probability of an arrival in each slice is $\approx \lambda\Delta$; slices are independent. The probability of no arrival across all $n$ slices is $(1 - \lambda t/n)^n \to e^{-\lambda t}$ as $n \to \infty$. This is $P(X > t)$ for the exponential distribution.

    ### Why mean equals standard deviation for the exponential

    If $E[X] = 1/\lambda$, then $E[X^2] = 2/\lambda^2$, so $\text{Var}(X) = 2/\lambda^2 - 1/\lambda^2 = 1/\lambda^2$ and $\text{SD}(X) = 1/\lambda = E[X]$. Equal mean and standard deviation means roughly one-third of gaps are longer than the mean.

    ### Why the busy fraction equals $\rho$

    Over a long run $T$, about $N \approx \lambda T$ customers are served, each occupying the server for mean $1/\mu$. Total service time $\approx N/\mu = \lambda T/\mu = \rho T$. Dividing by $T$ gives busy fraction $= \rho$.

    ### Why $\rho = 1$ is unstable

    At exact balance, the queue length after each service completion performs a symmetric random walk on the non-negative integers. This random walk is *null recurrent*: it returns to zero but with infinite expected return time, so the mean queue length is infinite even when arrivals and service are perfectly matched on average.
    """)
    return


if __name__ == "__main__":
    app.run()
