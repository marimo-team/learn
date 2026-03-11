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

    from asimpy import Environment, Process, Resource

    return Environment, Process, Resource, alt, mo, pl, random, statistics


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Pooled vs. Separate Queues

    ## *Why Airports Switched to Single Lines*

    A facility has two identical servers. Customers arrive as a Poisson process and each needs one server for an exponentially distributed service time. Which queueing discipline should the facility use?

    - Separate queues: each server has its own dedicated line; customers randomly pick a line on arrival and cannot switch.
    - Pooled queue: a single shared line feeds whichever server becomes free first.

    It turns out that pooling the queues is always better, even though both systems have identical total arrival rate, identical per-server service rate, and identical utilization $\rho$. The pooled system consistently produces shorter mean wait times, often by a factor of two or more at moderate utilization. The reason is that separate queues waste servers' idle time. In separate queues, one server may be idle while customers wait in the other line. Pooling eliminates this mismatch: a free server always serves the next waiting customer.

    ### Why Separate Queues Persist

    Despite being provably worse, separate queues feel fairer because customers can see their progress. Single lines eliminate the anxiety of watching the other queue move faster, but historically customers resisted them until airlines and banks demonstrated the improvement empirically in the 1960s–70s.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Implementation

    This tutorial explores this finding using two simulations run with identical random seeds:

    1. Pooled: `Resource(capacity=2)` with one arrival stream. The resource grants access to whichever capacity slot is free.
    2. Separate: two `Resource(capacity=1)` instances. Arrivals call `random.choice` to pick a server and cannot switch even if it is slower.

    The mean sojourn time is collected across a sweep of utilization levels $\rho$.
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

    arrival_rate_slider = mo.ui.slider(
        start=0.5,
        stop=1.9,
        step=0.05,
        value=1.8,
        label="Arrival rate",
    )

    seed_input = mo.ui.number(
        value=192,
        step=1,
        label="Random seed",
    )

    run_button = mo.ui.button(label="Run simulation")

    mo.vstack([
        sim_time_slider,
        arrival_rate_slider,
        seed_input,
        run_button,
    ])
    return arrival_rate_slider, seed_input, sim_time_slider


@app.cell
def _(arrival_rate_slider, seed_input, sim_time_slider):
    SIM_TIME = int(sim_time_slider.value)
    ARRIVAL_RATE = float(arrival_rate_slider.value)
    SEED = int(seed_input.value)
    SERVICE_RATE = 1.0
    N_SERVERS = 2
    RHO = ARRIVAL_RATE / (N_SERVERS * SERVICE_RATE)
    return ARRIVAL_RATE, N_SERVERS, RHO, SEED, SERVICE_RATE, SIM_TIME


@app.cell
def _(Process, SERVICE_RATE, random):
    class Customer(Process):
        def init(self, server, sojourn_times):
            self.server = server
            self.sojourn_times = sojourn_times

        async def run(self):
            arrival = self.now
            async with self.server:
                await self.timeout(random.expovariate(SERVICE_RATE))
            self.sojourn_times.append(self.now - arrival)

    return (Customer,)


@app.cell
def _(Customer, Process, random):
    class PooledArrivals(Process):
        def init(self, arrival_rate, server, sojourn_times):
            self.arrival_rate = arrival_rate
            self.server = server
            self.sojourn_times = sojourn_times

        async def run(self):
            while True:
                await self.timeout(random.expovariate(self.arrival_rate))
                Customer(self._env, self.server, self.sojourn_times)

    return (PooledArrivals,)


@app.cell
def _(Customer, Process, random):
    class SeparateArrivals(Process):
        def init(self, arrival_rate, servers, sojourn_times):
            self.arrival_rate = arrival_rate
            self.servers = servers
            self.sojourn_times = sojourn_times

        async def run(self):
            while True:
                await self.timeout(random.expovariate(self.arrival_rate))
                server = random.choice(self.servers)
                Customer(self._env, server, self.sojourn_times)

    return (SeparateArrivals,)


@app.cell
def _(
    ARRIVAL_RATE,
    Environment,
    N_SERVERS,
    PooledArrivals,
    Resource,
    SEED,
    SIM_TIME,
    random,
    statistics,
):
    def run_pooled(arrival_rate=ARRIVAL_RATE):
        random.seed(SEED)
        sojourn_times = []
        env = Environment()
        shared_server = Resource(env, capacity=N_SERVERS)
        PooledArrivals(env, arrival_rate, shared_server, sojourn_times)
        env.run(until=SIM_TIME)
        return statistics.mean(sojourn_times)

    return (run_pooled,)


@app.cell
def _(
    ARRIVAL_RATE,
    Environment,
    N_SERVERS,
    Resource,
    SEED,
    SIM_TIME,
    SeparateArrivals,
    random,
    statistics,
):
    def run_separate(arrival_rate=ARRIVAL_RATE):
        random.seed(SEED)
        sojourn_times = []
        env = Environment()
        servers = [Resource(env, capacity=1) for _ in range(N_SERVERS)]
        SeparateArrivals(env, arrival_rate, servers, sojourn_times)
        env.run(until=SIM_TIME)
        return statistics.mean(sojourn_times)

    return (run_separate,)


@app.cell
def _(ARRIVAL_RATE, N_SERVERS, SERVICE_RATE, pl, run_pooled, run_separate):
    def sweep():
        sweep_rows = []
        for rho in [0.5, 0.6, 0.7, 0.8, 0.9]:
            rate = rho * N_SERVERS * SERVICE_RATE
            pw = run_pooled(arrival_rate=rate)
            sw = run_separate(arrival_rate=rate)
            sweep_rows.append({"rho": rho, "pooled_W": pw, "separate_W": sw, "ratio": sw / pw})
        return pl.DataFrame(sweep_rows)

    df_sweep = sweep()
    pooled_W = run_pooled(arrival_rate=ARRIVAL_RATE)
    separate_W = run_separate(arrival_rate=ARRIVAL_RATE)
    return df_sweep, pooled_W, separate_W


@app.cell(hide_code=True)
def _(N_SERVERS, RHO, SERVICE_RATE, mo, pooled_W, separate_W):
    mo.md(f"""
    ## Results

    {N_SERVERS} servers, service rate {SERVICE_RATE}, utilisation ρ = {RHO:.2f}

    At ρ = {RHO:.2f}: pooled W = {pooled_W:.3f}, separate W = {separate_W:.3f}
    — separate queues are **{separate_W / pooled_W:.2f}×** slower
    """)
    return


@app.cell
def _(df_sweep):
    df_sweep
    return


@app.cell
def _(alt, df_sweep):
    df_plot = df_sweep.unpivot(
        on=["pooled_W", "separate_W"], index="rho", variable_name="system", value_name="W"
    )
    chart = (
        alt.Chart(df_plot)
        .mark_line(point=True)
        .encode(
            x=alt.X("rho:Q", title="Utilization per server (ρ)"),
            y=alt.Y("W:Q", title="Mean sojourn time (W)"),
            color=alt.Color("system:N", title="Queue type"),
            tooltip=["rho:Q", "system:N", "W:Q"],
        )
        .properties(title="Pooled vs. Separate Queues: Mean Sojourn Time")
    )
    chart
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Understanding the Math

    ### Why pooling always wins

    Two separate M/M/1 queues each running at utilization $\rho$ have mean sojourn time $W_{\text{sep}} = 1/(\mu(1-\rho))$. A pooled M/M/2 queue with the same total arrival rate has strictly lower mean sojourn time for every value of $0 < \rho < 1$. The proof uses the [Erlang-C formula](https://en.wikipedia.org/wiki/Erlang_(unit)#Erlang_C_formula), but the intuition is simpler: pooling converts two independent random processes into one, and the combined queue can exploit any idle capacity instantly. At $\rho = 0.8$, separate queues give roughly twice the mean wait of a pooled queue.

    ### Connection to variance reduction

    Think of the service delivered in a time window by two separate servers as two independent random variables $X_1$ and $X_2$. Their average $(X_1 + X_2)/2$ has variance $\sigma^2/2$, which is half the variance of either component alone. Pooling achieves something similar: by combining demand into one stream served by both servers, the system smooths out random fluctuations. The pooled queue is, in effect, averaging over both servers' idle periods instead of locking each idle period to a single lane.

    ### Rule of thumb

    At $\rho = 0.8$, separate queues produce roughly double the mean wait of a pooled queue. This factor grows as $\rho$ increases, because the $(1-\rho)$ term in the denominator amplifies any wasted capacity. The lesson: whenever you can route demand flexibly to a shared resource, do it.
    """)
    return


if __name__ == "__main__":
    app.run()
