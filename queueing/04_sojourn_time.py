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
    # Sojourn Time

    ## *How Long Does a Customer Actually Spend in the System?*

    The previous scenario measured $L$, the mean number of customers in the system at any moment. This scenario measures $W$, the mean time a single customer spends from arrival to departure. This is called the *sojourn time*, *residence time*, or *response time*, and has two components:

    - $W_q$: time spent waiting in the queue because the server is busy.
    - $W_s$: time spent in service while the server is working on this customer.

    $$W = W_q + W_s$$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The Surprising Finding

    For an M/M/1 queue, the mean sojourn time is:

    $$W = \frac{1}{\mu(1 - \rho)}$$

    This blows up as $\rho \to 1$, just like $L$. But the split between waiting and service shifts dramatically as load increases.

    | $\rho$ | $W_q$ (wait) | $W_s$ (service) | $W$ (total) |
    |:---:|:---:|:---:|:---:|
    | 0.1 | 0.11 | 1.00 | 1.11 |
    | 0.5 | 1.00 | 1.00 | 2.00 |
    | 0.9 | 9.00 | 1.00 | 10.00 |

    The mean service time $W_s = 1/\mu = 1.0$ is constant: the server always takes the same average time to serve one customer. All the extra delay at high load is pure waiting: $W_q = \rho/(\mu(1-\rho))$ grows without bound while $W_s$ stays fixed. At $\rho = 0.9$, 90% of a customer's time is spent waiting for the server to become free.

    This formula is closely connected to Little's Law:

    $$L = \lambda W$$

    Plugging in $W = 1/(\mu(1-\rho))$ and $\lambda = \rho\mu$:

    $$L = \rho\mu \cdot \frac{1}{\mu(1-\rho)} = \frac{\rho}{1-\rho}$$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Implementation

    `Customer` records its arrival time, then captures the exact moment it enters service (`service_start = self.now` inside the `async with self.server:` block, which only executes once the resource is acquired). The wait time is `service_start − arrival` and the sojourn time is `departure − arrival`.

    A `Monitor` samples the `in_system` counter periodically to estimate $L$ independently. The final dataframe reports $W_q$, $W_s$, $W$, the theoretical $W$, $L$ from sampling, and $L$ from Little's Law, allowing all three to be cross-checked.
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

    service_rate_slider = mo.ui.slider(
        start=1.0,
        stop=5.0,
        step=0.01,
        value=1.0,
        label="Service rate",
    )

    sample_interval_slider = mo.ui.slider(
        start=1.0,
        stop=5.0,
        step=1.0,
        value=1.0,
        label="Sample interval",
    )

    seed_input = mo.ui.number(
        value=192,
        step=1,
        label="Random seed",
    )

    run_button = mo.ui.button(label="Run simulation")

    mo.vstack([
        sim_time_slider,
        service_rate_slider,
        sample_interval_slider,
        seed_input,
        run_button,
    ])
    return (
        sample_interval_slider,
        seed_input,
        service_rate_slider,
        sim_time_slider,
    )


@app.cell
def _(
    sample_interval_slider,
    seed_input,
    service_rate_slider,
    sim_time_slider,
):
    SIM_TIME = int(sim_time_slider.value)
    SERVICE_RATE = float(service_rate_slider.value)
    SAMPLE_INTERVAL = float(sample_interval_slider.value)
    SEED = int(seed_input.value)
    return SAMPLE_INTERVAL, SEED, SERVICE_RATE, SIM_TIME


@app.cell
def _(Process, SERVICE_RATE, random):
    class Customer(Process):
        def init(self, server, in_system, sojourn_times, wait_times):
            self.server = server
            self.in_system = in_system
            self.sojourn_times = sojourn_times
            self.wait_times = wait_times

        async def run(self):
            arrival = self.now
            self.in_system[0] += 1
            async with self.server:
                service_start = self.now
                await self.timeout(random.expovariate(SERVICE_RATE))
            self.in_system[0] -= 1
            self.sojourn_times.append(self.now - arrival)
            self.wait_times.append(service_start - arrival)

    return (Customer,)


@app.cell
def _(Customer, Process, random):
    class Arrivals(Process):
        def init(self, rate, server, in_system, sojourn_times, wait_times):
            self.rate = rate
            self.server = server
            self.in_system = in_system
            self.sojourn_times = sojourn_times
            self.wait_times = wait_times

        async def run(self):
            while True:
                await self.timeout(random.expovariate(self.rate))
                Customer(self._env, self.server, self.in_system, self.sojourn_times, self.wait_times)

    return (Arrivals,)


@app.cell
def _(Process, SAMPLE_INTERVAL):
    class Monitor(Process):
        def init(self, in_system, samples):
            self.in_system = in_system
            self.samples = samples

        async def run(self):
            while True:
                self.samples.append(self.in_system[0])
                await self.timeout(SAMPLE_INTERVAL)

    return (Monitor,)


@app.cell
def _(
    Arrivals,
    Environment,
    Monitor,
    Resource,
    SERVICE_RATE,
    SIM_TIME,
    statistics,
):
    def simulate(rho):
        rate = rho * SERVICE_RATE
        env = Environment()
        server = Resource(env, capacity=1)
        in_system = [0]
        sojourn_times = []
        wait_times = []
        samples = []
        Arrivals(env, rate, server, in_system, sojourn_times, wait_times)
        Monitor(env, in_system, samples)
        env.run(until=SIM_TIME)
        mean_W = statistics.mean(sojourn_times)
        mean_Wq = statistics.mean(wait_times)
        mean_Ws = mean_W - mean_Wq
        mean_L = statistics.mean(samples)
        lam = len(sojourn_times) / SIM_TIME
        return {
            "rho": rho,
            "mean_Wq": round(mean_Wq, 4),
            "mean_Ws": round(mean_Ws, 4),
            "mean_W": round(mean_W, 4),
            "theory_W": round(1.0 / (SERVICE_RATE * (1.0 - rho)), 4),
            "L_sampled": round(mean_L, 4),
            "L_little": round(lam * mean_W, 4),
        }

    return (simulate,)


@app.cell
def _(SEED, pl, random, simulate):
    random.seed(SEED)
    df = pl.DataFrame([simulate(rho) for rho in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]])
    df
    return (df,)


@app.cell
def _(alt, df):
    df_plot = df.select(["rho", "mean_Wq", "mean_Ws"]).unpivot(
        on=["mean_Wq", "mean_Ws"],
        index="rho",
        variable_name="component",
        value_name="time",
    )
    chart = (
        alt.Chart(df_plot)
        .mark_area()
        .encode(
            x=alt.X("rho:Q", title="Utilization (ρ)"),
            y=alt.Y("time:Q", title="Mean time", stack="zero"),
            color=alt.Color("component:N", title="Component"),
            tooltip=["rho:Q", "component:N", "time:Q"],
        )
        .properties(title="Sojourn Time Components: Wq (waiting) + Ws (service) = W")
    )
    chart
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Understanding the Math

    ### Why $W_s = 1/\mu$ regardless of $\rho$

    Service time is drawn from an exponential distribution with rate $\mu$, so its mean is $1/\mu$. This is a property of the distribution, not of the queue. No matter how busy the server is, once it starts serving you it takes on average $1/\mu$ time.

    ### Deriving $W_q$

    Since $W = W_q + W_s$ and $W_s = 1/\mu$:

    $$W_q = W - W_s = \frac{1}{\mu(1-\rho)} - \frac{1}{\mu}
          = \frac{1}{\mu}\left(\frac{1}{1-\rho} - 1\right)
          = \frac{1}{\mu} \cdot \frac{\rho}{1-\rho}
          = \frac{\rho}{\mu(1-\rho)}$$

    Note that $W_q = \rho \cdot W$: at high load, almost all of $W$ is waiting.

    ### Units check

    $\lambda$ has units of [customers/time]; $W$ has units of [time]; so $L = \lambda W$ has units of [customers/time $\times$ time] $=$ [customers]. This count of people is dimensionless, as it should be. Checking units this way is a quick sanity test whenever you apply Little's Law to a real problem.
    """)
    return


if __name__ == "__main__":
    app.run()
