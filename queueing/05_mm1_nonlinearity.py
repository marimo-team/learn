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
    # M/M/1 Queue Nonlinearity

    ## *The 90% Utilization Trap*

    A single server handles jobs that arrive randomly and take a random amount of time to process. If both inter-arrival times and service times follow exponential distributions, this is called an *M/M/1 queue*, and is the simplest model in queueing theory.

    Managers often treat utilization linearly: "90% busy is only a little worse than 80% busy." The M/M/1 formula shows this intuition is badly wrong. The mean number of jobs in the system (waiting and being served) is:

    $$L = \frac{\rho}{1 - \rho}$$

    where $\rho = \lambda / \mu$ is the utilization ratio (arrival rate divided by service rate). The mean time a job spends in the system follows from Little's Law $L = \lambda W$:

    $$W = \frac{1}{\mu - \lambda} = \frac{1}{\mu(1 - \rho)}$$

    The denominator $(1 - \rho)$ causes both $L$ and $W$ to blow up as $\rho \to 1$.

    | $\rho$ | $L = \rho/(1-\rho)$ | Marginal $\Delta L$ per 0.1 step |
    |-------:|--------------------:|--------------------------------:|
    |  0.50  |               1.00  |                            —    |
    |  0.60  |               1.50  |                          +0.50  |
    |  0.70  |               2.33  |                          +0.83  |
    |  0.80  |               4.00  |                          +1.67  |
    |  0.90  |               9.00  |                          +5.00  |

    Each equal step in $\rho$ produces a larger jump in queue length than the previous step. Going from 80% to 90% utilization adds more queue length than going from 0% to 80% combined. This happens because the queue is stabilized by the gaps in service capacity. When $\rho = 0.9$, only 10% of capacity is slack. Any random burst of arrivals takes far longer to drain than when $\rho = 0.5$ and 50% of capacity is slack. The system spends most of its time recovering from bursts rather than idling.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Implementation

    The simulation uses a single `Resource(capacity=1)` as the server. A generator process creates customers with inter-arrival gaps drawn from $\text{Exp}(\lambda)$. Each customer records its arrival time, waits to acquire the server, receives $\text{Exp}(\mu)$ service, and logs its total sojourn time. The mean queue length $L$ is computed via Little's Law from the observed mean sojourn time.
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

    seed_input = mo.ui.number(
        value=192,
        step=1,
        label="Random seed",
    )

    run_button = mo.ui.run_button(label="Run simulation")

    mo.vstack([
        sim_time_slider,
        service_rate_slider,
        seed_input,
        run_button,
    ])
    return seed_input, service_rate_slider, sim_time_slider


@app.cell
def _(seed_input, service_rate_slider, sim_time_slider):
    SIM_TIME = int(sim_time_slider.value)
    SERVICE_RATE = float(service_rate_slider.value)
    SEED = int(seed_input.value)
    return SEED, SERVICE_RATE, SIM_TIME


@app.cell
def _(Process, random):
    class Customer(Process):
        def init(self, server, service_rate, sojourn_times):
            self.server = server
            self.service_rate = service_rate
            self.sojourn_times = sojourn_times

        async def run(self):
            arrival = self.now
            async with self.server:
                await self.timeout(random.expovariate(self.service_rate))
            self.sojourn_times.append(self.now - arrival)

    return (Customer,)


@app.cell
def _(Customer, Process, random):
    class ArrivalStream(Process):
        def init(self, arrival_rate, service_rate, server, sojourn_times):
            self.arrival_rate = arrival_rate
            self.service_rate = service_rate
            self.server = server
            self.sojourn_times = sojourn_times

        async def run(self):
            while True:
                await self.timeout(random.expovariate(self.arrival_rate))
                Customer(self._env, self.server, self.service_rate, self.sojourn_times)

    return (ArrivalStream,)


@app.cell
def _(
    ArrivalStream,
    Environment,
    Resource,
    SERVICE_RATE,
    SIM_TIME,
    statistics,
):
    def simulate(rho):
        arrival_rate = rho * SERVICE_RATE
        sojourn_times = []
        env = Environment()
        server = Resource(env, capacity=1)
        ArrivalStream(env, arrival_rate, SERVICE_RATE, server, sojourn_times)
        env.run(until=SIM_TIME)
        mean_W = statistics.mean(sojourn_times) if sojourn_times else 0.0
        sim_L = arrival_rate * mean_W
        theory_L = rho / (1.0 - rho)
        return sim_L, theory_L

    return (simulate,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Simulated vs. Theoretical Queue Length
    """)
    return


@app.cell
def _(SEED, pl, random, simulate):
    def sweep():
        random.seed(SEED)
        rhos = [0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95]
        sweep_rows = []
        for rho in rhos:
            sim_L, theory_L = simulate(rho)
            pct = 100.0 * (sim_L - theory_L) / theory_L
            sweep_rows.append({"rho": rho, "theory_L": theory_L, "sim_L": sim_L, "pct_error": pct})
        return pl.DataFrame(sweep_rows)

    df_sweep = sweep()
    df_sweep
    return (df_sweep,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Marginal Increase in L per 0.1 Step in ρ (Theory)
    """)
    return


@app.cell
def _(pl):
    def marginal():
        marginal_rows = []
        prev_L, prev_rho = None, None
        for rho in [0.5, 0.6, 0.7, 0.8, 0.9]:
            theory_L = rho / (1.0 - rho)
            if prev_L is not None:
                marginal_rows.append({"rho_from": prev_rho, "rho_to": rho, "delta_L": round(theory_L - prev_L, 4)})
            prev_L, prev_rho = theory_L, rho
        return pl.DataFrame(marginal_rows)

    df_marginal = marginal()
    df_marginal
    return


@app.cell
def _(alt, df_sweep):
    df_plot = df_sweep.unpivot(
        on=["theory_L", "sim_L"], index="rho", variable_name="source", value_name="L"
    )
    chart = (
        alt.Chart(df_plot)
        .mark_line(point=True)
        .encode(
            x=alt.X("rho:Q", title="Utilization (ρ)"),
            y=alt.Y("L:Q", title="Mean queue length (L)"),
            color=alt.Color("source:N", title="Source"),
            tooltip=["rho:Q", "source:N", "L:Q"],
        )
        .properties(title="M/M/1 Queue Length vs. Utilization")
    )
    chart
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Key Takeaway

    For any system approximated by an M/M/1 queue, **never target utilization above 80–85%** if low latency matters. The last few percent of throughput come at an enormous cost in queue length and wait time.
    """)
    return


if __name__ == "__main__":
    app.run()
