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


@app.cell(hide_code=True)
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
    # Little's Law

    ## *The Universal Conservation Law of Queues*

    Little's Law states that in a stable system, L = λW, where:

    - L = mean number of customers in the system
    - λ = mean arrival rate
    - W = mean time a customer spends in the system

    This notebook verifies Little's Law across three different queue configurations.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Why It Is Surprising

    Little's Law holds without any assumptions about the distribution of arrival rates or service times. It does not matter whether arrivals are Poisson, deterministic, or correlated, whether service times are exponential, constant, or heavy-tailed, whether there is one server or a hundred, or what scheduling discipline is used (FIFO, LIFO, random, or priority). As long as the system is stable and stationary, $L = \lambda W$. This universality is remarkable because almost every other formula in queueing theory *does* depend on distributional assumptions.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Practical Use

    Because $L = \lambda W$ is universal, it can be used to measure hard-to-observe quantities from easy-to-observe ones. For example, the mean number of requests in a web server ($L$) and the observed request rate ($\lambda$) immediately give the mean response time ($W = L/\lambda$) without needing to instrument individual request latencies.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Proof Sketch

    Consider a flow diagram where time runs horizontally and each customer traces a horizontal line from arrival to departure. The area under all lines equals both:

    - $\sum_i W_i$ (sum of individual sojourn times), and
    - $\int_0^T L(t)\,dt$ (integral of instantaneous queue length).

    Dividing both sides by $T$ and taking $T \to \infty$:

    $$\bar{L} = \lambda \bar{W}$$

    The argument is purely combinatorial: no probability is needed.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Three Configurations, One Law

    The simulation verifies Little's Law independently for:

    | Configuration | Arrivals     | Service Kind | Service       | Number of Servers |
    |:-------------|:------------|:--------------|:--------|---:|
    | M/M/1        | Poisson($\lambda$) | Random | Exp($\mu$)  | 1       |
    | M/D/1        | Poisson($\lambda$) | Deterministic | $1/\mu$ | 1 |
    | M/M/3        | Poisson($\lambda$) | Random | Exp($\mu$)  | 3       |

    For each configuration, $L$ is estimated two independent ways:

    1. Direct sampling: a monitor process samples the number of customers in the system every unit of simulated time; $L \approx \bar{n}_{\text{samples}}$.
    2. Little's Law: throughput $\lambda$ (completed jobs / total time) and mean sojourn $W$ are measured; $L_{\text{Little}} = \lambda W$.

    The two estimates agree to within simulation noise for all three very different configurations.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Random Processes
    """)
    return


@app.cell
def _(Process, random):
    class RandomCustomer(Process):
        def init(self, server, in_system, sojourn_times):
            self.server = server
            self.in_system = in_system
            self.sojourn_times = sojourn_times

        async def run(self):
            arrival = self.now
            self.in_system[0] += 1
            async with self.server:
                await self.timeout(random.expovariate(1.0))
            self.in_system[0] -= 1
            self.sojourn_times.append(self.now - arrival)

    return (RandomCustomer,)


@app.cell
def _(Process, RandomCustomer, random):
    class RandomArrivals(Process):
        def init(self, rate, server, in_system, sojourn_times):
            self.rate = rate
            self.server = server
            self.in_system = in_system
            self.sojourn_times = sojourn_times

        async def run(self):
            while True:
                await self.timeout(random.expovariate(self.rate))
                RandomCustomer(self._env, self.server, self.in_system, self.sojourn_times)

    return (RandomArrivals,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Deterministic Processes
    """)
    return


@app.cell
def _(Process):
    class DeterministicCustomer(Process):
        def init(self, server, service_time, in_system, sojourn_times):
            self.server = server
            self.service_time = service_time
            self.in_system = in_system
            self.sojourn_times = sojourn_times

        async def run(self):
            arrival = self.now
            self.in_system[0] += 1
            async with self.server:
                await self.timeout(self.service_time)
            self.in_system[0] -= 1
            self.sojourn_times.append(self.now - arrival)

    return (DeterministicCustomer,)


@app.cell
def _(DeterministicCustomer, Process, random):
    class DeterministicArrivals(Process):
        def init(self, rate, server, in_system, sojourn_times):
            self.rate = rate
            self.server = server
            self.in_system = in_system
            self.sojourn_times = sojourn_times
            self.service_time = 1.0

        async def run(self):
            while True:
                await self.timeout(random.expovariate(self.rate))
                DeterministicCustomer(self._env, self.server, self.service_time, self.in_system, self.sojourn_times)

    return (DeterministicArrivals,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Comparison
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    sim_time_slider = mo.ui.slider(
        start=0,
        stop=100_000,
        step=1_000,
        value=20_000,
        label="Simulation time",
    )

    sample_interval_slider = mo.ui.slider(
        start=1,
        stop=5,
        step=1,
        value=1,
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
        sample_interval_slider,
        seed_input,
        run_button,
    ])
    return run_button, sample_interval_slider, seed_input, sim_time_slider


@app.cell
def _(sample_interval_slider, sim_time_slider):
    SIM_TIME = int(sim_time_slider.value)
    SAMPLE_INTERVAL = int(sample_interval_slider.value)
    return SAMPLE_INTERVAL, SIM_TIME


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
def _(Monitor, SIM_TIME, statistics):
    def verify(label, env, in_system, sojourn_times, samples, arrival_rate):
        Monitor(env, in_system, samples)
        env.run(until=SIM_TIME)
        L_direct = statistics.mean(samples)
        W = statistics.mean(sojourn_times)
        n = len(sojourn_times)
        lam = n / SIM_TIME
        L_little = lam * W
        error = 100.0 * (L_little - L_direct) / L_direct
        return {
            "label": label,
            "lambda_obs": lam,
            "mean_W": W,
            "L_direct": L_direct,
            "L_little": L_little,
            "error_pct": error,
        }

    return (verify,)


@app.cell
def _(Environment, Resource, verify):
    def simulate(title, rows, arrivalsCls, lam, capacity):
        in_system = [0]
        sojourns = []
        samples = []
        env = Environment()
        server = Resource(env, capacity=capacity)
        arrivalsCls(env, lam, server, in_system, sojourns)
        rows.append(verify(title, env, in_system, sojourns, samples, lam))

    return (simulate,)


@app.cell
def _(
    DeterministicArrivals,
    RandomArrivals,
    pl,
    random,
    run_button,
    seed_input,
    simulate,
):
    run_button

    random.seed(int(seed_input.value))
    rows = []

    simulate("M/M/1 (rho=0.70, 1 server)", rows, RandomArrivals, 0.7, 1)
    simulate("M/D/1 (rho=0.70, deterministic service)", rows, DeterministicArrivals, 0.7, 1)
    simulate("M/M/3 (rho=0.80 per server, 3 servers)", rows, RandomArrivals, 2.4, 3)

    df = pl.DataFrame(rows)
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Verification: L = λW
    """)
    return


@app.cell
def _(df):
    df
    return


@app.cell
def _(alt, df, pl):
    points = (
        alt.Chart(df)
        .mark_point(size=100, filled=True)
        .encode(
            x=alt.X("L_direct:Q", title="L (direct sample)"),
            y=alt.Y("L_little:Q", title="L = λW (Little's Law)"),
            color=alt.Color("label:N", title="Configuration"),
            tooltip=["label:N", "L_direct:Q", "L_little:Q", "error_pct:Q"],
        )
    )
    max_val = max(df["L_direct"].to_list()) * 1.1
    diagonal = (
        alt.Chart(pl.DataFrame({"x": [0.0, max_val], "y": [0.0, max_val]}))
        .mark_line(color="gray", strokeDash=[4, 4])
        .encode(x="x:Q", y="y:Q")
    )
    (diagonal + points).properties(title="Little's Law: Direct Sample vs. λW")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Understanding the Math

    ### The area argument made concrete

    Draw a horizontal time axis from $t = 0$ to $t = T$. Each customer gets a horizontal bar starting at their arrival time and ending at their departure time. The length of their bar is exactly their sojourn time $W_i$ — the total time they spend in the system. At any moment $t$, the number of bars that cross that vertical slice is exactly $L(t)$, the instantaneous number of customers in the system.

    Now compute the total area under all the bars in two different ways. First, add up the lengths of all the bars: total area $= \sum_i W_i$. Second, integrate the height of the stack over time: total area $= \int_0^T L(t)\,dt$. These are the same area, so $\sum_i W_i = \int_0^T L(t)\,dt$.

    Divide both sides by $T$. The right side becomes the time-average $\bar{L}$. The left side becomes $(n/T) \cdot \bar{W}$, where $n$ is the total number of customers and $\bar{W}$ is their mean sojourn time. As $T \to \infty$, $n/T \to \lambda$ (the long-run arrival rate). That gives $\bar{L} = \lambda \bar{W}$, which is Little's Law.

    ### No distribution required

    The argument above uses only geometry. There is no probability distribution, no exponential assumption, no Poisson process. The shape of each bar (i.e., how long each customer takes) can be anything. This is why the law applies to M/M/1, M/D/1, M/M/3, and every other configuration equally.

    ### Using it in practice

    Suppose you run a web service. Your monitoring dashboard shows $\lambda = 500$ requests per second and your server logs show a mean response time of $W = 20$ milliseconds. Little's Law immediately tells you that the mean number of active requests in the system is $L = \lambda W = 500 \times 0.02 = 10$ requests. Alternatively, if you observe $L$ and $\lambda$ but not individual response times, you get $W = L/\lambda$ without any per-request timing instrumentation.

    ### Units check

    $\lambda$ has units of customers per unit time; $W$ has units of time; so $L = \lambda W$ is dimensionless — a pure count of customers. Always verify units when applying Little's Law to a new problem: if your units do not cancel correctly, you have applied the law incorrectly.

    ### Stability condition

    Little's Law requires the system to reach steady state: over the long run, arrivals and departures must balance. If $\lambda > \mu$ (the arrival rate exceeds the service rate), the queue grows without bound. $L = \infty$ and $W = \infty$; the law still holds, but it tells you the system is broken, not that it is well-behaved.
    """)
    return


if __name__ == "__main__":
    app.run()
