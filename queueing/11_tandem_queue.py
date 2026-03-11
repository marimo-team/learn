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

    import altair as alt
    import polars as pl

    from asimpy import Environment, Process, Queue

    return Environment, Process, Queue, alt, mo, pl, random


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Tandem Queue Blocking

    ## *Variability Travels Downstream*

    Two processing stages are arranged in series: Stage 1 feeds work into a bounded buffer, which feeds Stage 2. Both stages have the same mean service rate $\mu$, and the arrival rate $\lambda < \mu$ so neither stage is overloaded on average. However, Stage 1 has high variance (hyperexponential service); Stage 2 has zero variance (deterministic service).

    Even though both stages have identical mean throughput and the system is underloaded, Stage 2 sits idle for a substantial fraction of time when the buffer between them is small. The idle fraction only vanishes as the buffer size $K \to \infty$.

    High service-time variance at Stage 1 produces bursts of output—many jobs finish close together—followed by droughts. With a small buffer, the burst overflows (blocking Stage 1) and the drought starves Stage 2. Both effects reduce system throughput below what we would intuitively expect.

    ## Analysis

    For a two-stage tandem queue with a finite buffer of capacity $K$, the blocking probability at Stage 1 and the starvation probability at Stage 2 depend on the full service-time distributions, not just their means. The Kingman approximation gives the mean wait in a single G/G/1 queue as:

    $$W_q \approx \frac{\rho}{1-\rho} \cdot \frac{c_a^2 + c_s^2}{2} \cdot \frac{1}{\mu}$$

    where $c_a^2$ and $c_s^2$ are the squared coefficients of variation of inter-arrival and service times respectively. For a hyperexponential service distribution with $c_s^2 \gg 1$, waiting times are far higher than the M/M/1 formula predicts.

    In a tandem network, this extra variability propagates: the departure process of Stage 1 (which is the arrival process for Stage 2) has higher variance than Poisson when Stage 1 has high service variance. This is *Departure Process Variability Propagation* and is a key driver of manufacturing and supply-chain [bullwhip effects](https://en.wikipedia.org/wiki/Bullwhip_effect).

    ## Buffer as a Variability Absorber

    The buffer acts as a shock absorber. Each unit of additional buffer capacity $K$ reduces the starvation probability at Stage 2 by absorbing burst output from Stage 1. The marginal benefit decreases as $K$ grows, leading to a classic diminishing-returns relationship. Practitioners use this to size work-in-progress inventory (WIP) buffers in manufacturing cells.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Implementation

    - A `Source` process generates jobs with exponential inter-arrival times into an unlimited input queue.
    - `Stage1` pulls from the input queue, applies a hyperexponential service delay, and pushes to a bounded `Queue(max_capacity=K)`. If the buffer is full, Stage 1 blocks (back-pressure).
    - `Stage2` pulls from the bounded buffer, applies deterministic service, and records completion times.
    - Stage 2 idle time is measured as the wait inside `queue.get()` (time spent waiting for work to appear).

    The simulation sweeps $K$ from 1 to 21 and reports Stage 2 idle fraction.
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
        start=0.3,
        stop=0.95,
        step=0.05,
        value=0.8,
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
    MEAN_SERVICE = 1.0
    return ARRIVAL_RATE, MEAN_SERVICE, SEED, SIM_TIME


@app.cell
def _(MEAN_SERVICE, random):
    def high_variance_service():
        if random.random() < 0.80:
            return random.expovariate(5.0)
        return random.expovariate(1.0 / 4.5)

    def low_variance_service():
        return MEAN_SERVICE

    return high_variance_service, low_variance_service


@app.cell
def _(ARRIVAL_RATE, Process, random):
    class Source(Process):
        def init(self, buffer):
            self.buffer = buffer

        async def run(self):
            jid = 0
            while True:
                await self.timeout(random.expovariate(ARRIVAL_RATE))
                await self.buffer.put(jid)
                jid += 1

    return (Source,)


@app.cell
def _(Process, high_variance_service):
    class Stage1(Process):
        def init(self, inp, out, idle_tally):
            self.inp = inp
            self.out = out
            self.idle_tally = idle_tally

        async def run(self):
            while True:
                idle_start = self.now
                job = await self.inp.get()
                self.idle_tally.append(self.now - idle_start)
                await self.timeout(high_variance_service())
                await self.out.put(job)

    return (Stage1,)


@app.cell
def _(Process, low_variance_service):
    class Stage2(Process):
        def init(self, inp, idle_tally, completions):
            self.inp = inp
            self.idle_tally = idle_tally
            self.completions = completions

        async def run(self):
            while True:
                idle_start = self.now
                await self.inp.get()
                idle = self.now - idle_start
                self.idle_tally.append(idle)
                await self.timeout(low_variance_service())
                self.completions.append(self.now)

    return (Stage2,)


@app.cell
def _(Environment, Queue, SIM_TIME, Source, Stage1, Stage2):
    def simulate(buffer_capacity):
        env = Environment()
        input_q = Queue(env)
        middle_q = Queue(env, max_capacity=buffer_capacity)
        s2_idle = []
        completions = []
        Source(env, input_q)
        Stage1(env, input_q, middle_q, [])
        Stage2(env, middle_q, s2_idle, completions)
        env.run(until=SIM_TIME)
        n = len(completions)
        idle_total = sum(s2_idle)
        return {
            "buffer_capacity": buffer_capacity,
            "throughput": n / SIM_TIME,
            "stage2_idle_frac": idle_total / SIM_TIME,
            "n_completed": n,
        }

    return (simulate,)


@app.cell
def _(SEED, pl, random, simulate):
    random.seed(SEED)
    df = pl.DataFrame([simulate(buffer_capacity=k) for k in [1, 2, 3, 5, 8, 13, 21]])
    return (df,)


@app.cell(hide_code=True)
def _(ARRIVAL_RATE, MEAN_SERVICE, mo):
    mo.md(f"""
    ## Results

    Arrival rate: {ARRIVAL_RATE}, mean service per stage: {MEAN_SERVICE}

    - Stage 1: hyperexponential (high variance: 80% short mean=0.2, 20% long mean=4.5)
    - Stage 2: deterministic (zero variance)

    > As buffer capacity grows, Stage 2 idle fraction falls toward 0%.
    """)
    return


@app.cell
def _(df):
    df
    return


@app.cell
def _(alt, df):
    throughput_line = (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x=alt.X("buffer_capacity:Q", title="Buffer capacity (K)"),
            y=alt.Y("throughput:Q", title="Throughput (jobs/time)"),
            tooltip=["buffer_capacity:Q", "throughput:Q"],
        )
        .properties(title="Tandem Queue: Throughput vs. Buffer Capacity")
    )
    idle_line = (
        alt.Chart(df)
        .mark_line(point=True, strokeDash=[4, 4], color="orange")
        .encode(
            x=alt.X("buffer_capacity:Q"),
            y=alt.Y("stage2_idle_frac:Q", title="Stage 2 idle fraction"),
        )
    )
    (throughput_line + idle_line).resolve_scale(y="independent")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Understanding the Math

    ### Coefficient of variation

    The coefficient of variation (CV) of a random variable $X$ with mean $\mu$ and standard deviation $\sigma$ is defined as $c = \sigma / \mu$. It measures spread relative to the mean. A CV of 0 means the variable is deterministic: every value equals $\mu$. A CV of 1 means the spread equals the mean (the exponential distribution has CV exactly 1). A CV greater than 1 means the distribution is bursty: occasional very large values dominate, even if most values are small. The squared CV $c^2 = \sigma^2/\mu^2$ appears frequently in queueing formulas.

    ### Why high CV at Stage 1 creates bursts and droughts

    With a hyperexponential service distribution ($c_s^2 \gg 1$), Stage 1 sometimes completes several jobs in rapid succession (a burst) and sometimes spends a very long time on a single job (a drought). During a burst, jobs pile up in the buffer between stages. During a drought, Stage 2 exhausts the buffer and has to wait for Stage 1 to finish. This wastes capacity even though the system is underloaded on average.

    ### The buffer as a shock absorber

    A buffer of capacity $K$ can hold at most $K$ jobs between the two stages. It absorbs burst output from Stage 1 and releases it steadily to Stage 2. With a small buffer, a burst overflows (blocking Stage 1) or a drought empties the buffer (starving Stage 2). As $K$ grows, both effects weaken and Stage 2 idle time falls. However, the marginal benefit of each extra unit of buffer decreases. Real factories choose $K$ to balance the cost of holding inventory against the cost of machine starvation.

    ### Kingman's approximation

    For a single-stage queue with general service and arrival distributions, the approximate mean waiting time is:

    $$W_q \approx \frac{\rho}{1-\rho} \cdot \frac{c_a^2 + c_s^2}{2} \cdot \frac{1}{\mu}$$

    Here $c_a^2$ is the squared CV of inter-arrival times and $c_s^2$ is the squared CV of service times. Notice that the formula separates the utilization effect (the $\rho/(1-\rho)$ term) from the variability effect (the $(c_a^2 + c_s^2)/2$ term). When Stage 1 has $c_s^2 \gg 1$, wait time is far higher than the basic M/M/1 formula predicts, even at the same mean throughput. The mean alone does not tell you enough.

    ### Supply-chain connection

    In manufacturing, Stage 1 corresponds to a supplier and Stage 2 to a production line. High variability at the supplier forces the factory to hold large work-in-progress (WIP) buffers, tying up capital and floor space. The [Toyota Production System](https://en.wikipedia.org/wiki/Toyota_Production_System) explicitly targets CV reduction as the primary tool for shrinking necessary WIP by making every process more deterministic through standardized work and small batch sizes. The math here explains exactly why: lower $c_s^2$ directly reduces $W_q$ and the required buffer size $K$.
    """)
    return


if __name__ == "__main__":
    app.run()
