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

    from asimpy import Environment, Process, Queue

    return Environment, Process, Queue, alt, mo, pl, random, statistics


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Priority Starvation

    ## *When High-Priority Traffic Crowds Out Low-Priority Jobs*

    A single server processes two job classes:

    - High-priority jobs (class H) arrive frequently and are served quickly.
    - Low-priority jobs (class L) arrive rarely and take longer to serve.

    The server always picks the highest-priority job available. Total server utilization $\rho = \rho_H + \rho_L < 1$, so the server has spare capacity on average. Yet low-priority jobs can wait far longer than the utilization level suggests they should.

    ### Static Priority: Starvation at Moderate Load

    With a static priority queue, high-priority jobs *never* yield to low-priority ones. Even when $\rho_H < 1$, high-priority bursts can lock out low-priority jobs for extended periods. The mean wait for low-priority jobs under a static non-preemptive priority queue is:

    $$W_L = \frac{\overline{s}_L}{1-\rho_H} \cdot \frac{1}{1-\rho_H-\rho_L}$$

    This diverges as $\rho_H \to 1$ independently of $\rho_L$. As $\rho_H$ approaches 100%, low-priority jobs wait arbitrarily long, even if only a few low-priority jobs ever arrive.

    ### Aging: Solving Starvation Creates Oscillation

    The standard remedy for starvation is *priority aging*: a waiting job's priority improves over time until it eventually beats even high-priority arrivals. This guarantees finite wait for all jobs.

    However, aging introduces a new pathology. When aged low-priority jobs finally burst through, they occupy the server and leave a backlog of high-priority jobs waiting. The high-priority queue then drains, and the cycle repeats — producing oscillating bursts rather than smooth, uniform service.

    ### Intuition

    Suppose H jobs arrive in random bursts. During a burst, the server never pauses for L jobs. An L job unlucky enough to arrive at the start of a long burst must wait for every H job in that burst to be served before getting its turn. As bursts grow more frequent (larger $\rho_H$), the expected burst length grows, and with it the expected wait for that unlucky L job. The math confirms: starvation is a real risk at moderate $\rho_H$, not just at extreme loads.

    ### What aging does

    Aging assigns each waiting L job a maximum patience time $T_{\max}$. After waiting $T_{\max}$, the job is promoted to high priority. This caps the worst-case wait: no L job can wait longer than $T_{\max}$ plus one service time. Mathematically, the effective $W_L$ is bounded by $T_{\max} + 1/\mu_L$.

    ## Practical Implications

    Priority queues appear throughout computing:

    - OS scheduling: interactive processes (high priority) vs. batch jobs (low priority). Linux uses dynamic priority aging (nice values + sleep bonuses) to avoid starvation.
    - Network QoS: real-time traffic (VoIP, video) vs. bulk data. Traffic shaping with Deficit Round Robin (DRR) or Weighted Fair Queuing (WFQ) guarantees bandwidth shares without starvation.
    - Database query planning: short OLTP queries vs. long OLAP queries. Resource groups and query timeouts implement a form of aging.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Implementation

    Two runs are compared:

    1. Static priority: H jobs are inserted as `(0, ...)` and L jobs as `(1, ...)` into a `Queue(priority=True)`. The server always picks the smallest key first, so H jobs are always served before L jobs.
    2. Aging: an `Ager` process wakes up every `AGING_INTERVAL` time units, inspects waiting L jobs, and promotes sufficiently old ones by reducing their priority key until it falls below the H threshold and they move into the server's feed queue.
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

    aging_threshold_slider = mo.ui.slider(
        start=1.0,
        stop=60.0,
        step=1.0,
        value=15.0,
        label="Aging threshold",
    )

    seed_input = mo.ui.number(
        value=192,
        step=1,
        label="Random seed",
    )

    run_button = mo.ui.run_button(label="Run simulation")

    mo.vstack([
        sim_time_slider,
        aging_threshold_slider,
        seed_input,
        run_button,
    ])
    return aging_threshold_slider, seed_input, sim_time_slider


@app.cell
def _(aging_threshold_slider, seed_input, sim_time_slider):
    SIM_TIME = int(sim_time_slider.value)
    AGING_THRESHOLD = float(aging_threshold_slider.value)
    SEED = int(seed_input.value)
    SERVICE_RATE_HI = 2.0
    SERVICE_RATE_LO = 1.0
    ARRIVAL_RATE_LO = 0.2
    return (
        AGING_THRESHOLD,
        ARRIVAL_RATE_LO,
        SEED,
        SERVICE_RATE_HI,
        SERVICE_RATE_LO,
        SIM_TIME,
    )


@app.cell
def _(Process):
    class StaticPriorityServer(Process):
        def init(self, hi_q, lo_q, sojourn_hi, sojourn_lo):
            self.hi_q = hi_q
            self.lo_q = lo_q
            self.sojourn_hi = sojourn_hi
            self.sojourn_lo = sojourn_lo

        async def _serve(self, arrival, svc, record):
            await self.timeout(svc)
            record.append(self.now - arrival)

        async def run(self):
            while True:
                if not self.hi_q.is_empty():
                    arrival, svc = await self.hi_q.get()
                    await self._serve(arrival, svc, self.sojourn_hi)
                elif not self.lo_q.is_empty():
                    arrival, svc = await self.lo_q.get()
                    await self._serve(arrival, svc, self.sojourn_lo)
                else:
                    await self.timeout(0.01)

    return (StaticPriorityServer,)


@app.cell
def _(AGING_THRESHOLD, Process):
    class AgingServer(Process):
        def init(self, hi_q, lo_q, sojourn_hi, sojourn_lo):
            self.hi_q = hi_q
            self.lo_q = lo_q
            self.sojourn_hi = sojourn_hi
            self.sojourn_lo = sojourn_lo

        async def run(self):
            while True:
                lo_aged = (
                    not self.lo_q.is_empty()
                    and self.now - self.lo_q._items[0][0] >= AGING_THRESHOLD
                )
                if lo_aged:
                    arrival, svc = await self.lo_q.get()
                    await self.timeout(svc)
                    self.sojourn_lo.append(self.now - arrival)
                elif not self.hi_q.is_empty():
                    arrival, svc = await self.hi_q.get()
                    await self.timeout(svc)
                    self.sojourn_hi.append(self.now - arrival)
                elif not self.lo_q.is_empty():
                    arrival, svc = await self.lo_q.get()
                    await self.timeout(svc)
                    self.sojourn_lo.append(self.now - arrival)
                else:
                    await self.timeout(0.01)

    return (AgingServer,)


@app.cell
def _(Process, SERVICE_RATE_HI, random):
    class HiSource(Process):
        def init(self, rate, q):
            self.rate = rate
            self.q = q

        async def run(self):
            while True:
                await self.timeout(random.expovariate(self.rate))
                svc = random.expovariate(SERVICE_RATE_HI)
                await self.q.put((self.now, svc))

    return (HiSource,)


@app.cell
def _(ARRIVAL_RATE_LO, Process, SERVICE_RATE_LO, random):
    class LoSource(Process):
        def init(self, q):
            self.q = q

        async def run(self):
            while True:
                await self.timeout(random.expovariate(ARRIVAL_RATE_LO))
                svc = random.expovariate(SERVICE_RATE_LO)
                await self.q.put((self.now, svc))

    return (LoSource,)


@app.cell
def _(
    AgingServer,
    Environment,
    HiSource,
    LoSource,
    Queue,
    SIM_TIME,
    StaticPriorityServer,
    statistics,
):
    def simulate(arrival_rate_hi, use_aging):
        env = Environment()
        hi_q = Queue(env)
        lo_q = Queue(env)
        sojourn_hi = []
        sojourn_lo = []
        HiSource(env, arrival_rate_hi, hi_q)
        LoSource(env, lo_q)
        if use_aging:
            AgingServer(env, hi_q, lo_q, sojourn_hi, sojourn_lo)
        else:
            StaticPriorityServer(env, hi_q, lo_q, sojourn_hi, sojourn_lo)
        env.run(until=SIM_TIME)
        return sojourn_hi, sojourn_lo

    def mean_or_none(lst):
        return statistics.mean(lst) if lst else None

    def pct_or_none(lst, p):
        if not lst:
            return None
        return sorted(lst)[int(p * len(lst))]

    return mean_or_none, pct_or_none, simulate


@app.cell
def _(
    ARRIVAL_RATE_LO,
    SEED,
    SERVICE_RATE_HI,
    SERVICE_RATE_LO,
    mean_or_none,
    pl,
    random,
    simulate,
):
    def sweep():
        sweep_rows = []
        for rho_hi in [0.10, 0.20, 0.40, 0.60, 0.70, 0.80]:
            rate_hi = rho_hi * SERVICE_RATE_HI
            hi, lo = simulate(rate_hi, use_aging=False)
            rho_total = rho_hi + ARRIVAL_RATE_LO / SERVICE_RATE_LO
            sweep_rows.append({
                "rho_hi": rho_hi,
                "rho_total": rho_total,
                "mean_W_hi": mean_or_none(hi),
                "mean_W_lo": mean_or_none(lo),
            })
        return pl.DataFrame(sweep_rows)

    random.seed(SEED)
    df_sweep = sweep()
    return (df_sweep,)


@app.cell
def _(
    ARRIVAL_RATE_LO,
    SERVICE_RATE_HI,
    SERVICE_RATE_LO,
    mean_or_none,
    pct_or_none,
    pl,
    simulate,
):
    FIXED_RHO_HI = 0.70
    rho_total = FIXED_RHO_HI + ARRIVAL_RATE_LO / SERVICE_RATE_LO
    def compare():
        rate_hi = FIXED_RHO_HI * SERVICE_RATE_HI
        hi_static, lo_static = simulate(rate_hi, use_aging=False)
        hi_aging, lo_aging = simulate(rate_hi, use_aging=True)
    
        compare_rows = [
            {
                "policy": "static", "class": "hi", "n": len(hi_static),
                "mean_W": mean_or_none(hi_static),
                "p95": pct_or_none(hi_static, 0.95),
                "p99": pct_or_none(hi_static, 0.99),
            },
            {
                "policy": "static", "class": "lo", "n": len(lo_static),
                "mean_W": mean_or_none(lo_static),
                "p95": pct_or_none(lo_static, 0.95),
                "p99": pct_or_none(lo_static, 0.99),
            },
            {
                "policy": "aging", "class": "hi", "n": len(hi_aging),
                "mean_W": mean_or_none(hi_aging),
                "p95": pct_or_none(hi_aging, 0.95),
                "p99": pct_or_none(hi_aging, 0.99),
            },
            {
                "policy": "aging", "class": "lo", "n": len(lo_aging),
                "mean_W": mean_or_none(lo_aging),
                "p95": pct_or_none(lo_aging, 0.95),
                "p99": pct_or_none(lo_aging, 0.99),
            },
        ]
        return pl.DataFrame(compare_rows)

    df_compare = compare()
    return (df_compare, FIXED_RHO_HI, rho_total,)


@app.cell(hide_code=True)
def _(AGING_THRESHOLD, ARRIVAL_RATE_LO, SERVICE_RATE_LO, mo):
    mo.md(f"""
    ## Part 1 — Static Priority: Effect of Hi-Priority Load on Lo-Priority Wait

    Lo-priority: arrival rate {ARRIVAL_RATE_LO}, mean service {1 / SERVICE_RATE_LO:.1f},
    ρ_lo = {ARRIVAL_RATE_LO / SERVICE_RATE_LO:.2f}

    Aging threshold: {AGING_THRESHOLD} time units
    """)
    return


@app.cell
def _(df_sweep):
    df_sweep
    return


@app.cell(hide_code=True)
def _(FIXED_RHO_HI, mo, rho_total):
    mo.md(f"""
    ## Part 2 — Static vs. Aging at ρ_hi = {FIXED_RHO_HI:.2f}, ρ_total = {rho_total:.2f}
    """)
    return


@app.cell
def _(df_compare):
    df_compare
    return


@app.cell
def _(alt, df_compare, df_sweep):
    df_plot = df_sweep.unpivot(
        on=["mean_W_hi", "mean_W_lo"],
        index=["rho_hi", "rho_total"],
        variable_name="job_class",
        value_name="mean_W",
    )
    sweep_chart = (
        alt.Chart(df_plot)
        .mark_line(point=True)
        .encode(
            x=alt.X("rho_hi:Q", title="Hi-priority utilization (ρ_hi)"),
            y=alt.Y("mean_W:Q", title="Mean sojourn time (W)"),
            color=alt.Color("job_class:N", title="Job class"),
            tooltip=["rho_hi:Q", "job_class:N", "mean_W:Q"],
        )
        .properties(title="Priority Starvation: Effect of Hi-Priority Load")
    )
    compare_chart = (
        alt.Chart(df_compare)
        .mark_bar()
        .encode(
            x=alt.X("class:N", title="Job class"),
            y=alt.Y("mean_W:Q", title="Mean sojourn time (W)"),
            color=alt.Color("policy:N", title="Policy"),
            xOffset="policy:N",
            tooltip=["policy:N", "class:N", "mean_W:Q", "p99:Q"],
        )
        .properties(title="Static Priority vs. Aging")
    )
    (sweep_chart | compare_chart)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Understanding the Math

    ### Mean wait for two-priority queues

    Let $\lambda_i$, $\mu_i$, and $\rho_i = \lambda_i / \mu_i$ be the arrival rate, service rate, and utilization of class $i \in \{H, L\}$. For a non-preemptive priority queue:

    $$W_H = \frac{R_0}{1 - \rho_H}$$

    $$W_L = \frac{R_0}{(1 - \rho_H)(1 - \rho_H - \rho_L)}$$

    where $R_0 = \tfrac{1}{2}(\lambda_H \overline{s_H^2} + \lambda_L \overline{s_L^2})$ is the mean residual work seen by an arriving customer. The ratio $W_L / W_H = 1/(1 - \rho_H)$ grows without bound as $\rho_H \to 1$.

    ### Utilization of each class

    Let $\lambda_H$ be the arrival rate of high-priority jobs (H) and $\mu_H$ be their service rate. The utilization contributed by H jobs alone is $\rho_H = \lambda_H / \mu_H$ — the fraction of server time that H jobs would consume if they were the only class. Similarly, $\rho_L = \lambda_L / \mu_L$ for low-priority jobs. The total utilization is $\rho = \rho_H + \rho_L$. Requiring $\rho < 1$ means the server has enough capacity for both classes on average.

    ### Why "on average" is not enough

    Even when $\rho < 1$, randomness creates bursts of H arrivals. During a burst, the server is continuously occupied by H jobs, and L jobs must wait in the background. The mean wait for low-priority jobs in a non-preemptive priority queue is:

    $$W_L = \frac{R_0}{(1 - \rho_H)(1 - \rho_H - \rho_L)}$$

    where $R_0$ is the mean residual work in the system when a job arrives. The critical observation is the factor $(1 - \rho_H)$ in the denominator. As $\rho_H \to 1$, this factor approaches zero and $W_L \to \infty$ — even if $\rho_L$ stays small and the total load $\rho$ is comfortably below 1.

    ### The trade-off

    Without aging, $W_L$ can be infinite when $\rho_H$ is large. With aging, $W_L \leq T_{\max} + 1/\mu_L$, but during promotion events the effective $\rho_H$ spikes temporarily, increasing $W_H$. Choosing $T_{\max}$ is a design decision: a small $T_{\max}$ protects L jobs but forces more promotions and penalizes H jobs more often; a large $T_{\max}$ is kinder to H jobs but allows L jobs to wait longer. There is no setting that simultaneously minimizes both — the trade-off is fundamental.
    """)
    return


if __name__ == "__main__":
    app.run()
