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

    from asimpy import Environment, Process, Queue, PriorityQueue

    return Environment, Process, Queue, PriorityQueue, alt, mo, pl, random, statistics


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # The Convoy Effect

    ## *One Slow Job Ruins Everyone's Day*

    A single server processes jobs that arrive randomly (Poisson process). Most jobs are quick (exponential service with small mean), but a rare few are very slow (exponential service with large mean). This *hyperexponential* service distribution has high variance. This tutorial compares the performance of two scheduling disciplines in this situation:

    - FIFO (First In, First Out): jobs are served in the order they arrive.
    - SJF (Shortest Job First): the server always picks the shortest queued job next.

    The surprising result is that SJF dramatically outperforms FIFO: not just for the small jobs that directly benefit from skipping ahead, but also for mean sojourn time across *all* jobs. The improvement is most visible at the tail (95th and 99th percentiles) because FIFO creates a *convoy effect*: one long job blocks many short jobs behind it, inflating everyone's wait.

    ### The Convoy Metaphor

    Picture a one-lane road with one slow truck and many fast cars. Every car behind the truck must drive at truck speed; no overtaking allowed. The truck is the long job; the cars are the short jobs stuck behind it in FIFO order. SJF is like a passing lane: fast cars jump ahead of the truck and reach their destination much sooner. The truck itself arrives at the same time either way, but the total delay experienced by all vehicles plummets.

    ### Why FIFO Hurts with High Variance

    In FIFO, the server's current job is chosen at arrival time, not at decision time. When a slow job begins service, every subsequent arrival must join the queue and wait. The expected excess work in service (the remaining time of the current job, seen by an arriving customer) under FIFO is:

    $$W_{\text{FIFO}} = \frac{\lambda \overline{s^2}}{2(1-\rho)} + \frac{1}{\mu}$$

    where $\overline{s^2}$ is the second moment of service time. High variance inflates $\overline{s^2}$ without changing $\rho$, directly worsening wait time.

    ### SJF Minimises Mean Sojourn Time

    For a single server with non-preemptive SJF and any service-time distribution, the mean sojourn time is given by the formula below (which is discussed in "Understanding the Math" at the end of this lesson):

    $$W_{\text{SJF}} = \frac{1}{\mu} + \frac{\lambda \overline{s^2}}{2(1-\rho)}$$

    SJF achieves this minimum because short jobs that would otherwise be blocked by a long job are promoted ahead, reducing the total waiting work in the system.

    ## Practical Relevance

    Operating system CPU schedulers use time-quanta and priority aging to approximate SJF without knowing job sizes in advance. Database query planners estimate query cost and reorder execution to minimize blocking. The phenomenon reappears as *head-of-line blocking* in HTTP/1.1 (one slow response stalls a connection), motivating HTTP/2 multiplexing and HTTP/3's QUIC stream independence.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Implementation

    Jobs are placed in a `PriorityQueue` for SJF (tupled as `(service_time, job_id)` so shorter jobs sort earlier) or a plain `Queue()` for FIFO (tupled as `(job_id, service_time)` to preserve arrival order). The same hyperexponential service-time generator (90% short, 10% long) is used in both runs.
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
        start=0.1,
        stop=1.5,
        step=0.05,
        value=0.7,
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
    SHORT_RATE = 4.0
    LONG_RATE = 0.2
    LONG_PROB = 0.10
    return ARRIVAL_RATE, LONG_PROB, LONG_RATE, SEED, SHORT_RATE, SIM_TIME


@app.cell
def _(LONG_PROB, LONG_RATE, SHORT_RATE, random):
    def service_time():
        if random.random() < LONG_PROB:
            return random.expovariate(LONG_RATE)
        return random.expovariate(SHORT_RATE)

    return (service_time,)


@app.cell
def _(ARRIVAL_RATE, Process, random, service_time):
    class JobSource(Process):
        def init(self, job_queue, arrivals, sjf):
            self.job_queue = job_queue
            self.arrivals = arrivals
            self.sjf = sjf
            self._jid = 0

        async def run(self):
            while True:
                await self.timeout(random.expovariate(ARRIVAL_RATE))
                jid = self._jid
                self._jid += 1
                svc = service_time()
                self.arrivals[jid] = (self.now, svc)
                if self.sjf:
                    await self.job_queue.put((svc, jid))
                else:
                    await self.job_queue.put((jid, svc))

    return (JobSource,)


@app.cell
def _(Process):
    class Server(Process):
        def init(self, job_queue, arrivals, sojourn_times, sjf):
            self.job_queue = job_queue
            self.arrivals = arrivals
            self.sojourn_times = sojourn_times
            self.sjf = sjf

        async def run(self):
            while True:
                item = await self.job_queue.get()
                if self.sjf:
                    svc, jid = item
                else:
                    jid, svc = item
                await self.timeout(svc)
                arrival_time, _ = self.arrivals[jid]
                self.sojourn_times.append(self.now - arrival_time)

    return (Server,)


@app.cell
def _(Environment, JobSource, Queue, PriorityQueue, SIM_TIME, Server, statistics):
    def simulate(sjf):
        arrivals = {}
        sojourn_times = []
        env = Environment()
        q = PriorityQueue(env) if sjf else Queue(env)
        JobSource(env, q, arrivals, sjf)
        Server(env, q, arrivals, sojourn_times, sjf)
        env.run(until=SIM_TIME)
        return {
            "mean": statistics.mean(sojourn_times),
            "median": statistics.median(sojourn_times),
            "p95": sorted(sojourn_times)[int(0.95 * len(sojourn_times))],
            "p99": sorted(sojourn_times)[int(0.99 * len(sojourn_times))],
            "n": len(sojourn_times),
        }

    return (simulate,)


@app.cell
def _(LONG_PROB, LONG_RATE, SEED, SHORT_RATE, pl, random, simulate):
    def run_scenarios():
        fifo = simulate(sjf=False)
        sjf_res = simulate(sjf=True)
        rows = [
            {
                "metric": m,
                "fifo": fifo[m],
                "sjf": sjf_res[m],
                "improvement": fifo[m] / sjf_res[m],
            }
            for m in ("mean", "median", "p95", "p99")
        ]
        return pl.DataFrame(rows)

    random.seed(SEED)
    df = run_scenarios()
    mean_svc = (1 - LONG_PROB) / SHORT_RATE + LONG_PROB / LONG_RATE
    return df, mean_svc


@app.cell(hide_code=True)
def _(ARRIVAL_RATE, LONG_PROB, LONG_RATE, SHORT_RATE, mean_svc, mo):
    mo.md(f"""
    ## Summary Statistics

    Arrival rate: {ARRIVAL_RATE}, estimated mean service: {mean_svc:.3f}

    Short jobs: {100 * (1 - LONG_PROB):.0f}% (mean {1 / SHORT_RATE:.2f}),
    Long jobs: {100 * LONG_PROB:.0f}% (mean {1 / LONG_RATE:.1f})

    > **Note:** SJF is optimal for mean sojourn time but requires knowing job sizes in advance.
    """)
    return


@app.cell
def _(df):
    df
    return


@app.cell
def _(alt, df, pl):
    df_plot = df.filter(pl.col("metric") != "n").unpivot(
        on=["fifo", "sjf"],
        index="metric",
        variable_name="policy",
        value_name="sojourn_time",
    )
    chart = (
        alt.Chart(df_plot)
        .mark_bar()
        .encode(
            x=alt.X("metric:N", title="Metric"),
            y=alt.Y("sojourn_time:Q", title="Sojourn time"),
            color=alt.Color("policy:N", title="Policy"),
            xOffset="policy:N",
            tooltip=["metric:N", "policy:N", "sojourn_time:Q"],
        )
        .properties(title="Convoy Effect: FIFO vs. Shortest Job First")
    )
    chart
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Understanding the Math

    ### The second moment

    For a random variable $S$ representing service time, the second moment is $E[S^2]$. Recall from your statistics course that variance is $\text{Var}(S) = E[S^2] - (E[S])^2$, which rearranges to:

    $$E[S^2] = \text{Var}(S) + (E[S])^2$$

    This means high variance inflates $E[S^2]$ even if the mean $E[S]$ stays fixed. Doubling the spread of service times can quadruple $E[S^2]$, even with the same average service time.

    ### Why variance of service time hurts

    Imagine a FIFO server handling jobs that are either 0.1 minutes or 10 minutes long, with 90% being short and 10% being long. The mean service time is $0.9 \times 0.1 + 0.1 \times 10 = 1.09$ minutes, so utilization $\rho = \lambda / \mu$ might be modest. But when a 10-minute job starts, every job arriving during those 10 minutes must join the queue and wait. The longer $E[S^2]$, the more average work sits ahead of each arriving job.

    ### The Pollaczek–Khinchine formula

    The mean time a job spends waiting (not counting its own service time) in a FIFO single-server queue is:

    $$W_q = \frac{\lambda \cdot E[S^2]}{2(1 - \rho)}$$

    Here $\lambda$ is the arrival rate, $E[S^2]$ is the second moment of service time, and $\rho = \lambda \cdot E[S]$ is the server utilization. Both $\lambda$ and $E[S^2]$ appear in the numerator, so more variance means more waiting even at the same $\rho$. The $(1-\rho)$ denominator is the familiar blow-up term from M/M/1.
    """)
    return


if __name__ == "__main__":
    app.run()
