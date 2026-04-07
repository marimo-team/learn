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

    from asimpy import Environment, Event, Process, Queue

    return Environment, Event, Process, Queue, alt, mo, pl, random


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Late Merge

    ## *Courtesy Reduces Throughput*

    A two-lane road narrows to one lane at a construction zone. Drivers face a choice:

    - Early (courtesy) merge: upon seeing the "Lane Ends Ahead" sign, drivers immediately move from the closing lane into the open lane.
    - Late (zipper) merge: drivers use both lanes all the way to the merge point, then alternate — one car from each lane in turn, like a zipper.

    It turns out that late merging produces higher throughput and shorter queues than early merging, even though it feels less polite. Early merging creates a single long queue that wastes the closing lane's capacity. Late merging fully utilises both lanes up to the bottleneck, then processes cars at the same rate with a zipper pattern. This result is not merely theoretical: the Minnesota Department of Transportation, the German ADAC, and the UK Highway Code all recommend late merging in slow-moving traffic precisely because it is provably more efficient.

    The primary benefit of late merging is higher throughput: more cars complete the merge per unit time. Mean sojourn time for individual cars may actually be slightly longer under late merge, because the larger total buffer admits more cars into the system, increasing average queue occupancy. By Little's Law $L = \lambda W$, if $\lambda$ grows faster than $L$ falls, $W$ rises. This is not a disadvantage: it means more drivers successfully pass through rather than being turned away.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Why Early Merging Hurts

    With early merging:

    - All $N$ cars queue in one lane of capacity $K$.
    - When the single queue is full ($K$ cars), arriving cars are turned away (blocking), reducing throughput.
    - The merge point processes at rate $\mu$ regardless, but there are fewer cars available to process (the second lane is empty and wasted).

    With late merging:

    - Cars distribute across two lanes, each of capacity $K$ (total $2K$).
    - The merge point receives supply from both lanes, reducing starvation.
    - Blocking occurs only when *both* lanes are simultaneously full, a rarer event.

    The key metric is the *blocking probability*: the fraction of arriving cars turned away because the pre-merge buffer is full. Let $\rho = \lambda/\mu$ be the utilisation of the merge bottleneck. For a finite-buffer M/M/1/K queue the blocking probability is:

    $$P_{\text{block}} = \frac{(1-\rho)\rho^K}{1 - \rho^{K+1}}$$

    Early merge has buffer $K$; late merge effectively has buffer $2K$ (spread across two lanes). Since $P_{\text{block}}$ decreases exponentially in $K$, doubling the available buffer dramatically reduces blocking.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Implementation

    - **Early merge**: one `Queue(max_capacity=LANE_CAPACITY)` feeds a single `MergeServer`. Arrivals that find the lane full are counted as blocked and turned away.
    - **Late merge**: two `Queue(max_capacity=LANE_CAPACITY)` instances feed a zipper `MergeServer` that alternates between lanes. Arrivals pick the shorter lane; a car is blocked only if its chosen lane is full.

    Both systems have the same total arrival rate $\lambda$ and merge service rate $\mu$.
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

    lane_capacity_slider = mo.ui.slider(
        start=2,
        stop=30,
        step=1,
        value=10,
        label="Lane capacity (K)",
    )

    seed_input = mo.ui.number(
        value=192,
        step=1,
        label="Random seed",
    )

    run_button = mo.ui.run_button(label="Run simulation")

    mo.vstack([
        sim_time_slider,
        lane_capacity_slider,
        seed_input,
        run_button,
    ])
    return lane_capacity_slider, seed_input, sim_time_slider


@app.cell
def _(lane_capacity_slider, seed_input, sim_time_slider):
    SIM_TIME = int(sim_time_slider.value)
    LANE_CAPACITY = int(lane_capacity_slider.value)
    SEED = int(seed_input.value)
    ARRIVAL_RATE = 1.85
    MERGE_RATE = 2.0
    RHO = ARRIVAL_RATE / MERGE_RATE
    return ARRIVAL_RATE, LANE_CAPACITY, MERGE_RATE, RHO, SEED, SIM_TIME


@app.cell
def _(Event, Process):
    class EarlyMergeCar(Process):
        def init(self, lane, sojourn_times, blocked):
            self.lane = lane
            self.sojourn_times = sojourn_times
            self.blocked = blocked

        async def run(self):
            arrival = self.now
            if self.lane.is_full():
                self.blocked.append(1)
                return
            done = Event(self._env)
            await self.lane.put((arrival, done))
            await done
            self.sojourn_times.append(self.now - arrival)

    return (EarlyMergeCar,)


@app.cell
def _(Event, Process):
    class LateMergeCar(Process):
        def init(self, lane1, lane2, sojourn_times, blocked):
            self.lane1 = lane1
            self.lane2 = lane2
            self.sojourn_times = sojourn_times
            self.blocked = blocked

        async def run(self):
            arrival = self.now
            target = (
                self.lane1
                if len(self.lane1._items) <= len(self.lane2._items)
                else self.lane2
            )
            if target.is_full():
                self.blocked.append(1)
                return
            done = Event(self._env)
            await target.put((arrival, done))
            await done
            self.sojourn_times.append(self.now - arrival)

    return (LateMergeCar,)


@app.cell
def _(MERGE_RATE, Process, random):
    class MergeServer(Process):
        def init(self, lanes, zipper):
            self.lanes = lanes
            self.zipper = zipper
            self._turn = 0

        async def run(self):
            while True:
                if self.zipper:
                    served = False
                    for _ in range(len(self.lanes)):
                        idx = self._turn % len(self.lanes)
                        self._turn += 1
                        if not self.lanes[idx].is_empty():
                            _, arrival, done = (self.now,) + (await self.lanes[idx].get())
                            await self.timeout(random.expovariate(MERGE_RATE))
                            done.succeed()
                            served = True
                            break
                    if not served:
                        await self.timeout(0.05)
                else:
                    _, arrival, done = (self.now,) + (await self.lanes[0].get())
                    await self.timeout(random.expovariate(MERGE_RATE))
                    done.succeed()


    return (MergeServer,)


@app.cell
def _(ARRIVAL_RATE, EarlyMergeCar, LateMergeCar, Process, random):
    class ArrivalStream(Process):
        def init(self, lanes, sojourn_times, blocked, zipper):
            self.lanes = lanes
            self.sojourn_times = sojourn_times
            self.blocked = blocked
            self.zipper = zipper

        async def run(self):
            while True:
                await self.timeout(random.expovariate(ARRIVAL_RATE))
                if self.zipper:
                    LateMergeCar(self._env, self.lanes[0], self.lanes[1], self.sojourn_times, self.blocked)
                else:
                    EarlyMergeCar(self._env, self.lanes[0], self.sojourn_times, self.blocked)

    return (ArrivalStream,)


@app.cell
def _(ArrivalStream, Environment, MergeServer, Queue, SEED, SIM_TIME, random):
    def run_scenario(zipper, k):
        random.seed(SEED)
        env = Environment()
        sojourn_times = []
        blocked = []

        if zipper:
            lanes = [Queue(env, max_capacity=k), Queue(env, max_capacity=k)]
        else:
            lanes = [Queue(env, max_capacity=k)]
        ArrivalStream(env, lanes, sojourn_times, blocked, zipper)
        MergeServer(env, lanes, zipper)
        env.run(until=SIM_TIME)
        total = len(sojourn_times) + len(blocked)
        blocked_pct = 100.0 * len(blocked) / total if total else 0.0
        throughput = len(sojourn_times) / SIM_TIME
        mean_sojourn = sum(sojourn_times) / len(sojourn_times) if sojourn_times else 0.0
        return {
            "throughput": throughput,
            "blocked_pct": blocked_pct,
            "mean_sojourn": mean_sojourn,
            "total_buffer": k * (2 if zipper else 1),
        }

    return (run_scenario,)


@app.cell(hide_code=True)
def _(ARRIVAL_RATE, MERGE_RATE, RHO, mo):
    mo.md(f"""
    ## Main Results

    Arrival rate: {ARRIVAL_RATE}/unit, merge service rate: {MERGE_RATE}/unit,
    utilisation ρ = {RHO:.3f}
    """)
    return


@app.cell
def _(LANE_CAPACITY, pl, run_scenario):
    early = run_scenario(zipper=False, k=LANE_CAPACITY)
    late = run_scenario(zipper=True, k=LANE_CAPACITY)
    df_main = pl.DataFrame([
        {"strategy": "early", **early},
        {"strategy": "late", **late},
    ])
    df_main
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Effect of Buffer Size on Blocking Rate
    """)
    return


@app.cell
def _(pl, run_scenario):
    def sweep():
        sweep_rows = []
        for k in [5, 10, 15, 20, 30]:
            ep = run_scenario(zipper=False, k=k)["blocked_pct"]
            lp = run_scenario(zipper=True, k=k)["blocked_pct"]
            sweep_rows.append({"buffer_k": k, "early_blocked_pct": ep, "late_blocked_pct": lp})
        return pl.DataFrame(sweep_rows)

    df_sweep = sweep()
    df_sweep
    return (df_sweep,)


@app.cell
def _(alt, df_sweep):
    df_plot = df_sweep.unpivot(
        on=["early_blocked_pct", "late_blocked_pct"],
        index="buffer_k",
        variable_name="strategy",
        value_name="blocked_pct",
    )
    chart = (
        alt.Chart(df_plot)
        .mark_line(point=True)
        .encode(
            x=alt.X("buffer_k:Q", title="Buffer size per lane (K)"),
            y=alt.Y("blocked_pct:Q", title="Blocked cars (%)"),
            color=alt.Color("strategy:N", title="Merge strategy"),
            tooltip=["buffer_k:Q", "strategy:N", "blocked_pct:Q"],
        )
        .properties(title="Late Merge: Blocked Cars vs. Buffer Size")
    )
    chart
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Understanding the Math

    ### The finite-buffer formula

    For a queue with random arrivals, exponential service, a single server, and a buffer that holds at most $K$ cars (the M/M/1/K model), the blocking probability is:

    $$P_{\text{block}} = \frac{(1-\rho)\,\rho^K}{1 - \rho^{K+1}}$$

    As usual, $\rho = \lambda/\mu$ is the utilization. Notice that the numerator contains $\rho^K$. Because $\rho < 1$, increasing $K$ by 1 multiplies the numerator by $\rho < 1$, shrinking $P_{\text{block}}$ faster than linearly. Each extra slot in the buffer is more valuable than a simple linear reduction would suggest.

    ### Early vs. late merge in terms of $K$

    Early merging creates a single queue with buffer $K$: one lane's worth of space. Late merging uses both lanes up to the merge point, creating an effective buffer of $2K$ cars total. Plugging $2K$ into the formula instead of $K$ replaces $\rho^K$ with $\rho^{2K} = (\rho^K)^2$. Since $\rho^K < 1$, squaring it makes it much smaller. This is the why doubling the buffer dramatically reduces blocking.

    ### Intuition about two lanes

    Here is another way to see it. Under late merge, both lanes must be simultaneously full for a car to be blocked. Suppose each individual lane is full with probability $p$. If the two lanes are roughly independent, the probability both are full at once is approximately $p^2$. For example, if $p = 0.3$, then $p^2 = 0.09$ — blocking drops from 30% to 9%. Two lanes are dramatically more forgiving than one.

    ### Connection to throughput

    Throughput is the rate at which cars successfully pass through the merge: $\text{throughput} = \lambda \cdot (1 - P_{\text{block}})$. Every blocked car is a car that does not get through. Reducing $P_{\text{block}}$ by doubling $K$ therefore raises throughput nearly proportionally. Late merge does not speed up the bottleneck (the merge point still processes cars at rate $\mu$) but it ensures the bottleneck is never starved of cars to process, maximizing the number of drivers who make it through.

    ### The broader lesson

    The key insight is that the *structure* of a waiting space matters, not just its total size. Two separate lanes of capacity $K$ each are far better than one lane of capacity $2K$ because blocking requires both lanes to fill simultaneously. This logic generalises widely: in computer networks, having multiple independent paths reduces the chance a single congested link stalls all traffic; in hospitals, pooling patients across several triage nurses reduces the chance one idle nurse sits beside an overwhelmed colleague. Wherever there is a finite buffer feeding a shared bottleneck, the late-merge principle applies: spread the waiting space across parallel channels and blocking probability falls dramatically.
    """)
    return


if __name__ == "__main__":
    app.run()
