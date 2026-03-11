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

    return alt, mo, pl, random, statistics


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Rush Hour Displacement

    ## *If Everyone Avoids the Rush, It Shifts*

    $N$ commuters all want to leave for work at the same preferred time. The road has a fixed capacity: up to $C$ commuters per time slot travel quickly, but when more than $C$ try to leave in the same slot, everyone in that slot experiences extra delay proportional to the overload.

    Each day, commuters observe yesterday's travel times and shift their departure by one slot toward a less congested option with some probability. Much to their disappointment, the rush hour never disappears. Instead it:

    1. flattens slightly (spreading across more slots), but
    2. shifts its peak position over successive days, and
    3. reaches a new quasi-equilibrium that may be no less congested than the original, just at a different time.

    The intuition is that any slot that becomes less congested immediately attracts new commuters from adjacent overloaded slots, refilling it. Individual optimization is self-defeating in aggregate.

    The simulation in this tutorial shows emergent dynamics:

    - The arrival distribution begins concentrated at the preferred slot.
    - Commuters shift away from congested slots, spreading the peak.
    - The spreading creates new local peaks at adjacent slots, which then attract their own shifters.
    - Over many days the distribution oscillates or drifts without converging to zero congestion.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The Vickrey Bottleneck Model

    The classic model (Vickrey 1969) treats the road as a bottleneck with flow rate $s$ vehicles per unit time. At equilibrium, every commuter faces the same *generalized cost*:

    $$c = \alpha \cdot d + \beta \cdot \max(0,\, t^* - t_{\text{arr}}) + \gamma \cdot \max(0,\, t_{\text{arr}} - t^*)$$

    where $d$ is queuing delay, $t^*$ is the desired arrival time, and $\beta, \gamma$ are schedule-delay costs for early and late arrival respectively. Vickrey showed that at [Nash equilibrium](https://en.wikipedia.org/wiki/Nash_equilibrium) a departure queue forms with length that rises and then falls as commuters spread across time to equalize cost, but total system delay is unchanged.

    This model underlies modern road-pricing schemes: a time-varying toll that exactly offsets the schedule-delay cost eliminates queuing entirely while preserving the total commuting burden. In essence, the toll revenue replaces the wasted queuing time.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Implementation

    Rather than one continuous DES run, the simulation iterates over days. Each day's commute is computed analytically: the travel time for slot $s$ is $\delta_{\text{base}}$ when occupancy $\leq C$, and increases linearly with overflow otherwise. Commuters update their departure slot between days using a simple best-response rule with noise (they shift to a better neighbour with probability $p$).
    """)
    return


@app.cell
def _(mo):
    n_commuters_slider = mo.ui.slider(
        start=50,
        stop=500,
        step=50,
        value=200,
        label="Number of commuters",
    )

    n_days_slider = mo.ui.slider(
        start=10,
        stop=100,
        step=5,
        value=40,
        label="Number of days",
    )

    shift_prob_slider = mo.ui.slider(
        start=0.05,
        stop=0.8,
        step=0.05,
        value=0.3,
        label="Shift probability",
    )

    seed_input = mo.ui.number(
        value=192,
        step=1,
        label="Random seed",
    )

    run_button = mo.ui.button(label="Run simulation")

    mo.vstack([
        n_commuters_slider,
        n_days_slider,
        shift_prob_slider,
        seed_input,
        run_button,
    ])
    return n_commuters_slider, n_days_slider, seed_input, shift_prob_slider


@app.cell
def _(n_commuters_slider, n_days_slider, seed_input, shift_prob_slider):
    N_COMMUTERS = int(n_commuters_slider.value)
    N_DAYS = int(n_days_slider.value)
    SHIFT_PROB = float(shift_prob_slider.value)
    SEED = int(seed_input.value)
    N_SLOTS = 30
    PREFERRED_SLOT = 15
    ROAD_CAPACITY = 20
    OVERLOAD_DELAY = 3.0
    BASE_DELAY = 1.0
    return (
        BASE_DELAY,
        N_COMMUTERS,
        N_DAYS,
        N_SLOTS,
        OVERLOAD_DELAY,
        PREFERRED_SLOT,
        ROAD_CAPACITY,
        SEED,
        SHIFT_PROB,
    )


@app.cell
def _(BASE_DELAY, OVERLOAD_DELAY, ROAD_CAPACITY):
    def simulate_day(departure_slots):
        counts = {}
        for s in departure_slots:
            counts[s] = counts.get(s, 0) + 1
        travel_time = {}
        for slot, count in counts.items():
            if count <= ROAD_CAPACITY:
                travel_time[slot] = BASE_DELAY
            else:
                overflow = count - ROAD_CAPACITY
                travel_time[slot] = BASE_DELAY + OVERLOAD_DELAY * overflow / ROAD_CAPACITY
        return travel_time

    return (simulate_day,)


@app.cell
def _(BASE_DELAY, N_SLOTS, SHIFT_PROB, random):
    def update_slots(departure_slots, travel_times):
        new_slots = departure_slots[:]
        for i, s in enumerate(departure_slots):
            my_delay = travel_times.get(s, BASE_DELAY)
            candidates = []
            if s > 0:
                candidates.append(s - 1)
            if s < N_SLOTS - 1:
                candidates.append(s + 1)
            better = [c for c in candidates if travel_times.get(c, BASE_DELAY) < my_delay]
            if better and random.random() < SHIFT_PROB:
                new_slots[i] = random.choice(better)
        return new_slots

    return (update_slots,)


@app.cell
def _(N_SLOTS):
    def slot_distribution(slots):
        counts = [0] * N_SLOTS
        for s in slots:
            counts[s] += 1
        return counts

    return (slot_distribution,)


@app.cell
def _(
    BASE_DELAY,
    N_COMMUTERS,
    N_DAYS,
    N_SLOTS,
    PREFERRED_SLOT,
    SEED,
    pl,
    random,
    simulate_day,
    slot_distribution,
    statistics,
    update_slots,
):
    def simulate():
        departure_slots = [
            max(0, min(N_SLOTS - 1, PREFERRED_SLOT + round(random.gauss(0, 2))))
            for _ in range(N_COMMUTERS)
        ]
    
        rows = []
        for day in range(N_DAYS):
            travel_times = simulate_day(departure_slots)
            mean_delay = statistics.mean(
                travel_times.get(s, BASE_DELAY) for s in departure_slots
            )
            dist = slot_distribution(departure_slots)
            max_count = max(dist)
            rows.append({"day": day + 1, "mean_delay": mean_delay, "max_slot_count": max_count})
            departure_slots = update_slots(departure_slots, travel_times)

        return departure_slots, rows

    random.seed(SEED)
    departure_slots, rows = simulate()
    df = pl.DataFrame(rows)
    final_dist = slot_distribution(departure_slots)
    peak_slot = max(range(N_SLOTS), key=lambda s: final_dist[s])
    return df, peak_slot


@app.cell(hide_code=True)
def _(N_COMMUTERS, N_SLOTS, PREFERRED_SLOT, ROAD_CAPACITY, mo, peak_slot):
    mo.md(f"""
    ## Results

    {N_COMMUTERS} commuters, {N_SLOTS} slots, road capacity {ROAD_CAPACITY}/slot

    - Initial peak slot: {PREFERRED_SLOT}
    - Final peak slot: {peak_slot} ({'same' if peak_slot == PREFERRED_SLOT else 'shifted'})

    > **Observation:** the rush-hour peak flattens and shifts but does not disappear.
    """)
    return


@app.cell
def _(df):
    df
    return


@app.cell
def _(alt, df):
    chart = (
        alt.Chart(df)
        .mark_line()
        .encode(
            x=alt.X("day:Q", title="Day"),
            y=alt.Y("mean_delay:Q", title="Mean travel delay"),
            tooltip=["day:Q", "mean_delay:Q", "max_slot_count:Q"],
        )
        .properties(title="Rush Hour Displacement: Mean Delay Over Time")
    )
    chart
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Understanding the Math

    ### What is a congestion game?

    Each commuter (the "player") independently chooses a departure time slot. The delay experienced in any given slot depends on how many other commuters choose the same slot: if the slot is over capacity $C$, delay grows with the number of extra commuters. No central authority coordinates choices. This structure, where each player's cost depends on the collective choices of all players, is called a *congestion game*.

    ### Nash equilibrium in this context

    A Nash equilibrium is a distribution of departure times such that no individual commuter can reduce their own delay by unilaterally switching to a different slot. At equilibrium, every occupied slot has the same congestion-adjusted cost. If slot 15 were cheaper than slot 14, commuters from slot 14 would shift to slot 15 until the costs equalized. The equilibrium is therefore defined by: all slots with commuters in them have equal cost, and all empty slots have cost no lower than the occupied ones.

    ### Why Nash equilibrium is not the social optimum

    The social optimum minimizes total delay summed over all commuters. The Nash equilibrium minimizes each person's individual delay given everyone else's choices. These are generally different objectives. At Nash equilibrium, a commuter choosing a crowded slot ignores the extra delay they impose on every other commuter already in that slot. They feel only their own delay; the cost they impose on others is a negative externality that they do not internalize.

    ### Why the peak shifts but does not vanish

    Suppose slot 15 is heavily congested. Some commuters shift to slot 14, relieving slot 15. But now slot 14 is more congested, so its commuters shift to slot 13. The congestion wave ripples outward in both directions. Meanwhile, commuters who shifted away from slot 15 now observe it as less congested and some drift back. The system never reaches zero congestion: it perpetually redistributes congestion across nearby slots in a slow drift. The Nash equilibrium exists in theory, but the day-by-day best-response dynamics cycle around it rather than converging to it, particularly when commuters respond noisily to yesterday's conditions.
    """)
    return


if __name__ == "__main__":
    app.run()
