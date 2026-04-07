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
    import math
    import random

    import altair as alt
    import polars as pl

    from asimpy import Environment, Process

    return Environment, Process, alt, math, mo, pl, random


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Braess's Paradox

    ## *Adding a Road Makes Traffic Worse*

    A city has two routes from source $S$ to destination $T$:

    - **Top route** $S \to A \to T$: link $SA$ is congestion-dependent; link $AT$ has a fixed travel time.
    - **Bottom route** $S \to B \to T$: link $SB$ has a fixed travel time; link $BT$ is congestion-dependent.

    The network is symmetric. A city planner proposes adding a new shortcut link $A \to B$ with near-zero travel time, creating a third route $S \to A \to B \to T$. To her surprise, adding the shortcut makes everyone's travel time longer at the selfish-routing Nash equilibrium.

    ### Without the shortcut

    Both routes are symmetric. In equilibrium, traffic splits evenly. If $N/2$ drivers use each route and the congested links have delay $\alpha \cdot n$ (where $n$ is the number of cars):

    $$t_{\text{top}} = \frac{N}{2}\alpha + c = t_{\text{bottom}}$$

    ### With the shortcut $A \to B$

    Each driver thinks, "Link $AB$ is free; I can use $SA$, slip across to $B$, then take $BT$ instead of the slow constant link $AT$." All $N$ drivers make this choice. The Nash equilibrium has everyone on $S \to A \to B \to T$:

    $$t_{\text{shortcut}} = N\alpha + \varepsilon + N\alpha = 2N\alpha + \varepsilon$$

    Since $2N\alpha > \frac{N}{2}\alpha + c$ for typical parameters, travel times *increase* after the road is added. This is the paradox: individually rational decisions produce a collectively worse outcome. The ratio of Nash equilibrium cost to the socially optimal cost is called the *[price of anarchy](https://en.wikipedia.org/wiki/Price_of_anarchy)*.

    [Braess's paradox](https://en.wikipedia.org/wiki/Braess%27s_paradox) is not theoretical. Seoul, Stuttgart, and New York all observed traffic *improvements* after closing roads. Conversely, new roads in highly congested networks have sometimes worsened average travel times.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Implementation

    The simulation maintains a shared `LinkCounts` object tracking how many cars are currently on each link. Each `Car` process:

    1. Observes current link counts and computes expected travel time for each available route.
    2. Greedily picks the route with minimum expected time.
    3. Traverses each link in sequence, incrementing the count on entry and decrementing on exit; the delay is fixed at the count observed on entry.

    Two runs are compared: one with only the top and bottom routes, one with the $AB$ shortcut added. The simulation uses a probabilistic [logit](https://en.wikipedia.org/wiki/Logit) choice rule so that convergence to Nash equilibrium is smooth rather than instant.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    n_rounds_slider = mo.ui.slider(
        start=10,
        stop=200,
        step=10,
        value=80,
        label="Number of rounds",
    )

    beta_slider = mo.ui.slider(
        start=0.1,
        stop=2.0,
        step=0.1,
        value=0.5,
        label="Sensitivity (β)",
    )

    seed_input = mo.ui.number(
        value=192,
        step=1,
        label="Random seed",
    )

    run_button = mo.ui.run_button(label="Run simulation")

    mo.vstack([
        n_rounds_slider,
        beta_slider,
        seed_input,
        run_button,
    ])
    return beta_slider, n_rounds_slider, seed_input


@app.cell
def _(beta_slider, n_rounds_slider, seed_input):
    N_ROUNDS = int(n_rounds_slider.value)
    BETA = float(beta_slider.value)
    SEED = int(seed_input.value)
    N_DRIVERS = 4000
    CAPACITY = 100.0
    CONST_DELAY = 45.0
    return BETA, CAPACITY, CONST_DELAY, N_DRIVERS, N_ROUNDS, SEED


@app.cell
def _(CAPACITY, CONST_DELAY):
    def route_times(n_top, n_bot, n_short):
        n_sa = n_top + n_short
        n_bt = n_bot + n_short
        t_top = n_sa / CAPACITY + CONST_DELAY
        t_bot = CONST_DELAY + n_bt / CAPACITY
        t_short = n_sa / CAPACITY + n_bt / CAPACITY
        return t_top, t_bot, t_short

    return (route_times,)


@app.cell
def _(BETA, math):
    def logit_split(times):
        vals = [math.exp(-BETA * t) for t in times]
        total = sum(vals)
        return [v / total for v in vals]

    return (logit_split,)


@app.cell
def _(N_DRIVERS, N_ROUNDS, Process, logit_split, route_times):
    class RoutingGame(Process):
        def init(self, has_shortcut, history):
            self.has_shortcut = has_shortcut
            self.history = history
            self._n_top = N_DRIVERS // 2
            self._n_bot = N_DRIVERS - N_DRIVERS // 2
            self._n_short = 0

        async def run(self):
            for _ in range(N_ROUNDS):
                await self.timeout(1.0)
                t_top, t_bot, t_short = route_times(self._n_top, self._n_bot, self._n_short)
                if self.has_shortcut:
                    probs = logit_split([t_top, t_bot, t_short])
                    self._n_top = round(N_DRIVERS * probs[0])
                    self._n_bot = round(N_DRIVERS * probs[1])
                    self._n_short = N_DRIVERS - self._n_top - self._n_bot
                else:
                    probs = logit_split([t_top, t_bot])
                    self._n_top = round(N_DRIVERS * probs[0])
                    self._n_bot = N_DRIVERS - self._n_top
                    self._n_short = 0
                t_top2, t_bot2, t_short2 = route_times(self._n_top, self._n_bot, self._n_short)
                mean_t = (
                    self._n_top * t_top2 + self._n_bot * t_bot2 + self._n_short * t_short2
                ) / N_DRIVERS
                self.history.append({
                    "round": self.now,
                    "n_top": self._n_top,
                    "n_bot": self._n_bot,
                    "n_short": self._n_short,
                    "t_top": t_top2,
                    "t_bot": t_bot2,
                    "t_short": t_short2,
                    "mean": mean_t,
                })

    return (RoutingGame,)


@app.cell
def _(Environment, RoutingGame):
    def simulate(has_shortcut):
        history = []
        env = Environment()
        RoutingGame(env, has_shortcut, history)
        env.run()
        return history

    return (simulate,)


@app.cell
def _(CAPACITY, CONST_DELAY, N_DRIVERS, SEED, pl, random, simulate):
    random.seed(SEED)
    hist_no = simulate(has_shortcut=False)
    hist_yes = simulate(has_shortcut=True)
    df_no = pl.DataFrame(hist_no)
    df_yes = pl.DataFrame(hist_yes)
    eq_no = hist_no[-1]["mean"]
    eq_yes = hist_yes[-1]["mean"]
    n_half = N_DRIVERS / 2
    t_theory_no = n_half / CAPACITY + CONST_DELAY
    t_theory_yes = N_DRIVERS / CAPACITY + N_DRIVERS / CAPACITY
    return df_no, df_yes, eq_no, eq_yes, t_theory_no, t_theory_yes


@app.cell(hide_code=True)
def _(
    CAPACITY,
    CONST_DELAY,
    N_DRIVERS,
    eq_no,
    eq_yes,
    mo,
    t_theory_no,
    t_theory_yes,
):
    mo.md(f"""
    ## Results

    - Nash equilibrium **without** shortcut: **{eq_no:.2f}**
    - Nash equilibrium **with** shortcut: **{eq_yes:.2f}**
    - Adding the shortcut increased travel time by **{eq_yes - eq_no:.2f}** units
      ({100 * (eq_yes / eq_no - 1):.1f}% worse for every driver)

    Theory without shortcut (50/50 split): {t_theory_no:.2f}

    Theory with shortcut (all on SA→AB→BT): {t_theory_yes:.2f}

    Parameters: {N_DRIVERS} drivers, capacity={CAPACITY:.0f}, constant delay={CONST_DELAY}
    """)
    return


@app.cell
def _(alt, df_no, df_yes, pl):
    df_no_plot = df_no.select(["round", "mean"]).with_columns(
        pl.lit("without shortcut").alias("scenario")
    )
    df_yes_plot = df_yes.select(["round", "mean"]).with_columns(
        pl.lit("with shortcut").alias("scenario")
    )
    df_plot = pl.concat([df_no_plot, df_yes_plot])
    chart = (
        alt.Chart(df_plot)
        .mark_line()
        .encode(
            x=alt.X("round:Q", title="Round"),
            y=alt.Y("mean:Q", title="Mean travel time"),
            color=alt.Color("scenario:N", title="Network"),
            tooltip=["round:Q", "scenario:N", "mean:Q"],
        )
        .properties(title="Braess's Paradox: Convergence to Nash Equilibrium")
    )
    chart
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Understanding the Math

    ### Nash equilibrium

    A Nash equilibrium is a situation where every player has chosen a strategy and no single player can improve their own outcome by switching to a different strategy so long as everyone else stays put. Think of it as a stable fixed point: if you woke up one morning in a Nash equilibrium, you would have no reason to change what you are doing. Crucially, a Nash equilibrium need not be the best possible outcome for everyone collectively.

    ### The paradox, step by step

    Label the number of cars $N$ and suppose the congested links have delay $\alpha \cdot n$ where $n$ is the number of cars currently using that link. Without the shortcut, traffic splits evenly: $N/2$ cars use each route. Each driver's travel time is $(N/2)\alpha + c$, where $c$ is the fixed delay on the non-congested link. Neither route is faster than the other, so no driver wants to switch — that is Nash equilibrium.

    Now add the shortcut $A \to B$ with near-zero travel time $\varepsilon$. A single driver considering a switch reasons: "Link $AB$ is essentially free. If I take $SA$, cross to $B$, and take $BT$, I avoid the fixed cost $c$." If that driver is the only one to switch, it looks cheaper. But every driver makes the same calculation simultaneously. At the new equilibrium, all $N$ drivers pile onto $SA$ and $BT$:

    $$t_{\text{shortcut}} = N\alpha + \varepsilon + N\alpha = 2N\alpha + \varepsilon$$

    Since $2N\alpha > (N/2)\alpha + c$ for typical parameters, everyone is worse off than before the shortcut was built.

    ### The price of anarchy

    The social optimum would split traffic evenly at cost $(N/2)\alpha + c$, but selfish routing delivers $2N\alpha + \varepsilon$. The price of anarchy exceeds 1, meaning individual rationality destroys collective welfare.

    The [Prisoner's Dilemma](https://en.wikipedia.org/wiki/Prisoner's_dilemma) is the best-known example of this tension. Two suspects each choose independently to cooperate or defect. Defecting is a dominant strategy: it is better for you regardless of what the other person does. Yet if both defect, both get a worse outcome than if both had cooperated. Braess's paradox is the same logic scaled to $N$ drivers.

    ### The logit model

    The simulation uses a probabilistic choice rule: the probability a driver picks route $r$ is proportional to $\exp(-\beta \cdot t_r)$, where $t_r$ is the expected travel time on route $r$ and $\beta$ is a sensitivity parameter. When $\beta$ is large, drivers strongly prefer the fastest route and the outcome approaches the pure Nash equilibrium. When $\beta$ is small, drivers choose nearly randomly and the paradox weakens. The parameter $\beta$ captures how responsive real drivers are to time differences.
    """)
    return


if __name__ == "__main__":
    app.run()
