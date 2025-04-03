# Learn Functional Programming

_🚧 This collection is a
[work in progress](https://github.com/marimo-team/learn/issues/51)._

This series of marimo notebooks introduces the powerful paradigm of functional
programming through Python. Taking inspiration from Haskell and Category Theory,
we'll build a strong foundation in FP concepts that can transform how you
approach software development.

## What You'll Learn

**Using only Python's standard library**, we'll construct functional programming
concepts from first principles.

Topics include:

-   Recursion and higher-order functions
-   Category theory fundamentals
-   Functors, applicatives, and monads
-   Composable abstractions for robust code

## Timeline & Collaboration

I'm currently studying functional programming and Haskell, estimating about 2
months or even longer to complete this series. The structure may evolve as the
project develops.

If you're interested in collaborating or have questions, please reach out to me
on Discord (@eugene.hs).

**Running notebooks.** To run a notebook locally, use

```bash
uvx marimo edit <URL>
```

For example, run the `Functor` tutorial with

```bash
uvx marimo edit https://github.com/marimo-team/learn/blob/main/Functional_programming/05_functors.py
```

You can also open notebooks in our online playground by appending `marimo.app/`
to a notebook's URL:
[marimo.app/github.com/marimo-team/learn/blob/main/functional_programming/05_functors.py](https://marimo.app/https://github.com/marimo-team/learn/blob/main/functional_programming/05_functors.py).

# Description of notebooks

Check [here](https://github.com/marimo-team/learn/issues/51) for current series
structure.

| Notebook | Title | Description | Key Concepts | Prerequisites |
|----------|-------|-------------|--------------|---------------|
| [05. Functors](https://github.com/marimo-team/learn/blob/main/Functional_programming/05_functors.py) | Category and Functors | Learn why `len` is a _Functor_ from `list concatenation` to `integer addition`, how to _lift_ an ordinary function into a _computation context_, and how to write an _adapter_ between two categories. | Categories, Functors, Function lifting, Context mapping | Basic Python, Functions |
| [06. Applicatives](https://github.com/marimo-team/learn/blob/main/Functional_programming/06_applicatives.py) | Applicative programming with effects | Learn how to apply functions within a context, combining multiple effects in a pure way. Learn about the `pure` and `apply` operations that make applicatives powerful for handling multiple computations. | Applicative Functors, Pure, Apply, Effectful programming | Functors |

**Authors.**

Thanks to all our notebook authors!

-   [métaboulie](https://github.com/metaboulie)
