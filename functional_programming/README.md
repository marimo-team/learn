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

| Notebook                                                                                                          | Description                                                                                                                                                                                            | References                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| ----------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [05. Category and Functors](https://github.com/marimo-team/learn/blob/main/Functional_programming/05_functors.py) | Learn why `len` is a _Functor_ from `list concatenation` to `integer addition`, how to _lift_ an ordinary function into a _computation context_, and how to write an _adapter_ between two categories. | - [The Trivial Monad](http://blog.sigfpe.com/2007/04/trivial-monad.html) <br> - [Haskellwiki. Category Theory](https://en.wikibooks.org/wiki/Haskell/Category_theory) <br> - [Haskellforall. The Category Design Pattern](https://www.haskellforall.com/2012/08/the-category-design-pattern.html) <br> - [Haskellforall. The Functor Design Pattern](https://www.haskellforall.com/2012/09/the-functor-design-pattern.html) <br> - [Haskellwiki. Functor](https://wiki.haskell.org/index.php?title=Functor) <br> - [Haskellwiki. Typeclassopedia#Functor](https://wiki.haskell.org/index.php?title=Typeclassopedia#Functor) <br> - [Haskellwiki. Typeclassopedia#Category](https://wiki.haskell.org/index.php?title=Typeclassopedia#Category) |

**Authors.**

Thanks to all our notebook authors!

-   [métaboulie](https://github.com/metaboulie)
