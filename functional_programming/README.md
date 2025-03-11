# Functional Programming in Python with Marimo

_🚧 This collection is a [work in progress](https://github.com/marimo-team/learn/issues/51)._

This series of marimo notebooks introduces the powerful paradigm of functional programming through Python. Taking inspiration from Haskell and Category Theory, we'll build a strong foundation in FP concepts that can transform how you approach software development.

## What You'll Learn

Using only Python's standard library, we'll construct functional programming concepts from first principles.

Topics include:
- Recursion and higher-order functions
- Category theory fundamentals
- Functors, applicatives, and monads
- Composable abstractions for robust code

## Timeline & Collaboration

I'm currently studying functional programming and Haskell, estimating about 2 months to complete this series. The structure may evolve as the project develops.

If you're interested in collaborating or have questions, please reach out to me on Discord (@eugene.hs). I welcome contributors who share an interest in bringing functional programming concepts to the Python ecosystem.

**Running notebooks.** To run a notebook locally, use

```bash
uvx marimo edit <URL>
```

For example, run the `Functor` tutorial with

```bash
uvx marimo edit https://github.com/marimo-team/learn/blob/main/Functional_programming/05_functors.py
```

You can also open notebooks in our online playground by appending `marimo.app/`
to a notebook's URL: [marimo.app/github.com/marimo-team/learn/blob/main/functional_programming/05_functors.py](https://marimo.app/https://github.com/marimo-team/learn/blob/main/functional_programming/05_functors.py).

## Current series structure


| Notebook | Description | Status | Author |
|----------|-------------|--------|--------|
| Functional Programming Fundamentals | Core FP principles in Python, comparison with imperative programming, and Haskell-inspired thinking patterns | 🚧 | |
| Higher-Order Functions and Currying | Functions as first-class values, composition patterns, and implementing Haskell-style currying in Python | 🚧 | |
| Python's Functional Toolkit: functools, itertools and operator | Leveraging Python's built-in functional programming utilities, advanced iterator operations, and function transformations | 🚧 | |
| Recursion and Tail Recursion | Recursive problem solving, implementing tail-call optimization in Python, and trampoline techniques to avoid stack overflow | 🚧 | |
| Category Theory and Functors | Introduction to categories, morphisms, functor patterns, and implementing the functor interface and practical use cases | 🚧 | |
| Applicatives and Effectful Programming | Combining independent computations with effects, implementing the applicative interface and practical use cases | 🚧 | |
| Kleisli Category and Monads | Understanding monadic computation, composing impure functions, and implementing basic monads | 🚧 | |
| Monad Fail, Transformers and Fix | Error handling with MonadFail, combining monads with transformers, and handling recursive structures | 🚧 | |
| Monadic Parsing | Building a parser combinator library, implementing recursive descent parsers, and practical text processing | 🚧 | |
| Semigroups and Monoids | Composable operations, algebraic structures, laws, and practical applications for data aggregation | 🚧 | |
| Foldables and Traversables | Abstract folding beyond lists, traversing with effects, and implementing these interfaces for custom data types | 🚧 | |
| Bifunctors | Working with two-parameter type constructors, implementing the bifunctor interface, and practical examples | 🚧 | |
| Arrows | Arrow abstractions beyond monads, implementing the Arrow interface, and creating arrow-based computations | 🚧 | |
| Comonads | Understanding dual concepts to monads, implementing Store and Stream comonads, and practical applications | 🚧 | |
| Design Patterns in Functional Python | Applying FP concepts to solve real-world problems, functional architecture, and testing strategies | 🚧 | |

# Description of notebooks

## 05. Category and Functors

In [this notebook](https://github.com/marimo-team/learn/blob/main/Functional_programming/05_functors.py), you would learn:

* Why `len` is the *Functor* from the category of `list concatentation` to the category of `integer addition`
* How to *lift* an ordinary function to a specific *computation context*
* How to write an *adpter* between two categories

### References

- [Haskellforall.The.Category.Design.Pattern](https://www.haskellforall.com/2012/08/the-category-design-pattern.html)
- [Haskellforall.The.Functor.Design.Pattern](https://www.haskellforall.com/2012/09/the-functor-design-pattern.html)
- [Haskellwiki.Functor](https://wiki.haskell.org/index.php?title=Functor)
- [Haskellwiki.Typeclassopedia#Functor](https://wiki.haskell.org/index.php?title=Typeclassopedia#Functor)
- [Haskellwiki.Typeclassopedia#Category](https://wiki.haskell.org/index.php?title=Typeclassopedia#Category)

**Authors.**

Thanks to all our notebook authors!

* [eugene.hs](https://github.com/metaboulie)
