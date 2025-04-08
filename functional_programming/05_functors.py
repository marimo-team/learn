# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo",
# ]
# ///

import marimo

__generated_with = "0.12.5"
app = marimo.App(app_title="Category Theory and Functors")

@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        # Category Theory and Functors

        In this notebook, you will learn:

        * Why `length` is a *functor* from the category of `list concatenation` to the category of `integer addition`
        * How to *lift* an ordinary function into a specific *computational context*
        * How to write an *adapter* between two categories

        In short, a mathematical functor is a **mapping** between two categories in category theory. In practice, a functor represents a type that can be mapped over.

        /// admonition | Intuitions 

        - A simple intuition is that a `Functor` represents a **container** of values, along with the ability to apply a function uniformly to every element in the container.
        - Another intuition is that a `Functor` represents some sort of **computational context**.
        - Mathematically, `Functors` generalize the idea of a container or a computational context.
        ///

        We will start with intuition, introduce the basics of category theory, and then examine functors from a categorical perspective.

        /// details | Notebook metadata
            type: info

        version: 0.1.4 | last modified: 2025-04-08 | author: [métaboulie](https://github.com/metaboulie)<br/>
        reviewer: [Haleshot](https://github.com/Haleshot)

        ///
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        # Functor as a Computational Context

        A [**Functor**](https://wiki.haskell.org/Functor) is an abstraction that represents a computational context with the ability to apply a function to every value inside it without altering the structure of the context itself. This enables transformations while preserving the shape of the data.

        To understand this, let's look at a simple example.

        ## [The One-Way Wrapper Design Pattern](http://blog.sigfpe.com/2007/04/trivial-monad.html)

        Often, we need to wrap data in some kind of context. However, when performing operations on wrapped data, we typically have to:

        1. Unwrap the data.
        2. Modify the unwrapped data.
        3. Rewrap the modified data.

        This process is tedious and inefficient. Instead, we want to wrap data **once** and apply functions directly to the wrapped data without unwrapping it.

        /// admonition | Rules for a One-Way Wrapper

        1. We can wrap values, but we cannot unwrap them.
        2. We should still be able to apply transformations to the wrapped data.
        3. Any operation that depends on wrapped data should itself return a wrapped result.
        ///

        Let's define such a `Wrapper` class:

        ```python
        from dataclasses import dataclass
        from typing import TypeVar

        A = TypeVar("A")
        B = TypeVar("B")

        @dataclass
        class Wrapper[A]:
            value: A
        ```

        Now, we can create an instance of wrapped data:

        ```python
        wrapped = Wrapper(1)
        ```

        ### Mapping Functions Over Wrapped Data

        To modify wrapped data while keeping it wrapped, we define an `fmap` method:
        """
    )
    return


@app.cell
def _(B, Callable, Functor, dataclass):
    @dataclass
    class Wrapper[A](Functor):
        value: A

        @classmethod
        def fmap(cls, g: Callable[[A], B], fa: "Wrapper[A]") -> "Wrapper[B]":
            return Wrapper(g(fa.value))
    return (Wrapper,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        /// attention

        To distinguish between regular types and functors, we use the prefix `f` to indicate `Functor`.

        For instance,

        - `a: A` is a regular variable of type `A`
        - `g: Callable[[A], B]` is a regular function from type `A` to `B`
        - `fa: Functor[A]` is a *Functor* wrapping a value of type `A`  
        - `fg: Functor[Callable[[A], B]]` is a *Functor* wrapping a function from type `A` to `B`  

        and we will avoid using `f` to represent a function

        ///

        > Try with Wrapper below
        """
    )
    return


@app.cell
def _(Wrapper, pp):
    wrapper = Wrapper(1)

    pp(Wrapper.fmap(lambda x: x + 1, wrapper))
    pp(Wrapper.fmap(lambda x: [x], wrapper))
    return (wrapper,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        We can analyze the type signature of `fmap` for `Wrapper`:

        * `g` is of type `Callable[[A], B]`
        * `fa` is of type `Wrapper[A]`
        * The return value is of type `Wrapper[B]`

        Thus, in Python's type system, we can express the type signature of `fmap` as:

        ```python
        fmap(g: Callable[[A], B], fa: Wrapper[A]) -> Wrapper[B]:
        ```

        Essentially, `fmap`:

        1. Takes a function `Callable[[A], B]` and a `Wrapper[A]` instance as input.
        2. Applies the function to the value inside the wrapper.
        3. Returns a new `Wrapper[B]` instance with the transformed value, leaving the original wrapper and its internal data unmodified.

        Now, let's examine `list` as a similar kind of wrapper.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## The List Functor

        We can define a `List` class to represent a wrapped list that supports `fmap`:
        """
    )
    return


@app.cell
def _(B, Callable, Functor, dataclass):
    @dataclass
    class List[A](Functor):
        value: list[A]

        @classmethod
        def fmap(cls, g: Callable[[A], B], fa: "List[A]") -> "List[B]":
            return List([g(x) for x in fa.value])
    return (List,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""> Try with List below""")
    return


@app.cell
def _(List, pp):
    flist = List([1, 2, 3, 4])
    pp(List.fmap(lambda x: x + 1, flist))
    pp(List.fmap(lambda x: [x], flist))
    return (flist,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### Extracting the Type of `fmap`

        The type signature of `fmap` for `List` is:

        ```python
        fmap(g: Callable[[A], B], fa: List[A]) -> List[B]
        ```

        Similarly, for `Wrapper`:

        ```python
        fmap(g: Callable[[A], B], fa: Wrapper[A]) -> Wrapper[B]
        ```

        Both follow the same pattern, which we can generalize as:

        ```python
        fmap(g: Callable[[A], B], fa: Functor[A]) -> Functor[B]
        ```

        where `Functor` can be `Wrapper`, `List`, or any other wrapper type that follows the same structure.

        ### Functors in Haskell (optional)

        In Haskell, the type of `fmap` is:

        ```haskell
        fmap :: Functor f => (a -> b) -> f a -> f b
        ```

        or equivalently:

        ```haskell
        fmap :: Functor f => (a -> b) -> (f a -> f b)
        ```

        This means that `fmap` **lifts** an ordinary function into the **functor world**, allowing it to operate within a computational context.

        Now, let's define an abstract class for `Functor`.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Defining Functor

        Recall that, a **Functor** is an abstraction that allows us to apply a function to values inside a computational context while preserving its structure. 

        To define `Functor` in Python, we use an abstract base class:

        ```python
        @dataclass
        class Functor[A](ABC):
            @classmethod
            @abstractmethod
            def fmap(g: Callable[[A], B], fa: "Functor[A]") -> "Functor[B]":
                raise NotImplementedError
        ```

        We can now extend custom wrappers, containers, or computation contexts with this `Functor` base class, implement the `fmap` method, and apply any function.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # More Functor instances (optional)

        In this section, we will explore more *Functor* instances to help you build up a better comprehension.

        The main reference is [Data.Functor](https://hackage.haskell.org/package/base-4.21.0.0/docs/Data-Functor.html)
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## The [Maybe](https://hackage.haskell.org/package/base-4.21.0.0/docs/Data-Maybe.html#t:Maybe) Functor

        **`Maybe`** is a functor that can either hold a value (`Just(value)`) or be `Nothing` (equivalent to `None` in Python). 

        - It the value exists, `fmap` applies the function to this value inside the functor.
        - If the value is `None`, `fmap` simply returns `None`.

        /// admonition
        By using `Maybe` as a functor, we gain the ability to apply transformations (`fmap`) to potentially absent values, without having to explicitly handle the `None` case every time.
        ///

        We can implement the `Maybe` functor as:
        """
    )
    return


@app.cell
def _(B, Callable, Functor, dataclass):
    @dataclass
    class Maybe[A](Functor):
        value: None | A

        @classmethod
        def fmap(cls, g: Callable[[A], B], fa: "Maybe[A]") -> "Maybe[B]":
            return cls(None) if fa.value is None else cls(g(fa.value))

        def __repr__(self):
            return "Nothing" if self.value is None else f"Just({self.value!r})"
    return (Maybe,)


@app.cell
def _(Maybe, pp):
    pp(Maybe.fmap(lambda x: x + 1, Maybe(1)))
    pp(Maybe.fmap(lambda x: x + 1, Maybe(None)))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## The [Either](https://hackage.haskell.org/package/base-4.21.0.0/docs/Data-Either.html#t:Either) Functor

        The `Either` type represents values with two possibilities: a value of type `Either a b` is either `Left a` or `Right b`.

        The `Either` type is sometimes used to represent a value which is **either correct or an error**; by convention, the `left` attribute is used to hold an error value and the `right` attribute is used to hold a correct value.

        `fmap` for `Either` will ignore Left values, but will apply the supplied function to values contained in the Right.

        The implementation is: 
        """
    )
    return


@app.cell
def _(B, Callable, Functor, Union, dataclass):
    @dataclass
    class Either[A](Functor):
        left: A = None
        right: A = None

        def __post_init__(self):
            if (self.left is not None and self.right is not None) or (
                self.left is None and self.right is None
            ):
                raise TypeError(
                    "Provide either the value of the left or the value of the right."
                )

        @classmethod
        def fmap(
            cls, g: Callable[[A], B], fa: "Either[A]"
        ) -> Union["Either[A]", "Either[B]"]:
            if fa.left is not None:
                return cls(left=fa.left)
            return cls(right=g(fa.right))

        def __repr__(self):
            if self.left is not None:
                return f"Left({self.left!r})"
            return f"Right({self.right!r})"
    return (Either,)


@app.cell
def _(Either):
    print(Either.fmap(lambda x: x + 1, Either(left=TypeError("Parse Error"))))
    print(Either.fmap(lambda x: x + 1, Either(right=1)))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## The [RoseTree](https://en.wikipedia.org/wiki/Rose_tree) Functor

        A **RoseTree** is a tree where:

        - Each node holds a **value**.
        - Each node has a **list of child nodes** (which are also RoseTrees).

        This structure is useful for representing hierarchical data, such as:

        - Abstract Syntax Trees (ASTs)
        - File system directories
        - Recursive computations

        The implementation is:
        """
    )
    return


@app.cell
def _(B, Callable, Functor, dataclass):
    @dataclass
    class RoseTree[A](Functor):
        value: A  # The value stored in the node.
        children: list[
            "RoseTree[A]"
        ]  # A list of child nodes forming the tree structure.

        @classmethod
        def fmap(cls, g: Callable[[A], B], fa: "RoseTree[A]") -> "RoseTree[B]":
            """
            Applies a function to each value in the tree, producing a new `RoseTree[b]` with transformed values.

            1. `g` is applied to the root node's `value`.
            2. Each child in `children` recursively calls `fmap`.
            """
            return RoseTree(
                g(fa.value), [cls.fmap(g, child) for child in fa.children]
            )

        def __repr__(self) -> str:
            return f"Node: {self.value}, Children: {self.children}"
    return (RoseTree,)


@app.cell
def _(RoseTree, pp):
    rosetree = RoseTree(1, [RoseTree(2, []), RoseTree(3, [RoseTree(4, [])])])

    pp(rosetree)
    pp(RoseTree.fmap(lambda x: [x], rosetree))
    pp(RoseTree.fmap(lambda x: RoseTree(x, []), rosetree))
    return (rosetree,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Generic Functions that can be Used with Any Functor

        One of the powerful features of functors is that we can write **generic functions** that can work with any functor.

        Remember that in Haskell, the type of `fmap` can be written as:

        ```haskell
        fmap :: Functor f => (a -> b) -> (f a -> f b)
        ```

        Translating to Python, we get:

        ```python
        def fmap(g: Callable[[A], B]) -> Callable[[Functor[A]], Functor[B]]
        ```

        This means that `fmap`:

        - Takes an **ordinary function** `Callable[[A], B]` as input.
        - Outputs a function that:
            - Takes a **functor** of type `Functor[A]` as input.
            - Outputs a **functor** of type `Functor[B]`.

        Inspired by this, we can implement an `inc` function which takes a functor, applies the function `lambda x: x + 1` to every value inside it, and returns a new functor with the updated values.
        """
    )
    return


@app.cell
def _():
    inc = lambda functor: functor.fmap(lambda x: x + 1, functor)
    return (inc,)


@app.cell
def _(flist, inc, pp, rosetree, wrapper):
    pp(inc(wrapper))
    pp(inc(flist))
    pp(inc(rosetree))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        /// admonition | exercise
        Implement other generic functions and apply them to different *Functor* instances.
        ///
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Functor laws and utility functions""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Functor laws

        In addition to providing a function `fmap` of the specified type, functors are also required to satisfy two equational laws:

        ```haskell
        fmap id = id                    -- fmap preserves identity
        fmap (g . h) = fmap g . fmap h  -- fmap distributes over composition
        ```

        1. `fmap` should preserve the **identity function**, in the sense that applying `fmap` to this function returns the same function as the result.
        2. `fmap` should also preserve **function composition**. Applying two composed functions `g` and `h` to a functor via `fmap` should give the same result as first applying `fmap` to `g` and then applying `fmap` to `h`.

        /// admonition | 
        - Any `Functor` instance satisfying the first law `(fmap id = id)` will [automatically satisfy the second law](https://github.com/quchen/articles/blob/master/second_functor_law.md) as well.
        ///
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Functor laws verification

        We can define `id` and `compose` in `Python` as:
        """
    )
    return


@app.cell
def _():
    id = lambda x: x
    compose = lambda f, g: lambda x: f(g(x))
    return compose, id


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""We can add a helper function `check_functor_law` to verify that an instance satisfies the functor laws:""")
    return


@app.cell
def _(id):
    check_functor_law = lambda functor: repr(functor.fmap(id, functor)) == repr(
        functor
    )
    return (check_functor_law,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""We can verify the functor we've defined:""")
    return


@app.cell
def _(check_functor_law, flist, pp, rosetree, wrapper):
    for functor in (wrapper, flist, rosetree):
        pp(check_functor_law(functor))
    return (functor,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""And here is an `EvilFunctor`. We can verify it's not a valid `Functor`.""")
    return


@app.cell
def _(B, Callable, Functor, dataclass):
    @dataclass
    class EvilFunctor[A](Functor):
        value: list[A]

        @classmethod
        def fmap(
            cls, g: Callable[[A], B], fa: "EvilFunctor[A]"
        ) -> "EvilFunctor[B]":
            return (
                cls([fa.value[0]] * 2 + [g(x) for x in fa.value[1:]])
                if fa.value
                else []
            )
    return (EvilFunctor,)


@app.cell
def _(EvilFunctor, check_functor_law, pp):
    pp(check_functor_law(EvilFunctor([1, 2, 3, 4])))
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Utility functions

        ```python
        @classmethod
        def const(cls, fa: "Functor[A]", b: B) -> "Functor[B]":
            return cls.fmap(lambda _: b, fa)

        @classmethod
        def void(cls, fa: "Functor[A]") -> "Functor[None]":
            return cls.const(fa, None)

        @classmethod
        def unzip(
            cls, fab: "Functor[tuple[A, B]]"
        ) -> tuple["Functor[A]", "Functor[B]"]:
            return cls.fmap(lambda p: p[0], fab), cls.fmap(lambda p: p[1], fab)
        ```

        - `const` replaces all values inside a functor with a constant `b`
        - `void` is equivalent to `const(fa, None)`, transforming all values in a functor into `None`
        - `unzip` is a generalization of the regular *unzip* on a list of pairs
        """
    )
    return


@app.cell
def _(List, Maybe):
    print(Maybe.const(Maybe(0), 1))
    print(Maybe.const(Maybe(None), 1))
    print(List.const(List([1, 2, 3, 4]), 1))
    return


@app.cell
def _(List, Maybe):
    print(Maybe.void(Maybe(1)))
    print(List.void(List([1, 2, 3])))
    return


@app.cell
def _(List, Maybe):
    print(Maybe.unzip(Maybe(("Hello", "World"))))
    print(List.unzip(List([("I", "love"), ("really", "λ")])))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        /// admonition
        You can always override these utility functions with a more efficient implementation for specific instances.
        ///
        """
    )
    return


@app.cell
def _(List, RoseTree, flist, pp, rosetree):
    pp(RoseTree.const(rosetree, "λ"))
    pp(RoseTree.void(rosetree))
    pp(List.const(flist, "λ"))
    pp(List.void(flist))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""# Formal implementation of Functor""")
    return


@app.cell
def _(ABC, B, Callable, abstractmethod, dataclass):
    @dataclass
    class Functor[A](ABC):
        @classmethod
        @abstractmethod
        def fmap(cls, g: Callable[[A], B], fa: "Functor[A]") -> "Functor[B]":
            return NotImplementedError

        @classmethod
        def const(cls, fa: "Functor[A]", b: B) -> "Functor[B]":
            return cls.fmap(lambda _: b, fa)

        @classmethod
        def void(cls, fa: "Functor[A]") -> "Functor[None]":
            return cls.const(fa, None)

        @classmethod
        def unzip(
            cls, fab: "Functor[tuple[A, B]]"
        ) -> tuple["Functor[A]", "Functor[B]"]:
            return cls.fmap(lambda p: p[0], fab), cls.fmap(lambda p: p[1], fab)
    return (Functor,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Limitations of Functor

        Functors abstract the idea of mapping a function over each element of a structure. Suppose now that we wish to generalise this idea to allow functions with any number of arguments to be mapped, rather than being restricted to functions with a single argument. More precisely, suppose that we wish to define a hierarchy of `fmap` functions with the following types:

        ```haskell
        fmap0 :: a -> f a

        fmap1 :: (a -> b) -> f a -> f b

        fmap2 :: (a -> b -> c) -> f a -> f b -> f c

        fmap3 :: (a -> b -> c -> d) -> f a -> f b -> f c -> f d
        ```

        And we have to declare a special version of the functor class for each case.

        We will learn how to resolve this problem in the next notebook on `Applicatives`.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        # Introduction to Categories

        A [category](https://en.wikibooks.org/wiki/Haskell/Category_theory#Introduction_to_categories) is, in essence, a simple collection. It has three components: 

        - A collection of **objects**.
        - A collection of **morphisms**, each of which ties two objects (a _source object_ and a _target object_) together. If $f$ is a morphism with source object $C$ and target object $B$, we write $f : C → B$.
        - A notion of **composition** of these morphisms. If $g : A → B$ and $f : B → C$ are two morphisms, they can be composed, resulting in a morphism $f ∘ g : A → C$.

        ## Category laws

        There are three laws that categories need to follow. 

        1. The composition of morphisms needs to be **associative**. Symbolically, $f ∘ (g ∘ h) = (f ∘ g) ∘ h$

            - Morphisms are applied right to left, so with $f ∘ g$ first $g$ is applied, then $f$. 

        2. The category needs to be **closed** under the composition operation. So if $f : B → C$ and $g : A → B$, then there must be some morphism $h : A → C$ in the category such that $h = f ∘ g$. 

        3. Given a category $C$ there needs to be for every object $A$ an **identity** morphism, $id_A : A → A$ that is an identity of composition with other morphisms. Put precisely, for every morphism $g : A → B$: $g ∘ id_A = id_B ∘ g = g$

        /// attention | The definition of a category does not define: 

        - what `∘` is,
        - what `id` is, or
        - what `f`, `g`, and `h` might be. 

        Instead, category theory leaves it up to us to discover what they might be.
        ///
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## The Python category

        The main category we'll be concerning ourselves with in this part is the Python category, or we can give it a shorter name: `Py`. `Py` treats Python types as objects and Python functions as morphisms. A function `def f(a: A) -> B` for types A and B is a morphism in Python.

        Remember that we defined the `id` and `compose` function above as:

        ```Python
        def id(x: A) -> A:
            return x

        def compose(f: Callable[[B], C], g: Callable[[A], B]) -> Callable[[A], C]:
            return lambda x: f(g(x))  
        ```

        We can check second law easily. 

        For the first law, we have:

        ```python
        # compose(f, g) = lambda x: f(g(x))
        f ∘ (g ∘ h) 
        = compose(f, compose(g, h)) 
        = lambda x: f(compose(g, h)(x))
        = lambda x: f(lambda y: g(h(y))(x))
        = lambda x: f(g(h(x)))

        (f ∘ g) ∘ h 
        = compose(compose(f, g), h)
        = lambda x: compose(f, g)(h(x))
        = lambda x: lambda y: f(g(y))(h(x))
        = lambda x: f(g(h(x)))
        ```

        For the third law, we have: 

        ```python
        g ∘ id_A 
        = compose(g: Callable[[a], b], id: Callable[[a], a]) -> Callable[[a], b]
        = lambda x: g(id(x))
        = lambda x: g(x) # id(x) = x
        = g
        ```
        the similar proof can be applied to $id_B ∘ g =g$.

        Thus `Py` is a valid category.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        # Functors, again

        A functor is essentially a transformation between categories, so given categories $C$ and $D$, a functor $F : C → D$:

        - Maps any object $A$ in $C$ to $F ( A )$, in $D$.
        - Maps morphisms $f : A → B$ in $C$ to $F ( f ) : F ( A ) → F ( B )$ in $D$.

        /// admonition | 

        Endofunctors are functors from a category to itself.

        ///
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Functors on the category of Python

        Remember that a functor has two parts: it maps objects in one category to objects in another and morphisms in the first category to morphisms in the second. 

        Functors in Python are from `Py` to `Func`, where `Func` is the subcategory of `Py` defined on just that functor's types. E.g. the RoseTree functor goes from `Py` to `RoseTree`, where `RoseTree` is the category containing only RoseTree types, that is, `RoseTree[T]` for any type `T`. The morphisms in `RoseTree` are functions defined on RoseTree types, that is, functions `Callable[[RoseTree[T]], RoseTree[U]]` for types `T`, `U`.

        Recall the definition of `Functor`:

        ```Python
        @dataclass
        class Functor[A](ABC)
        ```

        And RoseTree: 

        ```Python
        @dataclass
        class RoseTree[A](Functor)
        ```

        **Here's the key part:** the _type constructor_ `RoseTree` takes any type `T` to a new type, `RoseTree[T]`. Also, `fmap` restricted to `RoseTree` types takes a function `Callable[[A], B]` to a function `Callable[[RoseTree[A]], RoseTree[B]]`.

        But that's it. We've defined two parts, something that takes objects in `Py` to objects in another category (that of `RoseTree` types and functions defined on `RoseTree` types), and something that takes morphisms in `Py` to morphisms in this category. So `RoseTree` is a functor. 

        To sum up:

        - We work in the category **Py** and its subcategories.  
        - **Objects** are types (e.g., `int`, `str`, `list`).  
        - **Morphisms** are functions (`Callable[[A], B]`).  
        - **Things that take a type and return another type** are type constructors (`RoseTree[T]`).  
        - **Things that take a function and return another function** are higher-order functions (`Callable[[Callable[[A], B]], Callable[[C], D]]`).  
        - **Abstract base classes (ABC)** and duck typing provide a way to express polymorphism, capturing the idea that in category theory, structures are often defined over multiple objects at once.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Functor laws, again

        Once again there are a few axioms that functors have to obey. 

        1. Given an identity morphism $id_A$ on an object $A$, $F ( id_A )$ must be the identity morphism on $F ( A )$, i.e.: $F({id} _{A})={id} _{F(A)}$
        2. Functors must distribute over morphism composition, i.e. $F(f\circ g)=F(f)\circ F(g)$
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        Remember that we defined the `id` and `compose` as 
        ```python
        id = lambda x: x
        compose = lambda f, g: lambda x: f(g(x))
        ```

        We can define `fmap` as: 

        ```python
        fmap = lambda g, functor: functor.fmap(g, functor)  
        ```

        Let's prove that `fmap` is a functor.

        First, let's define a `Category` for a specific `Functor`. We choose to define the `Category` for the `Wrapper` as `WrapperCategory` here for simplicity, but remember that `Wrapper` can be any `Functor`(i.e. `List`, `RoseTree`, `Maybe` and more):

        We define `WrapperCategory` as:

        ```python
        @dataclass
        class WrapperCategory:
            @staticmethod
            def id(wrapper: Wrapper[A]) -> Wrapper[A]:
                return Wrapper(wrapper.value)

            @staticmethod
            def compose(
                f: Callable[[Wrapper[B]], Wrapper[C]],
                g: Callable[[Wrapper[A]], Wrapper[B]],
                wrapper: Wrapper[A]
            ) -> Callable[[Wrapper[A]], Wrapper[C]]:
                return f(g(Wrapper(wrapper.value)))
        ```

        And `Wrapper` is:

        ```Python
        @dataclass
        class Wrapper[A](Functor):
            value: A

            @classmethod
            def fmap(cls, g: Callable[[A], B], fa: "Wrapper[A]") -> "Wrapper[B]":
                return Wrapper(g(fa.value))
        ```
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        We can prove that:

        ```python
        fmap(id, wrapper)
        = Wrapper.fmap(id, wrapper)
        = Wrapper(id(wrapper.value))
        = Wrapper(wrapper.value)
        = WrapperCategory.id(wrapper)
        ```
        and:
        ```python
        fmap(compose(f, g), wrapper)
        = Wrapper.fmap(compose(f, g), wrapper)
        = Wrapper(compose(f, g)(wrapper.value))
        = Wrapper(f(g(wrapper.value)))

        WrapperCategory.compose(fmap(f, wrapper), fmap(g, wrapper), wrapper)
        = fmap(f, wrapper)(fmap(g, wrapper)(wrapper))
        = fmap(f, wrapper)(Wrapper.fmap(g, wrapper))
        = fmap(f, wrapper)(Wrapper(g(wrapper.value)))
        = Wrapper.fmap(f, Wrapper(g(wrapper.value)))
        = Wrapper(f(Wrapper(g(wrapper.value)).value))
        = Wrapper(f(g(wrapper.value)))  # Wrapper(g(wrapper.value)).value = g(wrapper.value)
        ```

        So our `Wrapper` is a valid `Functor`.

        > Try validating functor laws for `Wrapper` below.
        """
    )
    return


@app.cell
def _(A, B, C, Callable, Wrapper, dataclass):
    @dataclass
    class WrapperCategory:
        @staticmethod
        def id(wrapper: Wrapper[A]) -> Wrapper[A]:
            return Wrapper(wrapper.value)

        @staticmethod
        def compose(
            f: Callable[[Wrapper[B]], Wrapper[C]],
            g: Callable[[Wrapper[A]], Wrapper[B]],
            wrapper: Wrapper[A],
        ) -> Callable[[Wrapper[A]], Wrapper[C]]:
            return f(g(Wrapper(wrapper.value)))
    return (WrapperCategory,)


@app.cell
def _(WrapperCategory, id, pp, wrapper):
    pp(wrapper.fmap(id, wrapper) == WrapperCategory.id(wrapper))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Length as a Functor

        Remember that a functor is a transformation between two categories. It is not only limited to a functor from `Py` to `Func`, but also includes transformations between other mathematical structures.

        Let’s prove that **`length`** can be viewed as a functor. Specifically, we will demonstrate that `length` is a functor from the **category of list concatenation** to the **category of integer addition**.

        ### Category of List Concatenation

        First, let’s define the category of list concatenation:
        """
    )
    return


@app.cell
def _(A, dataclass):
    @dataclass
    class ListConcatenation[A]:
        value: list[A]

        @staticmethod
        def id() -> "ListConcatenation[A]":
            return ListConcatenation([])

        @staticmethod
        def compose(
            this: "ListConcatenation[A]", other: "ListConcatenation[A]"
        ) -> "ListConcatenation[a]":
            return ListConcatenation(this.value + other.value)
    return (ListConcatenation,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        - **Identity**: The identity element is an empty list (`ListConcatenation([])`).
        - **Composition**: The composition of two lists is their concatenation (`this.value + other.value`).
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### Category of Integer Addition

        Now, let's define the category of integer addition:
        """
    )
    return


@app.cell
def _(dataclass):
    @dataclass
    class IntAddition:
        value: int

        @staticmethod
        def id() -> "IntAddition":
            return IntAddition(0)

        @staticmethod
        def compose(this: "IntAddition", other: "IntAddition") -> "IntAddition":
            return IntAddition(this.value + other.value)
    return (IntAddition,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        - **Identity**: The identity element is `IntAddition(0)` (the additive identity).
        - **Composition**: The composition of two integers is their sum (`this.value + other.value`).
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### Defining the Length Functor

        We now define the `length` function as a functor, mapping from the category of list concatenation to the category of integer addition:

        ```python
        length = lambda l: IntAddition(len(l.value))
        ```
        """
    )
    return


@app.cell(hide_code=True)
def _(IntAddition):
    length = lambda l: IntAddition(len(l.value))
    return (length,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""This function takes an instance of `ListConcatenation`, computes its length, and returns an `IntAddition` instance with the computed length.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### Verifying Functor Laws

        Now, let’s verify that `length` satisfies the two functor laws.

        **Identity Law**

        The identity law states that applying the functor to the identity element of one category should give the identity element of the other category.
        """
    )
    return


@app.cell
def _(IntAddition, ListConcatenation, length, pp):
    pp(length(ListConcatenation.id()) == IntAddition.id())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""This ensures that the length of an empty list (identity in the `ListConcatenation` category) is `0` (identity in the `IntAddition` category).""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        **Composition Law**

        The composition law states that the functor should preserve composition. Applying the functor to a composed element should be the same as composing the functor applied to the individual elements.
        """
    )
    return


@app.cell
def _(IntAddition, ListConcatenation, length, pp):
    lista = ListConcatenation([1, 2])
    listb = ListConcatenation([3, 4])
    pp(
        length(ListConcatenation.compose(lista, listb))
        == IntAddition.compose(length(lista), length(listb))
    )
    return lista, listb


@app.cell(hide_code=True)
def _(mo):
    mo.md("""This ensures that the length of the concatenation of two lists is the same as the sum of the lengths of the individual lists.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        # Further reading

        - [The Trivial Monad](http://blog.sigfpe.com/2007/04/trivial-monad.html)
        - [Haskellforall: The Category Design Pattern](https://www.haskellforall.com/2012/08/the-category-design-pattern.html)
        - [Haskellforall: The Functor Design Pattern](https://www.haskellforall.com/2012/09/the-functor-design-pattern.html)

            /// attention | ATTENTION 
            The functor design pattern doesn't work at all if you aren't using categories in the first place. This is why you should structure your tools using the compositional category design pattern so that you can take advantage of functors to easily mix your tools together. 
            ///

        - [Haskellwiki: Functor](https://wiki.haskell.org/index.php?title=Functor)
        - [Haskellwiki: Typeclassopedia#Functor](https://wiki.haskell.org/index.php?title=Typeclassopedia#Functor)
        - [Haskellwiki: Typeclassopedia#Category](https://wiki.haskell.org/index.php?title=Typeclassopedia#Category)
        - [Haskellwiki: Category Theory](https://en.wikibooks.org/wiki/Haskell/Category_theory)
        """
    )
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _():
    from abc import abstractmethod, ABC
    return ABC, abstractmethod


@app.cell(hide_code=True)
def _():
    from dataclasses import dataclass
    from typing import Callable, TypeVar, Union
    from pprint import pp
    return Callable, TypeVar, Union, dataclass, pp


@app.cell(hide_code=True)
def _(TypeVar):
    A = TypeVar("A")
    B = TypeVar("B")
    C = TypeVar("C")
    return A, B, C


if __name__ == "__main__":
    app.run()
