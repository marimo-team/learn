# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "marimo",
# ]
# ///

import marimo

__generated_with = "0.11.17"
app = marimo.App(app_title="Category Theory and Functors", css_file="")


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

        version: 0.1.0 | last modified: 2025-03-13 | author: [métaboulie](https://github.com/metaboulie)<br/>
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
        from typing import Callable, Generic, TypeVar

        a = TypeVar("a")
        b = TypeVar("b")

        @dataclass
        class Wrapper(Generic[a]):
            value: a
        ```

        Now, we can create an instance of wrapped data:

        ```python
        wrapped = Wrapper(1)
        ```

        ### Mapping Functions Over Wrapped Data

        To modify wrapped data while keeping it wrapped, we define an `fmap` method:

        ```python
        @dataclass
        class Wrapper(Generic[a]):
            value: a

            def fmap(self, func: Callable[[a], b]) -> "Wrapper[b]":
                return Wrapper(func(self.value))
        ```

        Now, we can apply transformations without unwrapping:

        ```python
        >>> wrapped.fmap(lambda x: x + 1)
        Wrapper(value=2)

        >>> wrapped.fmap(lambda x: [x])
        Wrapper(value=[1])
        ```

        > Try using the `Wrapper` in the cell below.
        """
    )
    return


@app.cell
def _(Callable, Functor, Generic, a, b, dataclass):
    @dataclass
    class Wrapper(Functor, Generic[a]):
        value: a

        def fmap(self, func: Callable[[a], b]) -> "Wrapper[b]":
            return Wrapper(func(self.value))

        def __repr__(self):
            return repr(self.value)


    wrapper = Wrapper(1)
    return Wrapper, wrapper


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        We can analyze the type signature of `fmap` for `Wrapper`:

        * `self` is of type `Wrapper[a]`
        * `func` is of type `Callable[[a], b]`
        * The return value is of type `Wrapper[b]`

        Thus, in Python's type system, we can express the type signature of `fmap` as:

        ```python
        def fmap(self: Wrapper[a], func: Callable[[a], b]) -> Wrapper[b]:
        ```

        Essentially, `fmap`:

        1. Takes a `Wrapper[a]` instance and a function `Callable[[a], b]` as input.
        2. Applies the function to the value inside the wrapper.
        3. Returns a new `Wrapper[b]` instance with the transformed value, leaving the original wrapper and its internal data unmodified.

        Now, let's examine `list` as a similar kind of wrapper.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## The List Wrapper

        We can define a `ListWrapper` class to represent a wrapped list that supports `fmap`:
        """
    )
    return


@app.cell
def _(Callable, Functor, Generic, a, b, dataclass):
    @dataclass
    class ListWrapper(Functor, Generic[a]):
        value: list[a]

        def fmap(self, func: Callable[[a], b]) -> "ListWrapper[b]":
            return ListWrapper([func(x) for x in self.value])

        def __repr__(self):
            return repr(self.value)


    list_wrapper = ListWrapper([1, 2, 3, 4])
    return ListWrapper, list_wrapper


@app.cell
def _(ListWrapper, mo):
    with mo.redirect_stdout():
        print(ListWrapper(value=[2, 3, 4, 5]))
        print(ListWrapper(value=[[1], [2], [3], [4]]))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### Extracting the Type of `fmap`

        The type signature of `fmap` for `ListWrapper` is:

        ```python
        def fmap(self: ListWrapper[a], func: Callable[[a], b]) -> ListWrapper[b]
        ```

        Similarly, for `Wrapper`:

        ```python
        def fmap(self: Wrapper[a], func: Callable[[a], b]) -> Wrapper[b]
        ```

        Both follow the same pattern, which we can generalize as:

        ```python
        def fmap(self: Functor[a], func: Callable[[a], b]) -> Functor[b]
        ```

        where `Functor` can be `Wrapper`, `ListWrapper`, or any other wrapper type that follows the same structure.

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
        from dataclasses import dataclass
        from typing import Callable, Generic, TypeVar
        from abc import ABC, abstractmethod

        a = TypeVar("a")
        b = TypeVar("b")

        @dataclass
        class Functor(ABC, Generic[a]):
            @abstractmethod
            def fmap(self, func: Callable[[a], b]) -> "Functor[b]":
                raise NotImplementedError
        ```

        We can now extend custom wrappers, containers, or computation contexts with this `Functor` base class, implement the `fmap` method, and apply any function.

        Next, let's implement a more complex data structure: [RoseTree](https://en.wikipedia.org/wiki/Rose_tree).
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Case Study: RoseTree

        A **RoseTree** is a tree where:

        - Each node holds a **value**.
        - Each node has a **list of child nodes** (which are also RoseTrees).

        This structure is useful for representing hierarchical data, such as:

        - Abstract Syntax Trees (ASTs)
        - File system directories
        - Recursive computations

        We can implement `RoseTree` by extending the `Functor` class:

        ```python
        from dataclasses import dataclass
        from typing import Callable, Generic, TypeVar

        a = TypeVar("a")
        b = TypeVar("b")

        @dataclass
        class RoseTree(Functor, Generic[a]):
            value: a
            children: list["RoseTree[a]"]

            def fmap(self, func: Callable[[a], b]) -> "RoseTree[b]":
                return RoseTree(
                    func(self.value), [child.fmap(func) for child in self.children]
                )

            def __repr__(self) -> str:
                return f"RoseNode({self.value}, {self.children})"
        ```

        - The function is applied **recursively** to each node's value.
        - The tree structure **remains unchanged**.
        - Only the values inside the tree are modified.

        > Try using `RoseTree` in the cell below.
        """
    )
    return


@app.cell(hide_code=True)
def _(Callable, Functor, Generic, a, b, dataclass, mo):
    @dataclass
    class RoseTree(Functor, Generic[a]):
        """
        ### Doc: RoseTree

        A Functor implementation of `RoseTree`, allowing transformation of values while preserving the tree structure.

        **Attributes**

        - `value (a)`: The value stored in the node.
        - `children (list[RoseTree[a]])`: A list of child nodes forming the tree structure.

        **Methods:**

        - `fmap(func: Callable[[a], b]) -> RoseTree[b]`
          ```Python
          def fmap(RoseTree[a], (a -> b)) -> RoseTree[b]
          ```
          Applies a function to each value in the tree, producing a new `RoseTree[b]` with transformed values.

        **Implementation logic:**

          - The function `func` is applied to the root node's `value`.
          - Each child in `children` recursively calls `fmap`, ensuring all values in the tree are mapped.
          - The overall tree structure remains unchanged.

        - `__repr__() -> str`: Returns a string representation of the node and its children.
        """

        value: a
        children: list["RoseTree[a]"]

        def fmap(self, func: Callable[[a], b]) -> "RoseTree[b]":
            return RoseTree(
                func(self.value), [child.fmap(func) for child in self.children]
            )

        def __repr__(self) -> str:
            return f"RoseNode({self.value}, {self.children})"


    mo.md(RoseTree.__doc__)
    return (RoseTree,)


@app.cell(hide_code=True)
def _(RoseTree, mo):
    ftree = RoseTree(1, [RoseTree(2, []), RoseTree(3, [RoseTree(4, [])])])

    with mo.redirect_stdout():
        print(ftree)
        print(ftree.fmap(lambda x: [x]))
        print(ftree.fmap(lambda x: RoseTree(x, [])))
    return (ftree,)


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
        def fmap(func: Callable[[a], b]) -> Callable[[Functor[a]], Functor[b]]
        ```

        This means that `fmap`:

        - Takes an **ordinary function** `Callable[[a], b]` as input.
        - Outputs a function that:
            - Takes a **functor** of type `Functor[a]` as input.
            - Outputs a **functor** of type `Functor[b]`.

        We can implement a similar idea in Python:

        ```python
        # fmap(func: Callable[[a], b]) -> Callable[[Functor[a]], Functor[b]]
        fmap = lambda func: lambda f: f.fmap(lambda x: func(x))

        # inc([Functor[a]) -> Functor[b]
        inc = fmap(lambda x: x + 1)
        ```

        - **`fmap`**: Lifts an ordinary function (`lambda x: func(x)`) to the functor world, allowing the function to operate on the wrapped value inside the functor.
        - **`inc`**: A specific instance of `fmap` that operates on any functor. It takes a functor, applies the function `lambda x: x + 1` to every value inside it, and returns a new functor with the updated values.

        Thus, **`fmap`** transforms an ordinary function into a **function that operates on functors**, and **`inc`** is a specific case where it increments the value inside the functor.

        ### Applying the `inc` Function to Various Functors

        You can now apply `inc` to any functor like `Wrapper`, `ListWrapper`, or `RoseTree`:

        ```python
        # Applying `inc` to a Wrapper
        wrapper = Wrapper(5)
        inc(wrapper)  # Wrapper(value=6)

        # Applying `inc` to a ListWrapper
        list_wrapper = ListWrapper([1, 2, 3])
        inc(list_wrapper)  # ListWrapper(value=[2, 3, 4])

        # Applying `inc` to a RoseTree
        tree = RoseTree(1, [RoseTree(2, []), RoseTree(3, [])])
        inc(tree)  # RoseTree(value=2, children=[RoseTree(value=3, children=[]), RoseTree(value=4, children=[])])
        ```

        > Try using `fmap` in the cell below.
        """
    )
    return


@app.cell(hide_code=True)
def _(ftree, list_wrapper, mo, wrapper):
    fmap = lambda func: lambda f: f.fmap(func)
    inc = fmap(lambda x: x + 1)
    with mo.redirect_stdout():
        print(inc(wrapper))
        print(inc(list_wrapper))
        print(inc(ftree))
    return fmap, inc


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
        - Any `Functor` instance satisfying the first law `(fmap id = id)` will automatically satisfy the [second law](https://github.com/quchen/articles/blob/master/second_functor_law.mo) as well.
        ///

        ### Functor Law Verification

        We can add a helper function `check_functor_law` in the `Functor` class to verify that an instance satisfies the functor laws.

        ```Python
        id = lambda x: x

        @dataclass
        class Functor(ABC, Generic[a]):
            @abstractmethod
            def fmap(self, func: Callable[[a], b]) -> "Functor[b]":
                return NotImplementedError

            def check_functor_law(self):
                return repr(self.fmap(id)) == repr(self)

            @abstractmethod
            def __repr__(self):
                return NotImplementedError
        ```

        We can verify the functor we've defined.
        """
    )
    return


@app.cell
def _():
    id = lambda x: x
    compose = lambda f, g: lambda x: f(g(x))
    return compose, id


@app.cell
def _(ftree, list_wrapper, mo, wrapper):
    with mo.redirect_stdout():
        print(wrapper.check_functor_law())
        print(list_wrapper.check_functor_law())
        print(ftree.check_functor_law())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""And here is an `EvilFunctor`. We can verify it's not a valid `Functor`.""")
    return


@app.cell
def _(Callable, Functor, Generic, a, b, dataclass):
    @dataclass
    class EvilFunctor(Functor, Generic[a]):
        value: list[a]

        def fmap(self, func: Callable[[a], b]) -> "EvilFunctor[b]":
            return (
                EvilFunctor([self.value[0]] * 2 + list(map(func, self.value[1:])))
                if self.value
                else []
            )

        def __repr__(self):
            return repr(self.value)
    return (EvilFunctor,)


@app.cell
def _(EvilFunctor):
    EvilFunctor([1, 2, 3, 4]).check_functor_law()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Final definition of Functor

        We can now draft the final definition of `Functor` with some utility functions.

        ```Python
        @dataclass
        class Functor(ABC, Generic[a]):
            @abstractmethod
            def fmap(self, func: Callable[[a], b]) -> "Functor[b]":
                return NotImplementedError

            def check_functor_law(self) -> bool:
                return repr(self.fmap(id)) == repr(self)

            def const_fmap(self, b) -> "Functor[b]":
                return self.fmap(lambda _: b)

            def void(self) -> "Functor[None]":
                return self.const_fmap(None)

            @abstractmethod
            def __repr__(self):
                return NotImplementedError
        ```
        """
    )
    return


@app.cell(hide_code=True)
def _(ABC, Callable, Generic, a, abstractmethod, b, dataclass, id, mo):
    @dataclass
    class Functor(ABC, Generic[a]):
        """
        ### Doc: Functor

        A generic interface for types that support mapping over their values.

        **Methods:**

        - `fmap(func: Callable[[a], b]) -> Functor[b]`
          Abstract method to apply a function `func` to transform the values inside the Functor.

        - `check_functor_law() -> bool`
          Verifies the identity law of functors: `fmap(id) == id`.
          This ensures that applying `fmap` with the identity function does not alter the structure.

        - `const_fmap(b) -> Functor[b]`
          Replaces all values inside the Functor with a constant `b`, preserving the original structure.

        - `void() -> Functor[None]`
          Equivalent to `const_fmap(None)`, transforming all values into `None`.

        - `__repr__()`
          Abstract method to define a string representation of the Functor.

        **Functor Laws:**
        A valid Functor implementation must satisfy:

        1. **Identity Law:** `F.fmap(id) == F`
        2. **Composition Law:** `F.fmap(f).fmap(g) == F.fmap(lambda x: g(f(x)))`
        """

        @abstractmethod
        def fmap(self, func: Callable[[a], b]) -> "Functor[b]":
            return NotImplementedError

        def check_functor_law(self) -> bool:
            return repr(self.fmap(id)) == repr(self)

        def const_fmap(self, b) -> "Functor[b]":
            return self.fmap(lambda _: b)

        def void(self) -> "Functor[None]":
            return self.const_fmap(None)

        @abstractmethod
        def __repr__(self):
            return NotImplementedError


    mo.md(Functor.__doc__)
    return (Functor,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""> Try with utility functions in the cell below""")
    return


@app.cell(hide_code=True)
def _(ftree, list_wrapper, mo):
    with mo.redirect_stdout():
        print(ftree.const_fmap("λ"))
        print(ftree.void())
        print(list_wrapper.const_fmap("λ"))
        print(list_wrapper.void())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Functors for Non-Iterable Types

        In the previous examples, we implemented functors for **iterables**, like `ListWrapper` and `RoseTree`, which are inherently **iterable types**. This is a natural fit for functors, as iterables can be mapped over.

        However, **functors are not limited to iterables**. There are cases where we want to apply the concept of functors to types that are not inherently iterable, such as types that represent optional values, computations, or other data structures.

        ### The Maybe Functor

        One example is the **`Maybe`** type from Haskell, which is used to represent computations that can either result in a value (`Just a`) or no value (`Nothing`). 

        We can define the `Maybe` functor as below:
        """
    )
    return


@app.cell
def _(Callable, Functor, Generic, a, b, dataclass):
    @dataclass
    class Just(Generic[a]):
        value: a

        def __init__(self, value: a):
            # If the value is already a `Just`, we extract the value, else we wrap it
            self.value = value.value if isinstance(value, Just) else value

        def __repr__(self):
            return f"Just {self.value}"


    @dataclass
    class Maybe(Functor, Generic[a]):
        value: None | Just[a]

        def fmap(self, func: Callable[[a], b]) -> "Maybe[b]":
            # Apply the function to the value inside `Just`, or return `Nothing` if value is None
            return (
                Maybe(Just(func(self.value.value))) if self.value else Maybe(None)
            )

        def __repr__(self):
            return repr(self.value) if self.value else "Nothing"
    return Just, Maybe


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        - **`Just`** is a wrapper that holds a value. We use it to represent the presence of a value.
        - **`Maybe`** is a functor that can either hold a `Just` value or be `Nothing` (equivalent to `None` in Python). The `fmap` method applies a function to the value inside the `Just` wrapper, if it exists. If the value is `None` (representing `Nothing`), `fmap` simply returns `Nothing`.

        By using `Maybe` as a functor, we gain the ability to apply transformations (`fmap`) to potentially absent values, without having to explicitly handle the `None` case every time.

        > Try using `Maybe` in the cell below.
        """
    )
    return


@app.cell
def _(Just, Maybe, ftree):
    mftree = Maybe(Just(ftree))
    mint = Maybe(Just(1))
    mnone = Maybe(None)
    return mftree, mint, mnone


@app.cell(hide_code=True)
def _(inc, mftree, mint, mnone, mo):
    with mo.redirect_stdout():
        print(mftree.check_functor_law())
        print(mint.check_functor_law())
        print(mnone.check_functor_law())
        print(mftree.fmap(inc))
        print(mint.fmap(lambda x: x + 1))
        print(mnone.fmap(lambda x: x + 1))
    return


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
        def id(x: Generic[a]) -> Generic[a]:
            return x

        def compose(f: Callable[[b], c], g: Callable[[a], b]) -> Callable[[a], c]:
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

        > Endofunctors are functors from a category to itself.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Functors on the category of Python

        Remember that a functor has two parts: it maps objects in one category to objects in another and morphisms in the first category to morphisms in the second. 

        Functors in Python are from `Py` to `func`, where `func` is the subcategory of `Py` defined on just that functor's types. E.g. the RoseTree functor goes from `Py` to `RoseTree`, where `RoseTree` is the category containing only RoseTree types, that is, `RoseTree[T]` for any type `T`. The morphisms in `RoseTree` are functions defined on RoseTree types, that is, functions `RoseTree[T] -> RoseTree[U]` for types `T`, `U`.

        Recall the definition of `Functor`:

        ```Python
        @dataclass
        class Functor(ABC, Generic[a])
        ```

        And RoseTree: 

        ```Python
        @dataclass
        class RoseTree(Functor, Generic[a])
        ```

        **Here's the key part:** the _type constructor_ `RoseTree` takes any type `T` to a new type, `RoseTree[T]`. Also, `fmap` restricted to `RoseTree` types takes a function `a -> b` to a function `RoseTree[a] -> RoseTree[b]`.

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

        1. Given an identity morphism $id_A$ on an object $A$, $F ( id_A )$ must be the identity morphism on $F ( A )$, i.e.: ${\displaystyle F(\operatorname {id} _{A})=\operatorname {id} _{F(A)}}$
        2. Functors must distribute over morphism composition, i.e. ${\displaystyle F(f\circ g)=F(f)\circ F(g)}$
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        Remember that we defined the `fmap` (not the `fmap` in `Functor` class) and `id` as 
        ```python
        # fmap :: Callable[[a], b] -> Callable[[Functor[a]], Functor[b]]
        fmap = lambda func: lambda f: f.fmap(func)
        id = lambda x: x
        compose = lambda f, g: lambda x: f(g(x))
        ```

        Let's prove that `fmap` is a functor.

        First, let's define a `Category` for a specific `Functor`. We choose to define the `Category` for the `Wrapper` as `WrapperCategory` here for simplicity, but remember that `Wrapper` can be any `Functor`(i.e. `ListWrapper`, `RoseTree`, `Maybe` and more):

        **Notice that** in this case, we can actually view `fmap` as:
        ```python
        # fmap :: Callable[[a], b] -> Callable[[Wrapper[a]], Wrapper[b]]
        fmap = lambda func: lambda wrapper: wrapper.fmap(func)
        ```

        We define `WrapperCategory` as:

        ```python
        @dataclass
        class WrapperCategory():
            @staticmethod
            def id() -> Callable[[Wrapper[a]], Wrapper[a]]:
                return lambda wrapper: Wrapper(wrapper.value)

            @staticmethod
            def compose(
                f: Callable[[Wrapper[b]], Wrapper[c]],
                g: Callable[[Wrapper[a]], Wrapper[b]],
            )   -> Callable[[Wrapper[a]], Wrapper[c]]:
                return lambda wrapper: f(g(Wrapper(wrapper.value)))
        ```

        And `Wrapper` is:

        ```Python
        @dataclass
        class Wrapper(Generic[a]):
            value: a

            def fmap(self, func: Callable[[a], b]) -> "Wrapper[b]":
                return Wrapper(func(self.value))
        ```
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        notice that

        ```python
        fmap(f)(wrapper) = wrapper.fmap(f)
        ```

        We can get:

        ```python
        fmap(id)
        = lambda wrapper: wrapper.fmap(id)
        = lambda wrapper: Wrapper(id(wrapper.value))
        = lambda wrapper: Wrapper(wrapper.value)
        = WrapperCategory.id()
        ```
        And:
        ```python
        fmap(compose(f, g))
        = lambda wrapper: wrapper.fmap(compose(f, g))
        = lambda wrapper: Wrapper(compose(f, g)(wrapper.value))
        = lambda wrapper: Wrapper(f(g(wrapper.value)))

        WrapperCategory.compose(fmap(f), fmap(g))
        = lambda wrapper: fmap(f)(fmap(g)(wrapper))
        = lambda wrapper: fmap(f)(wrapper.fmap(g))
        = lambda wrapper: fmap(f)(Wrapper(g(wrapper.value)))
        = lambda wrapper: Wrapper(g(wrapper.value)).fmap(f)
        = lambda wrapper: Wrapper(f(Wrapper(g(wrapper.value)).value))
        = lambda wrapper: Wrapper(f(g(wrapper.value)))
        = fmap(compose(f, g))
        ```

        So our `Wrapper` is a valid `Functor`.

        > Try validating functor laws for `Wrapper` below.
        """
    )
    return


@app.cell(hide_code=True)
def _(Callable, Wrapper, a, b, c, dataclass):
    @dataclass
    class WrapperCategory:
        @staticmethod
        def id() -> Callable[[Wrapper[a]], Wrapper[a]]:
            return lambda wrapper: Wrapper(wrapper.value)

        @staticmethod
        def compose(
            f: Callable[[Wrapper[b]], Wrapper[c]],
            g: Callable[[Wrapper[a]], Wrapper[b]],
        ) -> Callable[[Wrapper[a]], Wrapper[c]]:
            return lambda wrapper: f(g(Wrapper(wrapper.value)))
    return (WrapperCategory,)


@app.cell(hide_code=True)
def _(WrapperCategory, compose, fmap, id, mo, wrapper):
    with mo.redirect_stdout():
        print(fmap(id)(wrapper) == id(wrapper))
        print(
            fmap(compose(lambda x: x + 1, lambda x: x * 2))(wrapper)
            == WrapperCategory.compose(
                fmap(lambda x: x + 1), fmap(lambda x: x * 2)
            )(wrapper)
        )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Length as a Functor

        Remember that a functor is a transformation between two categories. It is not only limited to a functor from `Py` to `func`, but also includes transformations between other mathematical structures.

        Let’s prove that **`length`** can be viewed as a functor. Specifically, we will demonstrate that `length` is a functor from the **category of list concatenation** to the **category of integer addition**.

        ### Category of List Concatenation

        First, let’s define the category of list concatenation:
        """
    )
    return


@app.cell
def _(Generic, a, dataclass):
    @dataclass
    class ListConcatenation(Generic[a]):
        value: list[a]

        @staticmethod
        def id() -> "ListConcatenation[a]":
            return ListConcatenation([])

        @staticmethod
        def compose(
            this: "ListConcatenation[a]", other: "ListConcatenation[a]"
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
        """
    )
    return


@app.cell
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

        #### 1. **Identity Law**:
        The identity law states that applying the functor to the identity element of one category should give the identity element of the other category.
        """
    )
    return


@app.cell
def _(IntAddition, ListConcatenation, length):
    length(ListConcatenation.id()) == IntAddition.id()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""This ensures that the length of an empty list (identity in the `ListConcatenation` category) is `0` (identity in the `IntAddition` category).""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        #### 2. **Composition Law**:
        The composition law states that the functor should preserve composition. Applying the functor to a composed element should be the same as composing the functor applied to the individual elements.
        """
    )
    return


@app.cell
def _(ListConcatenation):
    lista = ListConcatenation([1, 2])
    listb = ListConcatenation([3, 4])
    return lista, listb


@app.cell
def _(IntAddition, ListConcatenation, length, lista, listb):
    length(ListConcatenation.compose(lista, listb)) == IntAddition.compose(
        length(lista), length(listb)
    )
    return


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
        - [Haskellwiki. Category Theory](https://en.wikibooks.org/wiki/Haskell/Category_theory)
        - [Haskellforall. The Category Design Pattern](https://www.haskellforall.com/2012/08/the-category-design-pattern.html)
        - [Haskellforall. The Functor Design Pattern](https://www.haskellforall.com/2012/09/the-functor-design-pattern.html)

            /// attention | ATTENTION 
            The functor design pattern doesn't work at all if you aren't using categories in the first place. This is why you should structure your tools using the compositional category design pattern so that you can take advantage of functors to easily mix your tools together. 
            ///

        - [Haskellwiki. Functor](https://wiki.haskell.org/index.php?title=Functor)
        - [Haskellwiki. Typeclassopedia#Functor](https://wiki.haskell.org/index.php?title=Typeclassopedia#Functor)
        - [Haskellwiki. Typeclassopedia#Category](https://wiki.haskell.org/index.php?title=Typeclassopedia#Category)
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
    from typing import Callable, Generic, TypeVar
    return Callable, Generic, TypeVar, dataclass


@app.cell(hide_code=True)
def _(TypeVar):
    a = TypeVar("a")
    b = TypeVar("b")
    c = TypeVar("c")
    return a, b, c


if __name__ == "__main__":
    app.run()
