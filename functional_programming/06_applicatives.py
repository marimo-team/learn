# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
# ]
# ///

import marimo

__generated_with = "0.12.9"
app = marimo.App(app_title="Applicative programming with effects")


@app.cell(hide_code=True)
def _(mo) -> None:
    mo.md(
        r"""
        # Applicative programming with effects

        `Applicative Functor` encapsulates certain sorts of *effectful* computations in a functionally pure way, and encourages an *applicative* programming style.

        Applicative is a functor with application, providing operations to

        + embed pure expressions (`pure`), and
        + sequence computations and combine their results (`apply`).

        In this notebook, you will learn:

        1. How to view `Applicative` as multi-functor intuitively.
        2. How to use `lift` to simplify chaining application.
        3. How to bring *effects* to the functional pure world.
        4. How to view `Applicative` as a lax monoidal functor.
        5. How to use `Alternative` to amalgamate multiple computations into a single computation.

        /// details | Notebook metadata
            type: info

        version: 0.1.3 | last modified: 2025-04-16 | author: [métaboulie](https://github.com/metaboulie)<br/>
        reviewer: [Haleshot](https://github.com/Haleshot)

        ///
        """
    )


@app.cell(hide_code=True)
def _(mo) -> None:
    mo.md(
        r"""
        # The intuition: [Multifunctor](https://arxiv.org/pdf/2401.14286)

        ## Limitations of functor

        Recall that functors abstract the idea of mapping a function over each element of a structure.

        Suppose now that we wish to generalise this idea to allow functions with any number of arguments to be mapped, rather than being restricted to functions with a single argument. More precisely, suppose that we wish to define a hierarchy of `fmap` functions with the following types:

        ```haskell
        fmap0 :: a -> f a

        fmap1 :: (a -> b) -> f a -> f b

        fmap2 :: (a -> b -> c) -> f a -> f b -> f c

        fmap3 :: (a -> b -> c -> d) -> f a -> f b -> f c -> f d
        ```

        And we have to declare a special version of the functor class for each case.
        """
    )


@app.cell(hide_code=True)
def _(mo) -> None:
    mo.md(
        r"""
        ## Defining Multifunctor

        /// admonition
        we use prefix `f` rather than `ap` to indicate *Applicative Functor*
        ///

        As a result, we may want to define a single `Multifunctor` such that:

        1. Lift a regular n-argument function into the context of functors

            ```python
            # lift a regular 3-argument function `g`
            g: Callable[[A, B, C], D]
            # into the context of functors
            fg: Callable[[Functor[A], Functor[B], Functor[C]], Functor[D]]
            ```

        3. Apply it to n functor-wrapped values

            ```python
            # fa: Functor[A], fb: Functor[B], fc: Functor[C]
            fg(fa, fb, fc)
            ```

        5. Get a single functor-wrapped result

            ```python
            fd: Functor[D]
            ```

        We will define a function `lift` such that

        ```python
        fd = lift(g, fa, fb, fc)
        ```
        """
    )


@app.cell(hide_code=True)
def _(mo) -> None:
    mo.md(
        r"""
        ## Pure, apply and lift

        Traditionally, applicative functors are presented through two core operations:

        1. `pure`: embeds an object (value or function) into the applicative functor

            ```python
            # a -> F a
            pure: Callable[[A], Applicative[A]]
            # for example, if `a` is
            a: A
            # then we can have `fa` as
            fa: Applicative[A] = pure(a)
            # or if we have a regular function `g`
            g: Callable[[A], B]
            # then we can have `fg` as
            fg: Applicative[Callable[[A], B]] = pure(g)
            ```

        2. `apply`: applies a function inside an applicative functor to a value inside an applicative functor

            ```python
            # F (a -> b) -> F a -> F b
            apply: Callable[[Applicative[Callable[[A], B]], Applicative[A]], Applicative[B]]
            # and we can have
            fd = apply(apply(apply(fg, fa), fb), fc)
            ```


        As a result,

        ```python
        lift(g, fa, fb, fc) = apply(apply(apply(pure(g), fa), fb), fc)
        ```
        """
    )


@app.cell(hide_code=True)
def _(mo) -> None:
    mo.md(
        r"""
        /// admonition | How to use *Applicative* in the manner of *Multifunctor*

        1. Define `pure` and `apply` for an `Applicative` subclass

            - We can define them much easier compared with `lift`.

        2. Use the `lift` method

            - We can use it much more convenient compared with the combination of `pure` and `apply`.


        ///

        /// attention | You can suppress the chaining application of `apply` and `pure` as:

        ```python
        apply(pure(g), fa) -> lift(g, fa)
        apply(apply(pure(g), fa), fb) -> lift(g, fa, fb)
        apply(apply(apply(pure(g), fa), fb), fc) -> lift(g, fa, fb, fc)
        ```

        ///
        """
    )


@app.cell(hide_code=True)
def _(mo) -> None:
    mo.md(
        r"""
        ## Abstracting applicatives

        We can now provide an initial abstraction definition of applicatives:

        ```python
        @dataclass
        class Applicative[A](Functor, ABC):
            @classmethod
            @abstractmethod
            def pure(cls, a: A) -> "Applicative[A]":
                raise NotImplementedError("Subclasses must implement pure")

            @classmethod
            @abstractmethod
            def apply(
                cls, fg: "Applicative[Callable[[A], B]]", fa: "Applicative[A]"
            ) -> "Applicative[B]":
                raise NotImplementedError("Subclasses must implement apply")

            @classmethod
            def lift(cls, f: Callable, *args: "Applicative") -> "Applicative":
                curr = cls.pure(f)
                if not args:
                    return curr
                for arg in args:
                    curr = cls.apply(curr, arg)
                return curr
        ```

        /// attention | minimal implementation requirement

        - `pure`
        - `apply`
        ///
        """
    )


@app.cell(hide_code=True)
def _(mo) -> None:
    mo.md(r"""# Instances, laws and utility functions""")


@app.cell(hide_code=True)
def _(mo) -> None:
    mo.md(
        r"""
        ## Applicative instances

        When we are actually implementing an *Applicative* instance, we can keep in mind that `pure` and `apply` fundamentally:

        - embed an object (value or function) to the computational context
        - apply a function inside the computation context to a value inside the computational context
        """
    )


@app.cell(hide_code=True)
def _(mo) -> None:
    mo.md(
        r"""
        ### The Wrapper Applicative

        - `pure` should simply *wrap* an object, in the sense that:

            ```haskell
            Wrapper.pure(1) => Wrapper(value=1)
            ```

        - `apply` should apply a *wrapped* function to a *wrapped* value

        The implementation is:
        """
    )


@app.cell
def _(Applicative, dataclass):
    @dataclass
    class Wrapper[A](Applicative):
        value: A

        @classmethod
        def pure(cls, a: A) -> "Wrapper[A]":
            return cls(a)

        @classmethod
        def apply(
            cls, fg: "Wrapper[Callable[[A], B]]", fa: "Wrapper[A]"
        ) -> "Wrapper[B]":
            return cls(fg.value(fa.value))
    return (Wrapper,)


@app.cell(hide_code=True)
def _(mo) -> None:
    mo.md(r"""> try with Wrapper below""")


@app.cell
def _(Wrapper) -> None:
    Wrapper.lift(
        lambda a: lambda b: lambda c: a + b * c,
        Wrapper(1),
        Wrapper(2),
        Wrapper(3),
    )


@app.cell(hide_code=True)
def _(mo) -> None:
    mo.md(
        r"""
        ### The List Applicative

        - `pure` should wrap the object in a list, in the sense that:

            ```haskell
            List.pure(1) => List(value=[1])
            ```

        - `apply` should apply a list of functions to a list of values
            - you can think of this as cartesian product, concatenating the result of applying every function to every value

        The implementation is:
        """
    )


@app.cell
def _(Applicative, dataclass, product):
    @dataclass
    class List[A](Applicative):
        value: list[A]

        @classmethod
        def pure(cls, a: A) -> "List[A]":
            return cls([a])

        @classmethod
        def apply(cls, fg: "List[Callable[[A], B]]", fa: "List[A]") -> "List[B]":
            return cls([g(a) for g, a in product(fg.value, fa.value)])
    return (List,)


@app.cell(hide_code=True)
def _(mo) -> None:
    mo.md(r"""> try with List below""")


@app.cell
def _(List) -> None:
    List.apply(
        List([lambda a: a + 1, lambda a: a * 2]),
        List([1, 2]),
    )


@app.cell
def _(List) -> None:
    List.lift(lambda a: lambda b: a + b, List([1, 2]), List([3, 4, 5]))


@app.cell(hide_code=True)
def _(mo) -> None:
    mo.md(
        r"""
        ### The Maybe Applicative

        - `pure` should wrap the object in a Maybe, in the sense that:

            ```haskell
            Maybe.pure(1)    => "Just 1"
            Maybe.pure(None) => "Nothing"
            ```

        - `apply` should apply a function maybe exist to a value maybe exist
            - if the function is `None` or the value is `None`, simply returns `None`
            - else apply the function to the value and wrap the result in `Just`

        The implementation is:
        """
    )


@app.cell
def _(Applicative, dataclass):
    @dataclass
    class Maybe[A](Applicative):
        value: None | A

        @classmethod
        def pure(cls, a: A) -> "Maybe[A]":
            return cls(a)

        @classmethod
        def apply(
            cls, fg: "Maybe[Callable[[A], B]]", fa: "Maybe[A]"
        ) -> "Maybe[B]":
            if fg.value is None or fa.value is None:
                return cls(None)

            return cls(fg.value(fa.value))

        def __repr__(self):
            return "Nothing" if self.value is None else f"Just({self.value!r})"
    return (Maybe,)


@app.cell(hide_code=True)
def _(mo) -> None:
    mo.md(r"""> try with Maybe below""")


@app.cell
def _(Maybe) -> None:
    Maybe.lift(
        lambda a: lambda b: a + b,
        Maybe(1),
        Maybe(2),
    )


@app.cell
def _(Maybe) -> None:
    Maybe.lift(
        lambda a: lambda b: None,
        Maybe(1),
        Maybe(2),
    )


@app.cell(hide_code=True)
def _(mo) -> None:
    mo.md(
        r"""
        ### The Either Applicative

        - `pure` should wrap the object in `Right`, in the sense that:

            ```haskell
            Either.pure(1) => Right(1)
            ```

        - `apply` should apply a function that is either on Left or Right to a value that is either on Left or Right
            - if the function is `Left`, simply returns the `Left` of the function
            - else `fmap` the `Right` of the function to the value

        The implementation is:
        """
    )


@app.cell
def _(Applicative, B, Callable, Union, dataclass):
    @dataclass
    class Either[A](Applicative):
        left: A = None
        right: A = None

        def __post_init__(self):
            if (self.left is not None and self.right is not None) or (
                self.left is None and self.right is None
            ):
                msg = "Provide either the value of the left or the value of the right."
                raise TypeError(
                    msg
                )

        @classmethod
        def pure(cls, a: A) -> "Either[A]":
            return cls(right=a)

        @classmethod
        def apply(
            cls, fg: "Either[Callable[[A], B]]", fa: "Either[A]"
        ) -> "Either[B]":
            if fg.left is not None:
                return cls(left=fg.left)
            return cls.fmap(fg.right, fa)

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


@app.cell(hide_code=True)
def _(mo) -> None:
    mo.md(r"""> try with `Either` below""")


@app.cell
def _(Either) -> None:
    Either.apply(Either(left=TypeError("Parse Error")), Either(right=2))


@app.cell
def _(Either) -> None:
    Either.apply(
        Either(right=lambda x: x + 1), Either(left=TypeError("Parse Error"))
    )


@app.cell
def _(Either) -> None:
    Either.apply(Either(right=lambda x: x + 1), Either(right=1))


@app.cell(hide_code=True)
def _(mo) -> None:
    mo.md(
        r"""
        ## Collect the list of response with sequenceL

        One often wants to execute a list of commands and collect the list of their response, and we can define a function `sequenceL` for this

        /// admonition
        In a further notebook about `Traversable`, we will have a more generic `sequence` that execute a **sequence** of commands and collect the **sequence** of their response, which is not limited to `list`.
        ///

        ```python
        @classmethod
        def sequenceL(cls, fas: list["Applicative[A]"]) -> "Applicative[list[A]]":
            if not fas:
                return cls.pure([])

            return cls.apply(
                cls.fmap(lambda v: lambda vs: [v] + vs, fas[0]),
                cls.sequenceL(fas[1:]),
            )
        ```

        Let's try `sequenceL` with the instances.
        """
    )


@app.cell
def _(Wrapper) -> None:
    Wrapper.sequenceL([Wrapper(1), Wrapper(2), Wrapper(3)])


@app.cell(hide_code=True)
def _(mo) -> None:
    mo.md(
        r"""
        /// attention
        For the `Maybe` Applicative, the presence of any `Nothing` causes the entire computation to return Nothing.
        ///
        """
    )


@app.cell
def _(Maybe) -> None:
    Maybe.sequenceL([Maybe(1), Maybe(2), Maybe(None), Maybe(3)])


@app.cell(hide_code=True)
def _(mo) -> None:
    mo.md(r"""The result of `sequenceL` for `List Applicative`  is the Cartesian product of the input lists, yielding all possible ordered combinations of elements from each list.""")


@app.cell
def _(List) -> None:
    List.sequenceL([List([1, 2]), List([3]), List([5, 6, 7])])


@app.cell(hide_code=True)
def _(mo) -> None:
    mo.md(
        r"""
        ## Applicative laws

        /// admonition | id and compose

        Remember that

        - `id = lambda x: x`
        - `compose = lambda f: lambda g: lambda x: f(g(x))`

        ///

        Traditionally, there are four laws that `Applicative` instances should satisfy. In some sense, they are all concerned with making sure that `pure` deserves its name:

        - The identity law:
          ```python
          # fa: Applicative[A]
          apply(pure(id), fa) = fa
          ```
        - Homomorphism:
          ```python
          # a: A
          # g: Callable[[A], B]
          apply(pure(g), pure(a)) = pure(g(a))
          ```
          Intuitively, applying a non-effectful function to a non-effectful argument in an effectful context is the same as just applying the function to the argument and then injecting the result into the context with pure.
        - Interchange:
          ```python
          # a: A
          # fg: Applicative[Callable[[A], B]]
          apply(fg, pure(a)) = apply(pure(lambda g: g(a)), fg)
          ```
          Intuitively, this says that when evaluating the application of an effectful function to a pure argument, the order in which we evaluate the function and its argument doesn't matter.
        - Composition:
          ```python
          # fg: Applicative[Callable[[B], C]]
          # fh: Applicative[Callable[[A], B]]
          # fa: Applicative[A]
          apply(fg, apply(fh, fa)) = lift(compose, fg, fh, fa)
          ```
          This one is the trickiest law to gain intuition for. In some sense it is expressing a sort of associativity property of `apply`.

        We can add 4 helper functions to `Applicative` to check whether an instance respects the laws or not:

        ```python
        @dataclass
        class Applicative[A](Functor, ABC):

            @classmethod
            def check_identity(cls, fa: "Applicative[A]"):
                if cls.lift(id, fa) != fa:
                    raise ValueError("Instance violates identity law")
                return True

            @classmethod
            def check_homomorphism(cls, a: A, f: Callable[[A], B]):
                if cls.lift(f, cls.pure(a)) != cls.pure(f(a)):
                    raise ValueError("Instance violates homomorphism law")
                return True

            @classmethod
            def check_interchange(cls, a: A, fg: "Applicative[Callable[[A], B]]"):
                if cls.apply(fg, cls.pure(a)) != cls.lift(lambda g: g(a), fg):
                    raise ValueError("Instance violates interchange law")
                return True

            @classmethod
            def check_composition(
                cls,
                fg: "Applicative[Callable[[B], C]]",
                fh: "Applicative[Callable[[A], B]]",
                fa: "Applicative[A]",
            ):
                if cls.apply(fg, cls.apply(fh, fa)) != cls.lift(compose, fg, fh, fa):
                    raise ValueError("Instance violates composition law")
                return True
        ```

        > Try to validate applicative laws below
        """
    )


@app.cell
def _():
    id = lambda x: x
    compose = lambda f: lambda g: lambda x: f(g(x))
    const = lambda a: lambda _: a
    return compose, const, id


@app.cell
def _(List, Wrapper) -> None:
    print("Checking Wrapper")
    print(Wrapper.check_identity(Wrapper.pure(1)))
    print(Wrapper.check_homomorphism(1, lambda x: x + 1))
    print(Wrapper.check_interchange(1, Wrapper.pure(lambda x: x + 1)))
    print(
        Wrapper.check_composition(
            Wrapper.pure(lambda x: x * 2),
            Wrapper.pure(lambda x: x + 0.1),
            Wrapper.pure(1),
        )
    )

    print("\nChecking List")
    print(List.check_identity(List.pure(1)))
    print(List.check_homomorphism(1, lambda x: x + 1))
    print(List.check_interchange(1, List.pure(lambda x: x + 1)))
    print(
        List.check_composition(
            List.pure(lambda x: x * 2), List.pure(lambda x: x + 0.1), List.pure(1)
        )
    )


@app.cell(hide_code=True)
def _(mo) -> None:
    mo.md(
        r"""
        ## Utility functions

        /// attention | using `fmap`
        `fmap` is defined automatically using `pure` and `apply`, so you can use `fmap` with any `Applicative`
        ///

        ```python
        @dataclass
        class Applicative[A](Functor, ABC):
            @classmethod
            def skip(
                cls, fa: "Applicative[A]", fb: "Applicative[B]"
            ) -> "Applicative[B]":
                '''
                Sequences the effects of two Applicative computations,
                but discards the result of the first.
                '''
                return cls.apply(cls.const(fa, id), fb)

            @classmethod
            def keep(
                cls, fa: "Applicative[A]", fb: "Applicative[B]"
            ) -> "Applicative[B]":
                '''
                Sequences the effects of two Applicative computations,
                but discard the result of the second.
                '''
                return cls.lift(const, fa, fb)

            @classmethod
            def revapp(
                cls, fa: "Applicative[A]", fg: "Applicative[Callable[[A], [B]]]"
            ) -> "Applicative[B]":
                '''
                The first computation produces values which are provided
                as input to the function(s) produced by the second computation.
                '''
                return cls.lift(lambda a: lambda f: f(a), fa, fg)
        ```

        - `skip` sequences the effects of two Applicative computations, but **discards the result of the first**. For example, if `m1` and `m2` are instances of type `Maybe[Int]`, then `Maybe.skip(m1, m2)` is `Nothing` whenever either `m1` or `m2` is `Nothing`; but if not, it will have the same value as `m2`.
        - Likewise, `keep` sequences the effects of two computations, but **keeps only the result of the first**.
        - `revapp` is similar to `apply`, but where the first computation produces value(s) which are provided as input to the function(s) produced by the second computation.
        """
    )


@app.cell(hide_code=True)
def _(mo) -> None:
    mo.md(
        r"""
        /// admonition | Exercise
        Try to use utility functions with different instances
        ///
        """
    )


@app.cell(hide_code=True)
def _(mo) -> None:
    mo.md(
        r"""
        # Formal implementation of Applicative

        Now, we can give the formal implementation of `Applicative`
        """
    )


@app.cell
def _(
    ABC,
    B,
    Callable,
    Functor,
    abstractmethod,
    compose,
    const,
    dataclass,
    id,
):
    @dataclass
    class Applicative[A](Functor, ABC):
        @classmethod
        @abstractmethod
        def pure(cls, a: A) -> "Applicative[A]":
            """Lift a value into the Structure."""
            msg = "Subclasses must implement pure"
            raise NotImplementedError(msg)

        @classmethod
        @abstractmethod
        def apply(
            cls, fg: "Applicative[Callable[[A], B]]", fa: "Applicative[A]"
        ) -> "Applicative[B]":
            """Sequential application."""
            msg = "Subclasses must implement apply"
            raise NotImplementedError(msg)

        @classmethod
        def lift(cls, f: Callable, *args: "Applicative") -> "Applicative":
            """Lift a function of arbitrary arity to work with values in applicative context."""
            curr = cls.pure(f)

            if not args:
                return curr

            for arg in args:
                curr = cls.apply(curr, arg)

            return curr

        @classmethod
        def fmap(
            cls, f: Callable[[A], B], fa: "Applicative[A]"
        ) -> "Applicative[B]":
            return cls.lift(f, fa)

        @classmethod
        def sequenceL(cls, fas: list["Applicative[A]"]) -> "Applicative[list[A]]":
            """
            Execute a list of commands and collect the list of their response.
            """
            if not fas:
                return cls.pure([])

            return cls.apply(
                cls.fmap(lambda v: lambda vs: [v, *vs], fas[0]),
                cls.sequenceL(fas[1:]),
            )

        @classmethod
        def skip(
            cls, fa: "Applicative[A]", fb: "Applicative[B]"
        ) -> "Applicative[B]":
            """
            Sequences the effects of two Applicative computations,
            but discards the result of the first.
            """
            return cls.apply(cls.const(fa, id), fb)

        @classmethod
        def keep(
            cls, fa: "Applicative[A]", fb: "Applicative[B]"
        ) -> "Applicative[B]":
            """
            Sequences the effects of two Applicative computations,
            but discard the result of the second.
            """
            return cls.lift(const, fa, fb)

        @classmethod
        def revapp(
            cls, fa: "Applicative[A]", fg: "Applicative[Callable[[A], [B]]]"
        ) -> "Applicative[B]":
            """
            The first computation produces values which are provided
            as input to the function(s) produced by the second computation.
            """
            return cls.lift(lambda a: lambda f: f(a), fa, fg)

        @classmethod
        def check_identity(cls, fa: "Applicative[A]") -> bool:
            if cls.lift(id, fa) != fa:
                msg = "Instance violates identity law"
                raise ValueError(msg)
            return True

        @classmethod
        def check_homomorphism(cls, a: A, f: Callable[[A], B]) -> bool:
            if cls.lift(f, cls.pure(a)) != cls.pure(f(a)):
                msg = "Instance violates homomorphism law"
                raise ValueError(msg)
            return True

        @classmethod
        def check_interchange(cls, a: A, fg: "Applicative[Callable[[A], B]]") -> bool:
            if cls.apply(fg, cls.pure(a)) != cls.lift(lambda g: g(a), fg):
                msg = "Instance violates interchange law"
                raise ValueError(msg)
            return True

        @classmethod
        def check_composition(
            cls,
            fg: "Applicative[Callable[[B], C]]",
            fh: "Applicative[Callable[[A], B]]",
            fa: "Applicative[A]",
        ) -> bool:
            if cls.apply(fg, cls.apply(fh, fa)) != cls.lift(compose, fg, fh, fa):
                msg = "Instance violates composition law"
                raise ValueError(msg)
            return True
    return (Applicative,)


@app.cell(hide_code=True)
def _(mo) -> None:
    mo.md(
        r"""
        # Effectful programming

        Our original motivation for applicatives was the desire to generalise the idea of mapping to functions with multiple arguments. This is a valid interpretation of the concept of applicatives, but from the three instances we have seen it becomes clear that there is also another, more abstract view.

         The arguments are no longer just plain values but may also have effects, such as the possibility of failure, having many ways to succeed, or performing input/output actions. In this manner, applicative functors can also be viewed as abstracting the idea of **applying pure functions to effectful arguments**, with the precise form of effects that are permitted depending on the nature of the underlying functor.
        """
    )


@app.cell(hide_code=True)
def _(mo) -> None:
    mo.md(
        r"""
        ## The IO Applicative

        We will try to define an `IO` applicative here.

        As before, we first abstract how `pure` and `apply` should function.

        - `pure` should wrap the object in an IO action, and make the object *callable* if it's not because we want to perform the action later:

            ```haskell
            IO.pure(1) => IO(effect=lambda: 1)
            IO.pure(f) => IO(effect=f)
            ```

        - `apply` should perform an action that produces a value, then apply the function with the value

        The implementation is:
        """
    )


@app.cell
def _(Applicative, Callable, dataclass):
    @dataclass
    class IO(Applicative):
        effect: Callable

        def __call__(self):
            return self.effect()

        @classmethod
        def pure(cls, a):
            return cls(a) if isinstance(a, Callable) else IO(lambda: a)

        @classmethod
        def apply(cls, fg, fa):
            return cls.pure(fg.effect(fa.effect()))
    return (IO,)


@app.cell(hide_code=True)
def _(mo) -> None:
    mo.md(r"""For example, a function that reads a given number of lines from the keyboard can be defined in applicative style as follows:""")


@app.cell
def _(IO):
    def get_chars(n: int = 3):
        return IO.sequenceL([
            IO.pure(input(f"input the {i}th str")) for i in range(1, n + 1)
        ])
    return (get_chars,)


@app.cell
def _() -> None:
    # get_chars()()
    return


@app.cell(hide_code=True)
def _(mo) -> None:
    mo.md(r"""# From the perspective of category theory""")


@app.cell(hide_code=True)
def _(mo) -> None:
    mo.md(
        r"""
        ## Lax Monoidal Functor

        An alternative, equivalent formulation of `Applicative` is given by
        """
    )


@app.cell
def _(ABC, Functor, abstractmethod, dataclass):
    @dataclass
    class Monoidal[A](Functor, ABC):
        @classmethod
        @abstractmethod
        def unit(cls) -> "Monoidal[Tuple[()]]":
            pass

        @classmethod
        @abstractmethod
        def tensor(
            cls, this: "Monoidal[A]", other: "Monoidal[B]"
        ) -> "Monoidal[Tuple[A, B]]":
            pass
    return (Monoidal,)


@app.cell(hide_code=True)
def _(mo) -> None:
    mo.md(
        r"""
        Intuitively, this states that a *monoidal functor* is one which has some sort of "default shape" and which supports some sort of "combining" operation.

        - `unit` provides the identity element
        - `tensor` combines two contexts into a product context

        More technically, the idea is that `monoidal functor` preserves the "monoidal structure" given by the pairing constructor `(,)` and unit type `()`.
        """
    )


@app.cell(hide_code=True)
def _(mo) -> None:
    mo.md(
        r"""
        Furthermore, to deserve the name "monoidal", instances of Monoidal ought to satisfy the following laws, which seem much more straightforward than the traditional Applicative laws:

        - Left identity

            `tensor(unit, v) ≅ v`

        - Right identity

            `tensor(u, unit) ≅ u`

        - Associativity

            `tensor(u, tensor(v, w)) ≅ tensor(tensor(u, v), w)`
        """
    )


@app.cell(hide_code=True)
def _(mo) -> None:
    mo.md(
        r"""
        /// admonition | ≅ indicates isomorphism

        `≅` refers to *isomorphism* rather than equality.

        In particular we consider `(x, ()) ≅ x ≅ ((), x)` and `((x, y), z) ≅ (x, (y, z))`

        ///
        """
    )


@app.cell(hide_code=True)
def _(mo) -> None:
    mo.md(
        r"""
        ## Mutual definability of Monoidal and Applicative

        We can implement `pure` and `apply` in terms of `unit` and `tensor`, and vice versa.

        ```python
        pure(a) = fmap((lambda _: a), unit)
        apply(fg, fa) = fmap((lambda pair: pair[0](pair[1])), tensor(fg, fa))
        ```

        ```python
        unit() = pure(())
        tensor(fa, fb) = lift(lambda fa: lambda fb: (fa, fb), fa, fb)
        ```
        """
    )


@app.cell(hide_code=True)
def _(mo) -> None:
    mo.md(
        r"""
        ## Instance: ListMonoidal

        - `unit` should simply return a empty tuple wrapper in a list

            ```haskell
            ListMonoidal.unit() => [()]
            ```

        - `tensor` should return the *cartesian product* of the items of 2 ListMonoidal instances

        The implementation is:
        """
    )


@app.cell
def _(B, Callable, Monoidal, dataclass, product):
    @dataclass
    class ListMonoidal[A](Monoidal):
        items: list[A]

        @classmethod
        def unit(cls) -> "ListMonoidal[Tuple[()]]":
            return cls([()])

        @classmethod
        def tensor(
            cls, this: "ListMonoidal[A]", other: "ListMonoidal[B]"
        ) -> "ListMonoidal[Tuple[A, B]]":
            return cls(list(product(this.items, other.items)))

        @classmethod
        def fmap(
            cls, f: Callable[[A], B], ma: "ListMonoidal[A]"
        ) -> "ListMonoidal[B]":
            return cls([f(a) for a in ma.items])
    return (ListMonoidal,)


@app.cell(hide_code=True)
def _(mo) -> None:
    mo.md(r"""> try with `ListMonoidal` below""")


@app.cell
def _(ListMonoidal):
    xs = ListMonoidal([1, 2])
    ys = ListMonoidal(["a", "b"])
    ListMonoidal.tensor(xs, ys)
    return xs, ys


@app.cell(hide_code=True)
def _(mo) -> None:
    mo.md(r"""and we can prove that `tensor(fa, fb) = lift(lambda fa: lambda fb: (fa, fb), fa, fb)`:""")


@app.cell
def _(List, xs, ys) -> None:
    List.lift(lambda fa: lambda fb: (fa, fb), List(xs.items), List(ys.items))


@app.cell(hide_code=True)
def _(ABC, B, Callable, abstractmethod, dataclass):
    @dataclass
    class Functor[A](ABC):
        @classmethod
        @abstractmethod
        def fmap(cls, f: Callable[[A], B], a: "Functor[A]") -> "Functor[B]":
            msg = "Subclasses must implement fmap"
            raise NotImplementedError(msg)

        @classmethod
        def const(cls, a: "Functor[A]", b: B) -> "Functor[B]":
            return cls.fmap(lambda _: b, a)

        @classmethod
        def void(cls, a: "Functor[A]") -> "Functor[None]":
            return cls.const_fmap(a, None)
    return (Functor,)


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _():
    from abc import ABC, abstractmethod
    from collections.abc import Callable
    from dataclasses import dataclass
    from typing import TypeVar, Union
    return ABC, Callable, TypeVar, Union, abstractmethod, dataclass


@app.cell(hide_code=True)
def _():
    from itertools import product
    return (product,)


@app.cell(hide_code=True)
def _(TypeVar):
    A = TypeVar("A")
    B = TypeVar("B")
    C = TypeVar("C")
    return A, B, C


@app.cell(hide_code=True)
def _(mo) -> None:
    mo.md(
        r"""
        # From Applicative to Alternative

        ## Abstracting Alternative

        In our studies so far, we saw that both `Maybe` and `List` can represent computations with a varying number of results.

        We use `Maybe` to indicate a computation can fail somehow and `List` for computations that can have many possible results. In both of these cases, one useful operation is amalgamating all possible results from multiple computations into a single computation.

        `Alternative` formalizes computations that support:

        - **Failure** (empty result)
        - **Choice** (combination of results)
        - **Repetition** (multiple results)

        It extends `Applicative` with monoidal structure, where:

        ```python
        @dataclass
        class Alternative[A](Applicative, ABC):
            @classmethod
            @abstractmethod
            def empty(cls) -> "Alternative[A]":
                '''Identity element for alternative computations'''

            @classmethod
            @abstractmethod
            def alt(
                cls, fa: "Alternative[A]", fb: "Alternative[A]"
            ) -> "Alternative[A]":
                '''Binary operation combining computations'''
        ```

        - `empty` is the identity element (e.g., `Maybe(None)`, `List([])`)
        - `alt` is a combination operator (e.g., `Maybe` fallback, list concatenation)

        `empty` and `alt` should satisfy the following **laws**:

        ```python
        # Left identity
        alt(empty, fa) == fa
        # Right identity
        alt(fa, empty) == fa
        # Associativity
        alt(fa, alt(fb, fc)) == alt(alt(fa, fb), fc)
        ```

        /// admonition
        Actually, `Alternative` is a *monoid* on `Applicative Functors`. We will talk about *monoid* and review these laws in the next notebook about `Monads`.
        ///

        /// attention | minimal implementation requirement
        - `empty`
        - `alt`
        ///
        """
    )


@app.cell(hide_code=True)
def _(mo) -> None:
    mo.md(
        r"""
        ## Instances of Alternative

        ### The Maybe Alternative

        - `empty`: the identity element of `Maybe` is `Maybe(None)`
        - `alt`: return the first element if it's not `None`, else return the second element
        """
    )


@app.cell
def _(Alternative, Maybe, dataclass):
    @dataclass
    class AltMaybe[A](Maybe, Alternative):
        @classmethod
        def empty(cls) -> "AltMaybe[A]":
            return cls(None)

        @classmethod
        def alt(cls, fa: "AltMaybe[A]", fb: "AltMaybe[A]") -> "AltMaybe[A]":
            if fa.value is not None:
                return cls(fa.value)
            return cls(fb.value)

        def __repr__(self):
            return "Nothing" if self.value is None else f"Just({self.value!r})"
    return (AltMaybe,)


@app.cell
def _(AltMaybe) -> None:
    print(AltMaybe.empty())
    print(AltMaybe.alt(AltMaybe(None), AltMaybe(1)))
    print(AltMaybe.alt(AltMaybe(None), AltMaybe(None)))
    print(AltMaybe.alt(AltMaybe(1), AltMaybe(None)))
    print(AltMaybe.alt(AltMaybe(1), AltMaybe(2)))


@app.cell
def _(AltMaybe) -> None:
    print(AltMaybe.check_left_identity(AltMaybe(1)))
    print(AltMaybe.check_right_identity(AltMaybe(1)))
    print(AltMaybe.check_associativity(AltMaybe(1), AltMaybe(2), AltMaybe(None)))


@app.cell(hide_code=True)
def _(mo) -> None:
    mo.md(
        r"""
        ### The List Alternative

        - `empty`: the identity element of `List` is `List([])`
        - `alt`: return the concatenation of 2 input lists
        """
    )


@app.cell
def _(Alternative, List, dataclass):
    @dataclass
    class AltList[A](List, Alternative):
        @classmethod
        def empty(cls) -> "AltList[A]":
            return cls([])

        @classmethod
        def alt(cls, fa: "AltList[A]", fb: "AltList[A]") -> "AltList[A]":
            return cls(fa.value + fb.value)
    return (AltList,)


@app.cell
def _(AltList) -> None:
    print(AltList.empty())
    print(AltList.alt(AltList([1, 2, 3]), AltList([4, 5])))


@app.cell
def _(AltList) -> None:
    AltList([1])


@app.cell
def _(AltList) -> None:
    AltList([1])


@app.cell
def _(AltList) -> None:
    print(AltList.check_left_identity(AltList([1, 2, 3])))
    print(AltList.check_right_identity(AltList([1, 2, 3])))
    print(
        AltList.check_associativity(
            AltList([1, 2]), AltList([3, 4, 5]), AltList([6])
        )
    )


@app.cell(hide_code=True)
def _(mo) -> None:
    mo.md(
        r"""
        ## some and many


        /// admonition | This section mainly refers to

        - https://stackoverflow.com/questions/7671009/some-and-many-functions-from-the-alternative-type-class/7681283#7681283

        ///

        First let's have a look at the implementation of `some` and `many`:

        ```python
        @classmethod
        def some(cls, fa: "Alternative[A]") -> "Alternative[list[A]]":
            # Short-circuit if input is empty
            if fa == cls.empty():
                return cls.empty()

            return cls.apply(
                cls.fmap(lambda a: lambda b: [a] + b, fa), cls.many(fa)
            )

        @classmethod
        def many(cls, fa: "Alternative[A]") -> "Alternative[list[A]]":
            # Directly return empty list if input is empty
            if fa == cls.empty():
                return cls.pure([])

            return cls.alt(cls.some(fa), cls.pure([]))
        ```

        So `some f` runs `f` once, then *many* times, and conses the results. `many f` runs f *some* times, or *alternatively* just returns the empty list.

        The idea is that they both run `f` as often as possible until it **fails**, collecting the results in a list. The difference is that `some f` immediately fails if `f` fails, while `many f` will still succeed and *return* the empty list in such a case. But what all this exactly means depends on how `alt` is defined.

        Let's see what it does for the instances `AltMaybe` and `AltList`.
        """
    )


@app.cell(hide_code=True)
def _(mo) -> None:
    mo.md(r"""For `AltMaybe`. `None` means failure, so some `None` fails as well and evaluates to `None` while many `None` succeeds and evaluates to `Just []`. Both `some (Just ())` and `many (Just ())` never return, because `Just ()` never fails.""")


@app.cell
def _(AltMaybe) -> None:
    print(AltMaybe.some(AltMaybe.empty()))
    print(AltMaybe.many(AltMaybe.empty()))


@app.cell(hide_code=True)
def _(mo) -> None:
    mo.md(r"""For `AltList`, `[]` means failure, so `some []` evaluates to `[]` (no answers) while `many []` evaluates to `[[]]` (there's one answer and it is the empty list). Again `some [()]` and `many [()]` don't return.""")


@app.cell
def _(AltList) -> None:
    print(AltList.some(AltList.empty()))
    print(AltList.many(AltList.empty()))


@app.cell(hide_code=True)
def _(mo) -> None:
    mo.md(r"""## Formal implementation of Alternative""")


@app.cell
def _(ABC, Applicative, abstractmethod, dataclass):
    @dataclass
    class Alternative[A](Applicative, ABC):
        """A monoid on applicative functors."""

        @classmethod
        @abstractmethod
        def empty(cls) -> "Alternative[A]":
            msg = "Subclasses must implement empty"
            raise NotImplementedError(msg)

        @classmethod
        @abstractmethod
        def alt(
            cls, fa: "Alternative[A]", fb: "Alternative[A]"
        ) -> "Alternative[A]":
            msg = "Subclasses must implement alt"
            raise NotImplementedError(msg)

        @classmethod
        def some(cls, fa: "Alternative[A]") -> "Alternative[list[A]]":
            # Short-circuit if input is empty
            if fa == cls.empty():
                return cls.empty()

            return cls.apply(
                cls.fmap(lambda a: lambda b: [a, *b], fa), cls.many(fa)
            )

        @classmethod
        def many(cls, fa: "Alternative[A]") -> "Alternative[list[A]]":
            # Directly return empty list if input is empty
            if fa == cls.empty():
                return cls.pure([])

            return cls.alt(cls.some(fa), cls.pure([]))

        @classmethod
        def check_left_identity(cls, fa: "Alternative[A]") -> bool:
            return cls.alt(cls.empty(), fa) == fa

        @classmethod
        def check_right_identity(cls, fa: "Alternative[A]") -> bool:
            return cls.alt(fa, cls.empty()) == fa

        @classmethod
        def check_associativity(
            cls, fa: "Alternative[A]", fb: "Alternative[A]", fc: "Alternative[A]"
        ) -> bool:
            return cls.alt(fa, cls.alt(fb, fc)) == cls.alt(cls.alt(fa, fb), fc)
    return (Alternative,)


@app.cell(hide_code=True)
def _(mo) -> None:
    mo.md(
        r"""
        /// admonition

        We will explore more about `Alternative` in a future notebooks about [Monadic Parsing](https://www.cambridge.org/core/journals/journal-of-functional-programming/article/monadic-parsing-in-haskell/E557DFCCE00E0D4B6ED02F3FB0466093)

        ///
        """
    )


@app.cell(hide_code=True)
def _(mo) -> None:
    mo.md(
        r"""
        # Further reading

        Notice that these reading sources are optional and non-trivial

        - [Applicaive Programming with Effects](https://www.staff.city.ac.uk/~ross/papers/Applicative.html)
        - [Equivalence of Applicative Functors and
        Multifunctors](https://arxiv.org/pdf/2401.14286)
        - [Applicative functor](https://wiki.haskell.org/index.php?title=Applicative_functor)
        - [Control.Applicative](https://hackage.haskell.org/package/base-4.21.0.0/docs/Control-Applicative.html#t:Applicative)
        - [Typeclassopedia#Applicative](https://wiki.haskell.org/index.php?title=Typeclassopedia#Applicative)
        - [Notions of computation as monoids](https://www.cambridge.org/core/journals/journal-of-functional-programming/article/notions-of-computation-as-monoids/70019FC0F2384270E9F41B9719042528)
        - [Free Applicative Functors](https://arxiv.org/abs/1403.0749)
        - [The basics of applicative functors, put to practical work](http://www.serpentine.com/blog/2008/02/06/the-basics-of-applicative-functors-put-to-practical-work/)
        - [Abstracting with Applicatives](http://comonad.com/reader/2012/abstracting-with-applicatives/)
        - [Static analysis with Applicatives](https://gergo.erdi.hu/blog/2012-12-01-static_analysis_with_applicatives/)
        - [Explaining Applicative functor in categorical terms - monoidal functors](https://cstheory.stackexchange.com/questions/12412/explaining-applicative-functor-in-categorical-terms-monoidal-functors)
        - [Applicative, A Strong Lax Monoidal Functor](https://beuke.org/applicative/)
        - [Applicative Functors](https://bartoszmilewski.com/2017/02/06/applicative-functors/)
        """
    )


if __name__ == "__main__":
    app.run()
