# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
# ]
# ///

import marimo

__generated_with = "0.12.0"
app = marimo.App(app_title="Applicative programming with effects")

@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Applicative programming with effects

        `Applicative Functor` encapsulates certain sorts of *effectful* computations in a functionally pure way, and encourages an *applicative* programming style.

        Applicative is a functor with application, providing operations to

        + embed pure expressions (`pure`), and
        + sequence computations and combine their results (`apply`).

        In this notebook, you will learn:

        1. How to view `applicative` as multi-functor.
        2. How to use `lift` to simplify chaining application.
        3. How to bring *effects* to the functional pure world.
        4. How to view `applicative` as lax monoidal functor

        /// details | Notebook metadata
            type: info

        version: 0.1.0 | last modified: 2025-04-02 | author: [métaboulie](https://github.com/metaboulie)<br/>

        ///

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
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
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Defining multifunctor

        As a result, we may want to define a single `Multifunctor` such that:

        1. Lift a regular n-argument function into the context of functors

            ```python
            # we use prefix `f` here to indicate `Functor`
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
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Pure, apply and lift

        Traditionally, applicative functors are presented through two core operations:

        1. `pure`: embeds an object (value or function) into the functor

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

        2. `apply`: applies a function inside a functor to a value inside a functor

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
    return


@app.cell(hide_code=True)
def _(mo):
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
        apply(pure(f), fa) -> lift(f, fa)
        apply(apply(pure(f), fa), fb) -> lift(f, fa, fb)
        apply(apply(apply(pure(f), fa), fb), fc) -> lift(f, fa, fb, fc)
        ```

        ///
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
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
                return NotImplementedError

            @classmethod
            @abstractmethod
            def apply(
                cls, fg: "Applicative[Callable[[A], B]]", fa: "Applicative[A]"
            ) -> "Applicative[B]":
                return NotImplementedError

            @classmethod
            def lift(cls, f: Callable, *args: "Applicative") -> "Applicative":
                curr = cls.pure(f)
                if not args:
                    return curr
                for arg in args:
                    curr = cls.apply(curr, arg)
                return curr
        ```

        /// attention | minimal implmentation requirement

        - `pure`
        - `apply`
        ///
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Instances, laws and utility functions""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Applicative instances

        When we are actually implementing an *Applicative* instance, we can keep in mind that `pure` and `apply` are fundamentally:

        - embed an object (value or function) to the computational context
        - apply a function inside the computation context to a value inside the computational context
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Wrapper

        - `pure` should simply *wrap* an object, in the sense that: 

            ```haskell
            Wrapper.pure(1) => Wrapper(value=1)
            ```

        - `apply` should apply a *wrapped* function to a *wrapper* value

        The implementation is: 
        """
    )
    return


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
def _(mo):
    mo.md(r"""> try with Wrapper below""")
    return


@app.cell
def _(Wrapper):
    Wrapper.lift(
        lambda a: lambda b: lambda c: a + b * c,
        Wrapper(1),
        Wrapper(2),
        Wrapper(3),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### List

        - `pure` should wrap the object in a list, in the sense that:

            ```haskell
            List.pure(1) => List(value=[1])
            ```

        - `apply` should apply a list of functions to a list of values
            - you can think of this as cartesian product, concating the result of applying every function to every value

        The implementation is:
        """
    )
    return


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
def _(mo):
    mo.md(r"""> try with List below""")
    return


@app.cell
def _(List):
    List.apply(
        List([lambda a: a + 1, lambda a: a * 2]),
        List([1, 2]),
    )
    return


@app.cell
def _(List):
    List.lift(lambda a: lambda b: a + b, List([1, 2]), List([3, 4, 5]))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Maybe

        - `pure` should wrap the object in a Maybe, in the sense that:

            ```haskell
            Maybe.pure(1)    => "Just 1"
            Maybe.pure(None) => "Nothing"
            ```

        - `apply` should apply a function maybe exist to a value maybe exist
            - if the function is `None`, apply returns `None`
            - else apply the function to the value and wrap the result in `Just`

        The implementation is: 
        """
    )
    return


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
            if fg.value is None:
                return cls(None)
            return cls(fg.value(fa.value))

        def __repr__(self):
            return "Nothing" if self.value is None else repr(f"Just {self.value}")
    return (Maybe,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""> try with Maybe below""")
    return


@app.cell
def _(Maybe):
    Maybe.lift(
        lambda a: lambda b: a + b,
        Maybe(1),
        Maybe(2),
    )
    return


@app.cell
def _(Maybe):
    Maybe.lift(
        lambda a: lambda b: None,
        Maybe(1),
        Maybe(2),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Applicative laws

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
          This one is the trickiest law to gain intuition for. In some sense it is expressing a sort of associativity property of (<*>).

        /// admonition | id and compose

        Remember that

        - id = lambda x: x
        - compose = lambda f: lambda g: lambda x: f(g(x))

        ///

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
    return


@app.cell
def _():
    id = lambda x: x
    compose = lambda f: lambda g: lambda x: f(g(x))
    const = lambda a: lambda _: a
    return compose, const, id


@app.cell
def _(List, Wrapper):
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
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Utility functions

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

        We can have a sense of how `skip` and `keep` work by trying out sequencing the effects of `Maybe`, where one computation is `None`
        """
    )
    return


@app.cell
def _(Maybe):
    print("Maybe.skip")
    print(Maybe.skip(Maybe(1), Maybe(None)))
    print(Maybe.skip(Maybe(None), Maybe(1)))

    print("\nMaybe.keep")
    print(Maybe.keep(Maybe(1), Maybe(None)))
    print(Maybe.keep(Maybe(None), Maybe(1)))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        /// admonition | exercise
        Try to use utility functions with different instances
        ///
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Formal implementation of Applicative

        Now, we can give the formal implementation of `Applicative`
        """
    )
    return


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
            return NotImplementedError

        @classmethod
        @abstractmethod
        def apply(
            cls, fg: "Applicative[Callable[[A], B]]", fa: "Applicative[A]"
        ) -> "Applicative[B]":
            """Sequential application."""
            return NotImplementedError

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
    return (Applicative,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Effectful Programming

        Our original motivation for applicatives was the desire the generalise the idea of mapping to functions with multiple arguments. This is a valid interpretation of the concept of applicatives, but from the three instances we have seen it becomes clear that there is also another, more abstract view.

         the arguments are no longer just plain values but may also have effects, such as the possibility of failure, having many ways to succeed, or performing input/output actions. In this manner, applicative functors can also be viewed as abstracting the idea of applying pure functions to effectful arguments, with the precise form of effects that are permitted depending on the nature of the underlying functor.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
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

        - `apply` should perform an action that produces a function and perform an action that produces a value, then call the function with the value

        The implementation is: 
        """
    )
    return


@app.cell
def _(Applicative, Callable, dataclass):
    @dataclass
    class IO(Applicative):
        effect: Callable

        def __call__(self):
            return self.effect()

        @classmethod
        def pure(cls, a):
            """Lift a value into the IO context"""
            return cls(a) if isinstance(a, Callable) else IO(lambda: a)

        @classmethod
        def apply(cls, f, a):
            """Applicative apply implementation"""
            return cls.pure(f()(a()))
    return (IO,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""For example, a function that reads a given number of lines from the keyboard can be defined in applicative style as follows:""")
    return


@app.cell
def _(IO):
    def get_chars(n: int = 3):
        if n <= 0:
            return ""
        return IO.lift(
            lambda: lambda s1: lambda: lambda s2: s1 + "\n" + s2,
            IO.pure(input(f"input line")),
            IO.pure(get_chars(n - 1)),
        )
    return (get_chars,)


@app.cell
def _():
    # print(get_chars()())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Collect the sequence of response with sequenceL""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        One often wants to execute a sequence of commands and collect the sequence of their response, and we can define a function `sequenceL` for this

        /// admonition
        sequenceL actually means that 

        > execute a list of commands and collect the list of their response
        ///
        """
    )
    return


@app.cell
def _(A, Applicative):
    def sequenceL(actions: list[Applicative[A]], ap) -> Applicative[list[A]]:
        if not actions:
            return ap.pure([])

        return ap.lift(
            lambda: lambda l1: lambda: lambda l2: list(l1) + list(l2),
            actions[0],
            sequenceL(actions[1:], ap),
        )
    return (sequenceL,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        This function transforms a list of applicative actions into a single such action that returns a list of result values, and captures a common pattern of applicative programming.

        And we can rewrite the `get_chars` more concisely: 
        """
    )
    return


@app.cell
def _(IO, sequenceL):
    def get_chars_sequenceL(n: int = 3):
        return sequenceL(
            [IO.pure(input(f"input the {i}nd str") for i in range(1, n + 1))], IO
        )
    return (get_chars_sequenceL,)


@app.cell
def _():
    # print(get_chars_sequenceL()())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# From the perspective of category theory""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Lax Monoidal Functor

        An alternative, equivalent formulation of `Applicative` is given by 
        """
    )
    return


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


@app.cell
def _(mo):
    mo.md(r"""fmap g fa = fmap (g . snd) (unit ** fa)""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Intuitively, this states that a *monoidal functor* is one which has some sort of "default shape" and which supports some sort of "combining" operation. 

        - `unit` provides the identity element
        - `tensor` combines two contexts into a product context

        More technically, the idea is that `monoidal functor` preserves the "monoidal structure" given by the pairing constructor `(,)` and unit type `()`. 
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
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
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        /// admonition | ≅ indicates isomorphism

        `≅` refers to *isomorphism* rather than equality.

        In particular we consider `(x, ()) ≅ x ≅ ((), x)` and `((x, y), z) ≅ (x, (y, z))`

        ///
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Mutual Definability of Monoidal and Applicative

        We can implement `pure` and `apply` in terms of `unit` and `tensor`, and vice versa.

        ```python
        pure(a) = fmap((lambda _: a), unit)
        apply(mf, mx) = fmap((lambda pair: pair[0](pair[1])), tensor(mf, mx))
        ```

        ```python
        unit() = pure(())
        tensor(fa, fb) = lift( ,fa, fb)
        ```
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Instance: ListMonoidal

        - `unit` should simply return a emtpy tuple wrapper in a list

            ```haskell
            ListMonoidal.unit() => [()]
            ```

        - `tensor` should return the *cartesian product* of the items of 2 ListMonoidal instances

        The implementation is:
        """
    )
    return


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

        def __repr__(self):
            return repr(self.items)
    return (ListMonoidal,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""> try with Maybe below""")
    return


@app.cell
def _(ListMonoidal):
    xs = ListMonoidal([1, 2])
    ys = ListMonoidal(["a", "b"])
    ListMonoidal.tensor(xs, ys)
    return xs, ys


@app.cell(hide_code=True)
def _(ABC, B, Callable, abstractmethod, dataclass):
    @dataclass
    class Functor[A](ABC):
        @classmethod
        @abstractmethod
        def fmap(cls, f: Callable[[A], B], a: "Functor[A]") -> "Functor[B]":
            return NotImplementedError

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
    from dataclasses import dataclass
    from abc import ABC, abstractmethod
    from typing import TypeVar, Union
    from collections.abc import Callable
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
def _(mo):
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
    return


if __name__ == "__main__":
    app.run()
