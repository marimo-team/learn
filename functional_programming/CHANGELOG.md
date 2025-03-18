# Changelog of the functional-programming course

## 2025-03-16

+ Use uppercased letters for `Generic` types, e.g. `A = TypeVar("A")`
+ Refactor the `Functor` class, changing `fmap` and utility methods to `classmethod`

    For example:

    ```python
    @dataclass
    class Wrapper(Functor, Generic[A]):
        value: A

        @classmethod
        def fmap(cls, f: Callable[[A], B], a: "Wrapper[A]") -> "Wrapper[B]":
            return Wrapper(f(a.value))

    >>> Wrapper.fmap(lambda x: x + 1, wrapper)
    Wrapper(value=2)
    ```

+ Move the `check_functor_law` method from `Functor` class to a standard function
- Rename `ListWrapper` to `List` for simplicity
- Remove the `Just` class
+ Rewrite proofs

## 2025-03-13

* `0.1.0` version of notebook `05_functors`

Thank [Akshay](https://github.com/akshayka) and [Haleshot](https://github.com/Haleshot) for reviewing

## 2025-03-11

* Demo version of notebook `05_functors.py`
