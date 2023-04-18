"""
Code for generalized cumulant decomposition
"""
# %%
from pprint import pprint
from typing import cast, TypeVar, Sequence, Iterable, Optional
from collections import defaultdict
from functools import lru_cache
from itertools import chain, combinations
import numpy as np
from sympy.utilities.iterables import multiset_partitions
from dataclasses import dataclass


Block = frozenset[int]
Partition = frozenset[Block]
T = TypeVar("T")


def part(p: list[list[int]]) -> Partition:
    """Convert p to a hashable representation of a partition."""
    return Partition([Block(b) for b in p])


@lru_cache(maxsize=None)
def partitions(items: frozenset[int]) -> list[Partition]:
    """Return all partitions of items."""
    parts = cast(list[list[list[int]]], multiset_partitions(list(items)))
    return [Partition([Block(b) for b in part]) for part in parts]


@lru_cache(maxsize=None)
def factorial(n: int) -> int:
    if n == 0 or n == 1:
        return 1
    return n * factorial(n - 1)


def remove_zeros(d: dict[T, int]) -> dict[T, int]:
    """Strip all items where value is zero.

    This simplifies coefficient dictionaries where we did d[k] += n; d[k] -= n.
    """
    return {k: v for k, v in d.items() if v != 0}


# %%
@lru_cache(maxsize=None)
def kf(pi: Partition) -> dict[Partition, int]:
    """Return a dict from E_partition: coefficient such that you can compute kf(pi) by summing the E terms."""
    first, *rest = list(pi)
    if rest:
        out: defaultdict[Partition, int] = defaultdict(int)
        for b0, coef0 in kf(frozenset([first])).items():  # note: could recurse on any block
            for b1, coef1 in kf(frozenset(rest)).items():
                out[frozenset([*b0, *b1])] += coef0 * coef1  # E_b0|b1 = coef0 * E_b0 [ coef1 * E_b1[f]]
        return remove_zeros(out)

    if len(first) == 1:
        return {frozenset([first]): 1}  # Trivial case of K_f({{X}}) = E_X

    out: defaultdict[Partition, int] = defaultdict(int)
    for other in partitions(first):
        if pi == other:
            out[pi] = 1
        else:
            for ob, on in kf(other).items():
                out[ob] -= on  # K_f[all] = E_all - (other K_fs)
    return remove_zeros(out)


# %%
# Check partitions look right
for n in range(1, 5):
    print(f"\n\n{n}:")
    pprint(partitions(frozenset(range(n))))

# %%
# Check against hand calculations N == 2
assert kf(part([[0]])) == {part([[0]]): 1}
assert kf(part([[1]])) == {part([[1]]): 1}
assert kf(part([[0], [1]])) == {part([[0], [1]]): 1}
assert kf(part([[0, 1]])) == {part([[0, 1]]): 1, part([[0], [1]]): -1}

# %%
# Check N == 3
assert kf(part([[0], [1], [2]])) == {part([[0], [1], [2]]): 1}
assert kf(part([[0], [1, 2]])) == {part([[0], [1, 2]]): 1, part([[0], [1], [2]]): -1}
assert kf(part([[0, 1], [2]])) == {part([[0, 1], [2]]): 1, part([[0], [1], [2]]): -1}
assert kf(part([[0, 2], [1]])) == {part([[0, 2], [1]]): 1, part([[0], [1], [2]]): -1}
assert kf(part([[0, 1, 2]])) == {
    part([[0], [1], [2]]): 2,
    part([[0], [1, 2]]): -1,
    part([[1], [0, 2]]): -1,
    part([[2], [0, 1]]): -1,
    part([[0, 1, 2]]): 1,
}
# %%
# Check one case of N == 4
assert kf(part([[0, 1], [2, 3]])) == {
    part([[0, 1], [2, 3]]): 1,
    part([[0], [1], [2, 3]]): -1,
    part([[0, 1], [2], [3]]): -1,
    part([[0], [1], [2], [3]]): 1,
}
# %%
# Note the SymPy ordering is different than the paper - we have the joint first and the independent last.
# KF is fast because cached, but this function is pretty slow because the number of cells goes up quickly (Bell number squared)
def matrix_form(n: int):
    parts = partitions(frozenset(range(n)))
    out = np.zeros((len(parts), len(parts)), dtype=np.int64)
    for i, part in enumerate(parts):
        for e, coef in kf(part).items():
            out[i, parts.index(e)] = coef
    return out


print(matrix_form(2))
print(matrix_form(3))
print(matrix_form(4))
# %%
# %%
def joint_coefs(n: int):
    """Coefficients of the joint term E_XYZ and also generally K_f_XYZ.

    Given by https://en.wikipedia.org/wiki/Cumulant#Joint_cumulants though not derived there.
    """
    return [factorial(len(pi) - 1) * (-1) ** (len(pi) - 1) for pi in partitions(frozenset(range(n)))]


# Check closed form for joint matches the recursive definition
# (In SymPy ordering the joint is the first row)
for n in range(1, 6):
    assert (matrix_form(n)[0] == joint_coefs(n)).all()

# %%
# Note this works on any T but the typechecker gets confused if you annotate with T :(
@lru_cache(maxsize=None)
def powerset(it: Iterable[int], max: Optional[int] = None, min=1) -> list[tuple[int, ...]]:
    """Return all subsets of it with length between min and max, inclusive."""
    items = list(it)
    if max is None:
        max = len(items)
    return [s for r in range(min, max + 1) for s in combinations(items, r)]


assert list(powerset(range(1, 4))) == [(1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)]
assert list(powerset(range(1, 4), min=0)) == [(), (1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)]
assert list(powerset(range(1, 4), max=2, min=1)) == [(1,), (2,), (3,), (1, 2), (1, 3), (2, 3)]
# %%
Expectation = frozenset[frozenset[int]]
Xs = frozenset[int]
Term = tuple[Expectation, Xs]


def mul_expectation(e1: Expectation, e2: Expectation):
    """Handle E_w|x E_yz E_{} = E_w|x|yz."""
    return frozenset([e for e in e1 | e2 if e])


def wick(x_s: Block) -> dict[Term, int]:
    """Return a Wick decomposition such that the terms sum to the product of x_s."""

    # Use an inner function to avoid passing x_s around
    @lru_cache(maxsize=None)
    def _wick(s: Block) -> dict[Term, int]:
        print(f"_wick: {s=}")
        if not s:
            return {(frozenset([frozenset()]), frozenset()): 1}

        terms: dict[Term, int] = defaultdict(int)
        terms[(frozenset([frozenset()]), x_s)] = 1  # product over all x_i
        for subset in powerset(s, min=0, max=len(s) - 1):  # strict subsets
            rest = frozenset([frozenset([i for i in x_s if i not in subset])])  # [n] \ S
            # distribute rest into each subterm by linearity of expectation
            for (sub_ex, sub_xs), sub_coef in _wick(frozenset(subset)).items():
                key = mul_expectation(rest, sub_ex), sub_xs
                terms[key] -= sub_coef
        return terms

    return remove_zeros(_wick(x_s))


# %%
# N = 0
assert wick(frozenset([])) == {(frozenset([frozenset()]), frozenset([])): 1}

# %%
# N = 1
expected = {
    (frozenset([frozenset()]), frozenset([1])): 1,  # x_1
    (frozenset([frozenset([1])]), frozenset()): -1,  # E_(X_1)
}
actual = wick(frozenset([1]))
assert actual == expected, actual

# %%
# N = 2
expected = {
    (frozenset([frozenset()]), frozenset([1, 2])): 1,
    (frozenset([frozenset([1, 2])]), frozenset()): -1,
    (frozenset([frozenset([2])]), frozenset([1])): -1,
    (frozenset([frozenset([1])]), frozenset([2])): -1,
    (frozenset([frozenset([1]), frozenset([2])]), frozenset()): 2,
}
actual = wick(frozenset([1, 2]))
assert actual == expected, actual


# %%
# N = 3
expected = {
    (frozenset([frozenset()]), frozenset([1, 2, 3])): 1,
    (frozenset([frozenset([3])]), frozenset([1, 2])): -1,
    (frozenset([frozenset([2])]), frozenset([1, 3])): -1,
    (frozenset([frozenset([1])]), frozenset([2, 3])): -1,
    (frozenset([frozenset([2, 3])]), frozenset([1])): -1,
    (frozenset([frozenset([2]), frozenset([3])]), frozenset([1])): 2,
    (frozenset([frozenset([1, 3])]), frozenset([2])): -1,
    (frozenset([frozenset([1]), frozenset([3])]), frozenset([1])): 2,
    (frozenset([frozenset([1, 2])]), frozenset([3])): -1,
    (frozenset([frozenset([1]), frozenset([2])]), frozenset([3])): 2,
    (frozenset([frozenset([1, 2, 3])]), frozenset([])): -1,
}
actual = wick(frozenset([1, 2, 3]))
assert actual == expected, actual

# %%
