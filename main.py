"""
Code for generalized cumulant decomposition
"""
# %%
from pprint import pprint
from typing import cast, TypeVar, Sequence
from collections import defaultdict
from functools import lru_cache

import numpy as np
from sympy.utilities.iterables import multiset_partitions

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
