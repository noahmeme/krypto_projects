import time
from math import gcd, isqrt, prod
from pathlib import Path

import sieve

# Number of relations to collect beyond the factor base size: L + EXTRA_L in total.
EXTRA_L = 10

# How relations are found:
#   "simple": try r = isqrt(k*n) + j for many k and j and trial-divide every r^2 mod n (the project PDF's method)
#   "real":   one polynomial f(x) = (r0 + x)^2 - n with log sieving, see sieve.py
SIEVE_METHOD = "real"


def read_n(path):
    # task14.txt holds the number in LongInt format, e.g. "[+ 196097436898174954190359]".
    text = Path(path).read_text()
    return int(text.strip().strip("[]").replace("+", "").strip())


def primes_below(bound):
    is_prime = [True] * bound
    for prime in range(2, isqrt(bound) + 1):
        if is_prime[prime]:
            for multiple in range(prime * prime, bound, prime):
                is_prime[multiple] = False
    return [prime for prime in range(2, bound) if is_prime[prime]]


def smooth_exponents(value, base):
    if value == 0:
        return None

    exponents = []
    for prime in base:
        count = 0
        while value % prime == 0:
            value //= prime
            count += 1
        exponents.append(count)

    return exponents if value == 1 else None


def parity_mask(exponents):
    # One bit per prime: 1 if its exponent is odd. This is the relation's row in the binary matrix.
    return sum((e % 2) << i for i, e in enumerate(exponents))


def add_row(parity, selected, pivots):
    while parity:
        column = parity.bit_length() - 1
        if column not in pivots:
            pivots[column] = (parity, selected)
            return None

        pivot_parity, pivot_selected = pivots[column]
        parity ^= pivot_parity
        selected ^= pivot_selected

    return selected


def collect_relations(n, base, max_k, max_j):
    target = len(base) + EXTRA_L
    relations = []
    seen = set()

    for k in range(1, max_k + 1):
        root = isqrt(k * n)
        for j in range(1, max_j + 1):
            r = root + j
            exponents = smooth_exponents(pow(r, 2, n), base)
            if exponents is None:
                continue

            mask = parity_mask(exponents)
            if mask in seen:
                continue
            seen.add(mask)
            relations.append((r, exponents, mask))
            if len(relations) == target:
                return relations

    return relations


def factor(n, bound, max_k, max_j, method="simple"):
    base = primes_below(bound)
    if method == "real":
        # Only primes that can divide f(x) are useful, which is about half of them.
        base = sieve.factor_base(n, base)
        relations = sieve.collect_relations(n, base, len(base) + EXTRA_L, smooth_exponents, parity_mask)
    else:
        relations = collect_relations(n, base, max_k, max_j)
    if len(relations) < len(base) + EXTRA_L:
        return None

    pivots = {}
    dependencies = []
    for index, (_, _, mask) in enumerate(relations):
        selected = add_row(mask, 1 << index, pivots)
        if selected is not None:
            dependencies.append(selected)

    for selected in dependencies:
        chosen = [
            relation
            for index, relation in enumerate(relations)
            if selected & (1 << index)
        ]

        x = prod(r for r, _, _ in chosen) % n
        totals = [sum(column) for column in zip(*(es for _, es, _ in chosen))]
        y = prod(pow(p, total // 2, n) for p, total in zip(base, totals)) % n

        d = gcd(x - y, n)
        if 1 < d < n:
            return d, n // d

    return None


def smoothness_bound(n, method="simple"):
    # Small numbers need a small factor base, or there are not enough distinct relations to find.
    if n < 10**6:
        return 100
    if n < 10**13:
        return 1_000
    if method == "real":
        # Sieving makes each extra prime cheap, so a larger B pays off.
        # Measured: group 14's N 0.04 s at B = 5,000; Exercise 4 0.37 s at B = 20,000 (5,000 runs out of x values).
        return 5_000 if n < 10**25 else 20_000
    # Measured: group 14's N (24 digits) is fastest at B = 2,000 (6.5 s vs 12.8 s at 10,000).
    if n < 10**25:
        return 2_000
    # Measured: the Exercise 4 number (29 digits) is fastest at B = 5,000 (83 s vs 97 s);
    # B <= 3,000 runs out of candidates before finding enough relations.
    return 5_000


if __name__ == "__main__":
    max_k = 300_000
    max_j = 8

    candidates = [
        ("Test", 323),
        ("Test", 307561),
        ("Test", 31741649),
        ("Test", 3205837387),
        ("Test", 392742364277),
        ("Group 14 N", read_n(Path(__file__).parent / "task14.txt")),
        ("Exercise 4", 92434447339770015548544881401),
    ]

    for label, n in candidates:
        B = smoothness_bound(n, SIEVE_METHOD)
        start = time.process_time()
        factors = factor(n, B, max_k, max_j, method=SIEVE_METHOD)
        elapsed = time.process_time() - start

        if factors is None:
            print(f"{label}: {n}  no factor found ({SIEVE_METHOD} sieve, B = {B:,})  [{elapsed:.2f} s CPU]")
        else:
            p, q = factors
            assert p * q == n
            print(f"{label}: {n} = {p} * {q}  ({SIEVE_METHOD} sieve, B = {B:,})  [{elapsed:.2f} s CPU]")
