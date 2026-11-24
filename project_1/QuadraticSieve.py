from math import gcd, isqrt, prod

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
    target = len(base) + 5
    relations = []
    seen = set()

    for k in range(1, max_k + 1):
        root = isqrt(k * n)
        for j in range(1, max_j + 1):
            r = root + j
            exponents = smooth_exponents(pow(r, 2, n), base)
            if exponents is None:
                continue

            mask = sum((e % 2) << i for i, e in enumerate(exponents))
            if mask in seen:
                continue
            seen.add(mask)
            relations.append((r, exponents, mask))
            if len(relations) == target:
                return relations

    return relations


def factor(n, bound, max_k, max_j):
    base = primes_below(bound)
    relations = collect_relations(n, base, max_k, max_j)
    if len(relations) < len(base) + 5:
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


if __name__ == "__main__":
    n = 92434447339770015548544881401
    B = 10_000
    max_k = 300_000
    max_j = 8

    factors = factor(n, B, max_k, max_j)
    if factors is None:
        print(f"No factor found with B = {B:,}, max_k = {max_k:,}, and max_j = {max_j:,}.")
    else:
        print(*factors)
