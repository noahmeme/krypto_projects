"""Real sieving for the Quadratic Sieve, using one polynomial.

Instead of trial-dividing every candidate, we use

    f(x) = (r0 + x)^2 - n,   r0 = isqrt(n) + 1,   x = 0, 1, 2, ...

and the fact that a prime p divides f(x) exactly when (r0 + x)^2 = n (mod p).
If t is a square root of n mod p, that happens for x = t - r0 (mod p) and
x = -t - r0 (mod p): every p-th x, from two starting points. So for each prime
we add log(p) to every p-th slot of an array, and only slots whose total gets
close to log f(x) are trial-divided.

Every f(x) is congruent to r^2 mod n (with r = r0 + x), so a smooth f(x) is a
relation r^2 = f(x) (mod n) exactly like in the simple version.
"""
from math import isqrt, log

# Primes below this are not sieved: they hit very often (p = 3 hits every
# 1.5 slots) but add little log, so they cost more than they help. SLACK
# makes up for their missing contribution.
SMALL_PRIME_LIMIT = 30

# How far below log f(x) a slot's sum may be and still get trial-divided.
# It covers the skipped small primes, prime powers (only log p is added once)
# and rounding. Larger slack: more false candidates. Smaller: missed relations.
# Measured: 12 was fastest for both group 14's N and the Exercise 4 number (tried 6-18).
SLACK = 12.0

BLOCK_SIZE = 65_536     # number of x values sieved at a time
CHUNK = 1_024           # the threshold log f(x) is recomputed every CHUNK slots


def legendre(n, p):
    """1 if n is a nonzero square mod the odd prime p, p - 1 if not, 0 if p divides n."""
    return pow(n, (p - 1) // 2, p)


def sqrt_mod(n, p):
    """All square roots of n mod the prime p (Tonelli–Shanks). n must be a square mod p."""
    n %= p
    if p == 2 or n == 0:
        return [n]
    if p % 4 == 3:                          # easy case: one exponentiation
        t = pow(n, (p + 1) // 4, p)
        return [t, p - t]

    # Write p - 1 = q * 2^s with q odd, and find a non-square z.
    q, s = p - 1, 0
    while q % 2 == 0:
        q //= 2
        s += 1
    z = 2
    while legendre(z, p) != p - 1:
        z += 1

    m, c, t, r = s, pow(z, q, p), pow(n, q, p), pow(n, (q + 1) // 2, p)
    while t != 1:
        i, t2 = 0, t                        # smallest i with t^(2^i) = 1
        while t2 != 1:
            t2 = t2 * t2 % p
            i += 1
        b = pow(c, 1 << (m - i - 1), p)
        m, c, t, r = i, b * b % p, t * b * b % p, r * b % p
    return [r, p - r]


def factor_base(n, primes):
    """Keep only the primes that can divide some f(x): 2, and odd p where n is a square mod p."""
    return [p for p in primes if p == 2 or legendre(n, p) == 1]


def collect_relations(n, base, target, smooth_exponents, parity_mask, max_blocks=200):
    """Sieve block by block until `target` relations (r, exponents, mask) are found."""
    r0 = isqrt(n) + 1

    # For every sieved prime: its log and the offsets x mod p where p divides f(x).
    sieve_primes = []
    for p in base:
        if p < SMALL_PRIME_LIMIT:
            continue
        offsets = sorted({(t - r0) % p for t in sqrt_mod(n, p)})
        sieve_primes.append((p, log(p), offsets))

    relations = []
    seen = set()
    candidates = 0

    for block in range(max_blocks):
        start = block * BLOCK_SIZE
        logs = [0.0] * BLOCK_SIZE

        # The sieve itself: only additions, no divisions.
        for p, log_p, offsets in sieve_primes:
            for offset in offsets:
                for x in range((offset - start) % p, BLOCK_SIZE, p):
                    logs[x] += log_p

        # Pick out slots that are probably smooth and trial-divide only those.
        for chunk_start in range(0, BLOCK_SIZE, CHUNK):
            # f(x) only grows with x, so using the chunk's first value never rejects a smooth slot.
            # For large n, log f(x) barely changes within a chunk anyway.
            r_first = r0 + start + chunk_start
            threshold = log(max(r_first * r_first - n, 1)) - SLACK

            for i, total in enumerate(logs[chunk_start:chunk_start + CHUNK]):
                if total < threshold:
                    continue
                candidates += 1
                r = r0 + start + chunk_start + i
                exponents = smooth_exponents(r * r - n, base)
                if exponents is None:
                    continue

                mask = parity_mask(exponents)
                if mask in seen:
                    continue
                seen.add(mask)
                relations.append((r, exponents, mask))
                if len(relations) == target:
                    collect_relations.stats = {"x_values": start + chunk_start + i + 1, "candidates": candidates}
                    return relations

    collect_relations.stats = {"x_values": max_blocks * BLOCK_SIZE, "candidates": candidates}
    return relations
