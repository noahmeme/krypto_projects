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

collect_relations_mpqs() is the multiple-polynomial version (MPQS): f(x) above
grows linearly with x, so smooth values get rarer the longer we sieve. Instead,
it sieves many short intervals, each with its own polynomial whose values stay
about as small as at the start.
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


# ---------------------------------------------------------------------------
# Multiple polynomials (MPQS)
#
# For a = q^2 (q a prime with n a square mod q) and b with b^2 = n (mod a):
#
#     (a*x + b)^2 - n = a * h(x),   h(x) = a*x^2 + 2*b*x + c,   c = (b^2 - n) / a
#
# With r = (a*x + b) * q^(-1) mod n we get r^2 = h(x) (mod n), so a smooth h(x)
# is a relation. Choosing a close to sqrt(2n) / M keeps |h(x)| below about
# M * sqrt(n / 2) for x in [-M, M]. Every new q gives a fresh polynomial with
# equally small values, so we never have to sieve far out where values are big.
#
# h(x) is negative near the middle of the interval, so the factor base gets an
# extra "prime" -1 in front (base[0] == -1): exponent 1 means h(x) < 0.
# ---------------------------------------------------------------------------

def is_prime(m):
    """Miller-Rabin. q can be huge for big n (about 4e22 for RSA-100), so trial division is far too slow.
    With these 12 bases the answer is exact for every m < 3.3e24, and almost certain above that."""
    bases = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    if m < 2:
        return False
    for p in bases:
        if m % p == 0:
            return m == p
    d, s = m - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in bases:
        x = pow(a, d, m)
        if x in (1, m - 1):
            continue
        for _ in range(s - 1):
            x = x * x % m
            if x == m - 1:
                break
        else:
            return False
    return True


def next_polynomial_prime(n, after):
    """The next prime q > after with n a square mod q, used for a = q^2."""
    q = after + 1 if after % 2 == 0 else after + 2
    while not (is_prime(q) and n % q and legendre(n, q) == 1):
        q += 2
    return q


def collect_relations_mpqs(n, base, target, smooth_exponents, parity_mask, max_polynomials=50_000):
    """Sieve one interval per polynomial until `target` relations (r, exponents, mask) are found.

    base[0] must be -1 (the sign); base[1:] are the primes from factor_base().
    """
    primes = base[1:]
    half = BLOCK_SIZE // 2                     # x runs from -half to half - 1
    roots = [(p, log(p), sorted(set(sqrt_mod(n, p)))) for p in primes if p >= SMALL_PRIME_LIMIT]

    relations = []
    seen = set()
    candidates = 0
    q = max(isqrt(isqrt(2 * n) // half), 2)    # a = q^2 should be about sqrt(2n) / half

    for polynomial in range(1, max_polynomials + 1):
        q = next_polynomial_prime(n, q)
        a = q * q
        t = sqrt_mod(n, q)[0]
        # Hensel lifting from t^2 = n (mod q) to b^2 = n (mod q^2).
        b = (t + (n - t * t) // q * pow(2 * t, -1, q) % q * q) % a
        q_inv = pow(q, -1, n)

        def h(x):
            return ((a * x + b) ** 2 - n) // a

        # The sieve: p divides h(x) exactly when a*x + b = +-s (mod p), s^2 = n.
        logs = [0.0] * BLOCK_SIZE
        for p, log_p, square_roots in roots:
            if p == q:
                continue                       # a = q^2 has no inverse mod q
            a_inv = pow(a, -1, p)
            for s in square_roots:
                first = ((s - b) * a_inv + half) % p
                for i in range(first, BLOCK_SIZE, p):
                    logs[i] += log_p

        vertex = -b // a                       # where h(x) is most negative
        for chunk_start in range(0, BLOCK_SIZE, CHUNK):
            lo, hi = chunk_start - half, chunk_start - half + CHUNK - 1
            # |h| on this chunk is largest at an end or at the vertex.
            biggest = max(abs(h(lo)), abs(h(hi)), abs(h(vertex)) if lo <= vertex <= hi else 0)
            threshold = log(max(biggest, 1)) - SLACK

            for i, total in enumerate(logs[chunk_start:chunk_start + CHUNK]):
                if total < threshold:
                    continue
                candidates += 1
                x = lo + i
                value = h(x)
                if value == 0:
                    continue
                exponents = smooth_exponents(abs(value), primes)
                if exponents is None:
                    continue

                exponents = [1 if value < 0 else 0] + exponents
                mask = parity_mask(exponents)
                if mask in seen:
                    continue
                seen.add(mask)
                relations.append(((a * x + b) * q_inv % n, exponents, mask))
                if len(relations) == target:
                    collect_relations_mpqs.stats = {"polynomials": polynomial, "candidates": candidates}
                    return relations

    collect_relations_mpqs.stats = {"polynomials": max_polynomials, "candidates": candidates}
    return relations
