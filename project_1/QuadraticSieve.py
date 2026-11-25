import os
import sys
import time
from math import gcd, isqrt, prod
from pathlib import Path
from subprocess import DEVNULL, run
from tempfile import TemporaryDirectory

# GaussBin.exe on Windows (download from Canvas). On macOS/Linux, compile the included source in this folder:
#   clang++ -O2 -o GaussBin GaussBin_unix.cpp
GAUSS_BIN = Path(__file__).resolve().with_name("GaussBin.exe" if sys.platform == "win32" else "GaussBin")
PRIMES_FILE = Path(__file__).resolve().with_name("prim_2_24.txt")

# Generate prime numbers below bound and return list with them
def primes_below(bound):
    is_prime = [True] * bound
    for prime in range(2, isqrt(bound) + 1):
        if is_prime[prime]:
            for multiple in range(prime * prime, bound, prime):
                is_prime[multiple] = False
    return [prime for prime in range(2, bound) if is_prime[prime]]

# Read primes below bound from file with primes and return them in a list
def read_primes_below(bound):
    primes = []
    with PRIMES_FILE.open(encoding="ascii") as prime_file:
        for line in prime_file:
            for token in line.split():
                prime = int(token)
                if prime >= bound:
                    return primes
                primes.append(prime)
    return primes

# Calculates exponents for the primes in base to form the value.
# If the value isn't 1 at the end some prime not included in the base is needed.
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

# Runs the given code for performing Gaussian elimination. Sets up temporary files to run it so they get deleted later.
# Writes the full exponents to a file that the program reads, runs and then returns the selector vectors with one bit
# per original relation (r, exponents). Each vector describes a combination of rows whose exponent sums are all even.
def gauss_dependencies(relations, base_size):
    with TemporaryDirectory(prefix="gaussbin-") as directory:
        input_path = Path(directory) / "matrix.txt"
        output_path = Path(directory) / "solutions.txt"

        with input_path.open("w", encoding="ascii") as matrix:
            matrix.write(f"{len(relations)} {base_size}\n")
            for _, exponents in relations:
                matrix.write(" ".join(map(str, exponents)) + "\n")

        run([str(GAUSS_BIN), str(input_path), str(output_path)], stdout=DEVNULL, check=True)

        with output_path.open(encoding="ascii") as output:
            count = int(output.readline())
            return [output.readline().split() for _ in range(count)]

# Packs each exponent's parity into one bit. Each bit comes from exponent % 2.
def parity_mask(exponents):
    return sum((exponent % 2) << column for column, exponent in enumerate(exponents))

# Uses existing pivots to reduce a row. If its highest remaining 1 is in a column without a pivot, it stores the reduced
# row as a new pivot and return None. If the row reduces to zero, it returns a mask of the original relations that
# combine to give zero.
def add_row(parity, selected, pivots):
    while parity:
        column = parity.bit_length() - 1
        if column not in pivots:
            pivots[column] = (parity, selected)
            return None

        # XOR adds the parity rows and their selectors modulo 2.
        pivot_parity, pivot_selected = pivots[column]
        parity ^= pivot_parity
        selected ^= pivot_selected

    return selected

# Yields a selector vector of original rows to combine to get zero parity. If the row becomes a pivot it won't produce
# any selector. The selectors refer to original relations in input order.
def python_dependencies(relations):
    pivots = {}
    for index, (_, exponents) in enumerate(relations):
        selected = add_row(parity_mask(exponents), 1 << index, pivots)
        if selected is not None:
            # A zero parity row gives a dependency in the original row order.
            yield [
                "1" if selected & (1 << row) else "0"
                for row in range(len(relations))
            ]


# Finds relations, meaning candidate value r and the exponents for r^2 mod n over the factor base, as (r, exponents).
# The set seen contains all the parity patterns, meaning if the exponents are even or odd, of the seen numbers to not
# add new numbers that correspond to the same parity pattern as an already added number.
# The list of candidates will be at most the length set by target, but could be less if no more candidates are accepted.
def collect_relations(n, base, max_k, max_j, target):
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
            relations.append((r, exponents))
            if len(relations) == target:
                return relations

    return relations

# Calculates the factors by producing the base of primes and the relations before checking there are enough relations
# according to the target.
def factor(n, bound, max_k, max_j, *, extra_relations=10, use_prime_file=True, use_gauss_bin=True):
    base = read_primes_below(bound) if use_prime_file else primes_below(bound)
    target = len(base) + extra_relations
    relations = collect_relations(n, base, max_k, max_j, target)
    if len(relations) < target:
        return None

    if use_gauss_bin:
        dependencies = gauss_dependencies(relations, len(base))
    else:
        dependencies = python_dependencies(relations)

    # Selectors tells us what rows from relations to add to form our candidates by marking them with 0 and 1.
    # Then chosen contains the (r, exponents) pairs whose selector is 1, x is the product of the chosen r values reduced
    # modulo n, totals is a list showing the sums column by column of the exponents from the chosen relations and y is
    # the product of the base primes powered to the half of values in totals. The reason for taking half is because we
    # use the formula x^2 = y^2 mod n so now we then get only y instead of y^2.
    for selectors in dependencies:
        chosen = [
            relation for relation, bit in zip(relations, selectors, strict=True) if bit == "1"
        ]
        x = prod(r for r, _ in chosen) % n
        totals = [sum(column) for column in zip(*(es for _, es in chosen))]
        y = prod(pow(p, total // 2, n) for p, total in zip(base, totals)) % n

        d = gcd(x - y, n)
        if 1 < d < n:
            return d, n // d

    return None


def cpu_time():
    # CPU seconds used by this program plus finished child processes (GaussBin runs as a child process,
    # so time.process_time() alone would miss it). On Windows the child part is always 0.
    t = os.times()
    return t.user + t.system + t.children_user + t.children_system


if __name__ == "__main__":
    n = 196097436898174954190359
    B = 8_167 # This is the 1025th prime, so only the 1024 first will be included.
    max_k = 300_000
    max_j = 8
    extra_relations = 10

    use_prime_file = True  # Set to True to read prim_2_24.txt.
    use_gauss_bin = True  # Set to False to use Python Gaussian elimination.

    # Elapsed time is measured the same way on every OS and always includes GaussBin.
    # CPU time only includes GaussBin on macOS/Linux (see cpu_time above).
    start_cpu, start_wall = cpu_time(), time.perf_counter()
    factors = factor(
        n, B, max_k, max_j,
        extra_relations=extra_relations,
        use_prime_file=use_prime_file,
        use_gauss_bin=use_gauss_bin,
    )
    elapsed_cpu = cpu_time() - start_cpu
    elapsed_wall = time.perf_counter() - start_wall
    print(f"CPU time: {elapsed_cpu:.2f} s")
    print(f"Elapsed time: {elapsed_wall:.2f} s")

    if factors is None:
        print(f"No factor found with B = {B:,}, max_k = {max_k:,}, and max_j = {max_j:,}.")
    else:
        p, q = factors
        print(p, q)
        product = p * q
        if product == n:
            print("Check shows that factors are correct.")
        else:
            print("Check shows that factors are incorrect. Product = ", product)