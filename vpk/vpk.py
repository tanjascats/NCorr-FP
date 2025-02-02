import pandas as pd

MAX = 1048576  #2 ** 63 - 1


def record_vpk(row, n_lowest_hashes=7):
    # Compute absolute hash values for each cell (convert cell to string)
    hash_values = [abs(hash(str(val))) for val in row]
    # Sort the hash values in ascending order
    hash_values.sort()
    # Take the three smallest values (or fewer if row has less than three values)
    selected = hash_values[:n_lowest_hashes]
    # Combine them using prime multipliers to get a single integer.
    # (These primes are chosen arbitrarily to mix the numbers.)
    primes = [1000003, 1000033, 1000211, 1000231, 1000249, 1000253, 1000271, 1000289, 1000297]
    combined = 0
    for i, h in enumerate(selected):
        # combined += h
        combined += h * primes[i]
    # Return a value between 0 and MAX
    return combined % MAX


def record_marking_attr(row):
    # Compute absolute hash values for each cell (convert cell to string)
    return max(row, key=lambda x: abs(hash(x)))
