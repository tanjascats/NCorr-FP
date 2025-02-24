import pandas as pd


def add_virtual_primary_key(csv_file_path):
    """
    Reads a CSV file of tabular data, creates a virtual primary key for each row by:
      - Computing the hash (absolute value) of each cell (converted to string)
      - Sorting these hash values and taking the three smallest
      - Combining these three values with a linear combination using prime multipliers
      - Reducing the combined value modulo MAX_INT (2**63 - 1)
    Inserts the virtual primary key as the first column named "Id" and returns the new DataFrame.

    Args:
        csv_file_path (str): Path to the CSV file containing the tabular data.

    Returns:
        pd.DataFrame: A new DataFrame with the virtual primary key column "Id" inserted at the beginning.
    """
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file_path)

    # Define maximum integer value (here we use 2^63 - 1)
    MAX = 1048576  #2 ** 63 - 1

    def compute_virtual_pk(row, n_lowest_hashes=7):  # <--- adjust vpk parameters here
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

    # Apply the function to each row to create a new column "Id"
    df['Id'] = df.apply(compute_virtual_pk, axis=1)

    # Move the "Id" column to be the first column
    cols = df.columns.tolist()
    cols = ['Id'] + [c for c in cols if c != 'Id']
    df = df[cols]

    return df


if __name__ == '__main__':
    data_with_vpk = add_virtual_primary_key("datasets/adult_train.csv")
    print(data_with_vpk.head(100))
    print(len(data_with_vpk['Id'].unique()))
    print(len(data_with_vpk['Id']))
    data_with_vpk.to_csv("output_with_id.csv", index=False)
