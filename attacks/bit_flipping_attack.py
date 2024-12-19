from attacks.attack import Attack
import time
import random
import numpy as np


class BitFlippingAttack(Attack):

    def __init__(self):
        super().__init__()

    """
    Runs the attack; gets a copy of 'dataset' with 'fraction' altered items
    fraction [0,1]
    """
    def run(self, dataset, fraction):
        start = time.time()
        # never alters the ID or the target (if provided)
        altered = dataset.copy()
        for i in range(int(fraction*(dataset.size-len(dataset)))):
            row = random.choice(dataset['Id'])
            column = random.choice(dataset.columns.drop(labels=["Id"]))
            value = dataset[column][row]
            if dataset[column].dtype == 'O':
                # categorical
                domain = list(set(dataset[column][:]))
                domain.remove(value)
                new_value = random.choice(domain)
            else:
                # numerical
                new_value = value ^ 1  # flipping the least significant bit
            altered.at[row, column] = new_value

        print("Bit-flipping attack runtime on " + str(fraction*100) + "% of entries: " +
              str(time.time() - start) + " sec.")
        return altered

    def run_temp(self, dataset, fraction):
        start = time.time()
        # never alters the ID or the target (if provided)
        #altered = dataset.copy()
        for i in range(int(fraction * (dataset.size - len(dataset)))):
            row = random.choice(dataset['Id'])
            column = random.choice(dataset.columns.drop(labels=["Id"]))
            value = dataset[column][row]
            if dataset[column].dtype == 'O':
                # categorical
                domain = list(set(dataset[column][:]))
                #domain.remove(value)
                new_value = random.choice(domain)
            #altered.at[row, column] = new_value

        print("Bit-flipping attack runtime on " + str(fraction * 100) + "% of entries: " +
              str(time.time() - start) + " sec.")
        return True


class FlippingAttack(Attack):
    def __init__(self):
        super().__init__()

    """
       Flips 'fraction' data values from random values from attribute domains. 
       Runs the attack; gets a copy of 'dataset' with 'fraction' altered items
       fraction [0,1]
       """

    def run(self, dataset, fraction, random_state=0):
        start = time.time()
        # Create a copy to avoid modifying the original DataFrame
        modified_df = dataset.copy()

        total_elements = modified_df.size
        n = int(fraction * total_elements)

        if fraction > 1.0:
            raise ValueError("Number of flips exceeds the total number of elements in the DataFrame.")

        # Randomly choose n indices from the DataFrame
        np.random.seed(random_state)
        random_indices = np.random.choice(total_elements, n, replace=False)
        rows, cols = np.unravel_index(random_indices, modified_df.shape)

        # Runtime improvement: precompute unique values for each column
        unique_values = {col: modified_df[col].unique() for col in modified_df.columns}

        for row, col in zip(rows, cols):
            col_name = modified_df.columns[col]
            column_values = unique_values[col_name]
            current_value = modified_df.iat[row, col]

            # Exclude the current value to ensure a different value is chosen
            new_value = np.random.choice(column_values[column_values != current_value])
            modified_df.iat[row, col] = new_value

        print("Flipping attack runtime on " + str(fraction * 100) + "% of entries: " +
              str(time.time() - start) + " sec.")

        return modified_df


class ClusterFlippingAttack(Attack):
    def __init__(self):
        super().__init__()

    """
        Knowledge attack for NCorr-FP
        Clusters the data according to a NN algorithm. 
       Flips 'fraction' data values inside the cluster. 
       Runs the attack; gets a copy of 'dataset' with 'fraction' altered items
       fraction [0,1]
       """
    def run(self, dataset, fraction, cluster_size=0.05, cluster=None, random_state=0):
        start = time.time()
        modified_df = dataset.copy()

        # todo
        return modified_df, cluster