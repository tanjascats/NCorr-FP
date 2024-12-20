from attacks.attack import Attack
import time
import random
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd


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
    def find_cluster(self, data, cluster_size, tolerance=10, max_iterations=100):
        # Normalize the data
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)

        # Initial guess for the number of clusters
        n_clusters = len(data) // cluster_size

        iteration = 0
        while iteration < max_iterations:
            # Apply k-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            kmeans.fit(data_scaled)
            labels = kmeans.labels_
            centroids = kmeans.cluster_centers_

            # Calculate the average distance to centroid for each cluster
            distances = np.zeros(n_clusters)
            cluster_sizes = np.zeros(n_clusters)
            for i in range(n_clusters):
                cluster_points = data_scaled[labels == i]
                centroid = centroids[i]
                distances[i] = np.mean(np.sqrt(np.sum((cluster_points - centroid) ** 2, axis=1)))
                cluster_sizes[i] = len(cluster_points)

            # Identify the cluster with the minimum average distance
            best_cluster_idx = np.argmin(distances)
            best_cluster_size = cluster_sizes[best_cluster_idx]

            # Check if the size of the best cluster is within the tolerance of the desired size
            if abs(best_cluster_size - cluster_size) <= tolerance:
                print(f"Found suitable cluster in {iteration + 1} iterations.")
                best_cluster_members = data[labels == best_cluster_idx]
                return pd.DataFrame(best_cluster_members, columns=data.columns), best_cluster_size

            # Adjust the number of clusters
            if best_cluster_size < cluster_size:
                n_clusters -= max(1, n_clusters // 10)
            else:
                n_clusters += max(1, n_clusters // 10)

            iteration += 1

        print("Maximum iterations reached without finding a suitable cluster.")
        return None, None

    def run(self, dataset, fraction, cluster_size_factor=5, cluster=None, random_state=0):
        """

        Args:
            dataset:
            fraction: (float) flips fraction*full_data_size values (all of them inside the cluster)
            cluster_size_factor:
            cluster:
            random_state:

        Returns:

        """
        # adapt the cluster size factor --> cs_factor*fraction < 1 # todo

        start = time.time()
        modified_df = dataset.copy()

        if cluster is None:
            cluster, size = self.find_cluster(dataset, cluster_size=10)  # todo

        # Ensure that specific_rows is a valid subset of DataFrame's index
        specific_rows = [row for row in cluster.index if row in dataset.index]

        # Calculate the number of elements in the specified rows
        total_elements = modified_df.loc[specific_rows].size
        n = int(fraction * total_elements)

        if fraction > 1.0:
            raise ValueError("Number of flips exceeds the total number of elements in the DataFrame.")

        # Randomly choose n indices from the specified rows in the DataFrame
        np.random.seed(random_state)
        row_indices = np.random.choice(specific_rows, size=n, replace=True)
        col_indices = np.random.randint(0, modified_df.shape[1], size=n)

        # Runtime improvement: precompute unique values for each column
        unique_values = {col: modified_df[col].unique() for col in modified_df.columns}

        # Iterate over selected row and column indices to flip values
        for row, col in zip(row_indices, col_indices):
            col_name = modified_df.columns[col]
            column_values = unique_values[col_name]
            current_value = modified_df.at[row, col_name]

            # Exclude the current value to ensure a different value is chosen
            possible_values = column_values[column_values != current_value]
            if possible_values.size > 0:
                new_value = np.random.choice(possible_values)
                modified_df.at[row, col_name] = new_value
            else:
                # In case all values in the column are the same
                modified_df.at[row, col_name] = current_value

        print("Flipping attack runtime on " + str(fraction * 100) + "% of entries: " +
              str(time.time() - start) + " sec.")

        return modified_df