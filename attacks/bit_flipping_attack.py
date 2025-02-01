from attacks.attack import Attack
import time
import random
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd
from NCorrFP.NCorrFP import NCorrFP
from NCorrFP.demo import Demo
from datasets import *


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


class InfluentialRecordFlippingAttack(Attack):
    def __init__(self):
        super().__init__()

    """
        Knowledge attack for NCorr-FP
        Clusters the data according to a NN algorithm. 
       Flips 'fraction' data values inside the cluster. 
       Runs the attack; gets a copy of 'dataset' with 'fraction' altered items
       fraction [0,1]
       """
    def find_influential_records(self, data, cluster_size, gamma=1, k=None, sk=999, fp_len=100):
        # direct approach: emulate the embedding with attacker's parameters
        # list of arrays, arrays contain indices of neighbourhood
        if k is None:
            k = int(0.01 * data.dataframe.shape[0])
        param = {'gamma': gamma,  # , 4, 8, 16, 32], # --> might have some influence
                 'k': k,  # 1% of data size (knowledgeable)
                 'fingerprint_length': fp_len,  # , 256, 512],#, 128, 256],  # , 128, 256],
                 'n_recipients': 20,
                 'sk': sk,  # attacker's secret key
                 'id': 0
                 }
        scheme = NCorrFP(gamma=param['gamma'], fingerprint_bit_length=param['fingerprint_length'], k=param['k'],
                         number_of_recipients=param['n_recipients'], fingerprint_code_type='tardos')
        fp_data = scheme.insertion(data, secret_key=param['sk'], recipient_id=param['id'], save_computation=True)
        detected_fp, votes, suspect_probvec = scheme.detection(fp_data, secret_key=param['sk'],
                                                               primary_key='Id')
        demo = Demo(scheme)
        neighborhoods = [
            demo.insertion_iter_log[i]['neighbors'] for i in range(len(demo.insertion_iter_log))
        ]

        # Flatten the list of neighbourhoods into a single array
        all_indices = np.concatenate(neighborhoods)

        # count the frequency of each index
        index_series = pd.Series(all_indices)
        frequency_counts = index_series.value_counts()

        # the top N most frequent indices, i.e. the most influential records
        if cluster_size == 'all':
            top_n_idx = frequency_counts.keys()
        else:
            top_n = cluster_size  # This is an absolute number of records
            top_n_idx = frequency_counts.head(top_n).keys()

        return data.dataframe.iloc[top_n_idx]

    def run(self, dataset, fraction, cluster=None, data_name='adult', cluster_length=10000, random_state=0):
        """
        Runs the cluster & flip attack.
        Args:
            dataset:
            fraction: (float) flips fraction*full_data_size values (all of them inside the cluster)
            cluster_length:
            cluster: (pandas.DataFrame)
            random_state:

        Returns: modified dataset, cluster

        """
        start = time.time()
        modified_df = dataset.copy()

        if data_name == 'adult':
            d = Adult()
        else:
            d = None

        # find the most influential records
        if cluster is None:
            if fraction * len(
                    modified_df) >= cluster_length:  # if we are trying to flip more than there is n the cluster
                # increase the cluster
                cluster_length = int(np.ceil(fraction * len(modified_df)))
                print("!Warning: all data values in the cluster are being flipped.")
            cluster = self.find_influential_records(d, cluster_size=cluster_length)
        else:
            print('Cluster size: ', len(cluster))
            print('- that is {}% of data length'.format(100*len(cluster)/len(modified_df)))
            print('Trying to flip ', fraction)
            if len(cluster) < fraction*len(modified_df):
                exit('Cluster is not sufficiently large.')

        cluster_frac = fraction*modified_df.size / cluster.size

        # flip the values inside the cluster
        flipping = FlippingAttack()
        modified_cluster = flipping.run(cluster, cluster_frac, random_state)
        print(modified_cluster)

        # fill in the flipped cluster
        modified_df.update(modified_cluster)

        print("Flipping attack runtime on " + str(fraction * 100) +
              "% of entries: ({}% of cluster entries)".format(100*cluster_frac) +
              str(time.time() - start) + " sec.")

        return modified_df, cluster
