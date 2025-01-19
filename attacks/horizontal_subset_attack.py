import time
import numpy as np
import pandas as pd
from datasets import *
from attacks.attack import Attack
from NCorrFP.NCorrFP import NCorrFP
from NCorrFP.demo import Demo


class HorizontalSubsetAttack(Attack):

    def __init__(self):
        super().__init__()

    """
    Runs the attack; gets a random subset of a dataset of size fraction*data_size
    fraction [0,1]
    """
    def run(self, dataset, fraction, random_state=None):
        if fraction < 0 or fraction > 1:
            return None

        start = time.time()
        subset = dataset.sample(frac=fraction, random_state=random_state)
        print("Subset attack runtime on " + str(int(fraction*len(dataset))) + " out of " + str(len(dataset)) +
              " entries: " + str(time.time()-start) + " sec.")
        return subset


class InfluentialRecordDeletionAttack(Attack):
    """
    Knowledge attack for NCorr-FP
    Finds the influential records in the data according to the NN algorithm.
    Deletes most influential records.
    """
    def __init__(self):
        super().__init__()

    def find_influential_records(self, data, cluster_size):
        # direct approach: emulate the embedding with attacker's parameters
        # list of arrays, arrays contain indices of neighbourhood
        param = {'gamma': 1,  # , 4, 8, 16, 32], # --> might have some influence
                 'k': int(0.01 * data.dataframe.shape[0]),  # 1% of data size (knowledgeable)
                 'fingerprint_length': 100,  # , 256, 512],#, 128, 256],  # , 128, 256],
                 'n_recipients': 20,
                 'sk': 999,  # attacker's secret key
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

    def run(self, dataset, fraction, importance_order=None, data_name='adult', cluster_length=10000, random_state=0):
        start = time.time()
        modified_df = dataset.copy()

        if data_name == 'adult':
            d = Adult()
        else:
            d = None

        # find the most influential records
        if importance_order is None:
            if fraction * len(
                    modified_df) >= cluster_length:  # if we are trying to flip more than there is n the cluster
                # increase the cluster
                cluster_length = int(np.ceil(fraction * len(modified_df)))
                print("!Warning: all data values in the cluster are being flipped.")
            cluster = self.find_influential_records(d, cluster_size=cluster_length)
        else:
            print('Trying to delete ', fraction)

        # read from the cluster factor * attack_strength
        # among these records remove attack strength
        cluster_factor = 1.5
        if cluster_factor*fraction < 1.0:
            cluster = importance_order.head(int(cluster_factor*fraction*len(d.dataframe)))
        else:
            cluster = importance_order
        print("Cluster size: " + str(len(cluster)))
        # from the cluster remove absolute number of records fraction*len(dataset)
        cluster_fraction = 1 / cluster_factor
        subset = cluster.sample(frac=cluster_fraction, random_state=random_state)
        # subset are the records that need to be removed
        print("Influential record removal attack on " + str(len(subset)) + " records.")
        modified_df = modified_df.drop(subset.index, axis=0, errors='ignore')

        return modified_df

