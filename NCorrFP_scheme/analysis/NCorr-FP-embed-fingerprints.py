import sys
sys.path.insert(0, '../dissertation')  # make the script standalone for running on server

import datasets
from datasets import CovertypeSample
from NCorrFP_scheme.NCorrFP import NCorrFP

import argparse
import os
from itertools import product


def embed_fingerprints(data, params):
    """
    Embeds the fingerprints into the data for the experiments.
    Args:
        data (Dataset): Dataset instance of a dataset to fingerprint.
        params (dict): A dictionary of fingerprinting parameters

    Returns (string): path to the directory of experiment datasets

    """
    # Generate all parameter combinations
    combinations = list(product(*params.values()))
    # Iterate through parameter combinations
    for combination in combinations:
        param = dict(zip(params.keys(), combination))

        param_string = '_'.join(f"{key}{value}" for key, value in param.items())
        file_name = data.name + "_" + param_string + '.csv'
        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), data.name + "-fp", file_name)

        # skip the file if it already exists
        if not os.path.exists(file_path):
            scheme = NCorrFP(gamma=param['gamma'], fingerprint_bit_length=param['fingerprint_length'], k=param['k'],
                             number_of_recipients=param['n_recipients'], fingerprint_code_type='hash')

            scheme.insertion(data, secret_key=param['sk'], recipient_id=param['id'],
                             correlated_attributes=data.correlated_attrs, save_computation=True, outfile=file_path)
        else:
            print("- File already exists. {}".format(file_name))
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), data.name + "-fp")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Load a dataset with optional configurations.")
    # Required positional argument
    parser.add_argument("dataset", type=str, help="Dataset name.")
    # Parse arguments
    args = parser.parse_args()

    # --- Read data --- #
    data = None
    if args.dataset == 'covertype-sample':
        data = CovertypeSample()
    if data is None:
        exit('Please provide a valid dataset name ({})'.format(datasets.__all__))
    print(data.dataframe.head(3))

    # --- Define parameters --- #
    params = {'gamma': [2, 4, 8, 16, 32],
              'k': [300], #, 500],
              'fingerprint_length': [512],#, 128, 256],  # , 128, 256],
              'n_recipients': [20],
              'sk': [100 + i for i in range(10)],
              'id': [0]} #i for i in range(20)]}  # + i for i in range(10)]}  # 10)]}  # #sk-s = #experiments

    embed_fingerprints(data, params)
