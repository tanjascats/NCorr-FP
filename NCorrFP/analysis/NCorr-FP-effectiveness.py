# perform analysis on effectiveness of NCorr FP
# 1. Vote errors
# 2. Fingerprint bit errors
# 3. True negatives confidence
import sys
sys.path.insert(0, '../dissertation')  # make the script standalone for running on server

import datasets
from datasets import CovertypeSample, Adult
from NCorrFP.NCorrFP import NCorrFP

import pandas as pd
import argparse
import os
from itertools import product
from datetime import datetime


def vote_error_rate(votes, fingerprint):
    """
        Calculate the rate of wrong votes based on the fingerprint and votes array.

        Args:
        - fingerprint (list of int): A list of bits (0 or 1) representing the fingerprint.
        - votes (list of lists): A list where each element is a list of two integers representing
                                 the votes for 0 and 1 at each position in the fingerprint.

        Returns:
        - float: The rate total number of wrong votes / total votes.
        """
    wrong_votes = 0
    total_votes = 0
    # Iterate over each position in the fingerprint and corresponding votes
    for i, bit in enumerate(fingerprint):
        # If the bit is 0, count votes for 1 as wrong votes; otherwise, count votes for 0 as wrong votes
        if bit == 0:
            wrong_votes += votes[i][1]  # votes for 1 are wrong when bit is 0
            total_votes += wrong_votes + votes[i][0]
        else:
            wrong_votes += votes[i][0]  # votes for 0 are wrong when bit is 1
            total_votes += wrong_votes + votes[i][1]

    return wrong_votes / total_votes


def effectiveness(dataset='covertype-sample', save_results='effectiveness'):
    """
    Perform analysis on effectiveness of NCorr-FP
        1. Vote errors
        2. Fingerprint bit errors (for the correct recipient)
        3. True negatives confidence (matching with wrong fingerprints)
    Args:
        dataset: dataset name; it's expected that it's predefined in the module dataset
        save_results: name extension for the file; in case it's None, the results are not saved into a file

    Returns: pd.DataFrame of the effectiveness results

    """
    print('NCorr-FP: Effectiveness.\nData: {}'.format(dataset))
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    save_results += f"_{dataset}_{timestamp}.csv"  # out file

    # --- Read data --- #
    data = None
    if dataset == 'covertype-sample':
        data = CovertypeSample()
    elif dataset == 'adult':
        data = Adult()
    if data is None:
        exit('Please provide a valid dataset name ({})'.format(datasets.__all__))
    print(data.dataframe.head(3))

    # --- Define parameters --- #
    params = {'gamma': [32],#, 4, 8, 16, 32],
              'k': [300, 450],
              'fingerprint_length': [128, 256, 512], #, 128, 256],
              'n_recipients': [20],
              'sk': [100 + i for i in range(10)], #10)]}  # #sk-s = #experiments
              'id': [0],#i for i in range(20)],
              'code': ['tardos', 'hash']
              }

    # --- Initialise the results --- #
    results = {key: [] for key in list(params.keys()) + ['embedding_ratio', 'vote_error', 'tp', 'tn']}

    # --- Run the detection and count: --- #
    #   1. wrong votes
    #   2. fingerprint detection confidence
    #   3. detection confidence for the wrong recipients
    combinations = list(product(*params.values()))
    # check if all the files are there
    folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), dataset + '-fp')
    file_count = len([file for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file))])
    if file_count < len(combinations):
        print("WARNING: It seems that some fingerprinted datasets are missing.")
    # Iterate through parameter combinations
    for combination in combinations:
        param = dict(zip(params.keys(), combination))

        param_string = '_'.join(f"{key}{value}" for key, value in param.items())
        file_name = data.name + "_" + param_string + '.csv'
        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), data.name + "-fp", file_name)

        # Check if it's a file (skip folders)
        if os.path.isfile(file_path):
            print(f"Reading file: {file_name}")

            # Open and read the file (assuming csv files here)
            try:
                fingerprinted_data = pd.read_csv(file_path)
            except pd.errors.EmptyDataError:
                print(f"Empty file: {file_name}")
                continue
            scheme = NCorrFP(gamma=param['gamma'], fingerprint_bit_length=param['fingerprint_length'], k=param['k'],
                             number_of_recipients=param['n_recipients'], fingerprint_code_type=param['code'])
            detected_fp, votes, suspect_probvec = scheme.detection(fingerprinted_data, secret_key=param['sk'],
                                                                   primary_key='Id',
                                                                   correlated_attributes=data.correlated_attributes,
                                                                   original_columns=list(data.columns))
            real_fp = scheme.create_fingerprint(recipient_id=param['id'], secret_key=param['sk'])

            # record the parameters for the results
            for key, values in param.items():
                results[key].append(values)
            results['embedding_ratio'].append(1.0 / param['gamma'])

            # --- Count the rate of wrong votes (ideally, 0) --- #
            vote_error_rates = vote_error_rate(votes, real_fp)
            results['vote_error'].append(vote_error_rates)

            # --- Get the fingerprint extraction confidence for the correct recipient (ideally, 1.0) --- #
            true_pos_confidence = suspect_probvec[param['id']]
            results['tp'].append(true_pos_confidence)

            # --- Get the avg confidence for all the wrong recipients (ideally, ~0.5) --- #
            wrong_recipient_confs = [value for key, value in suspect_probvec.items() if key != param['id']]
            # true_neg_confidence = np.mean(wrong_recipient_confs)  # np.std(wrong_recipient_confs)
            results['tn'].append(wrong_recipient_confs)

    results_frame = pd.DataFrame(results).explode('tn', ignore_index=True)
    if save_results is not None:
        results_frame.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), save_results), index=False)
        print(f"Results saved in {os.path.join(os.path.dirname(os.path.abspath(__file__)), save_results)}")
    return results_frame


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Load a dataset with optional configurations.")
    # Required positional argument
    parser.add_argument("dataset", type=str, help="Dataset name.")
    # Parse arguments
    args = parser.parse_args()

    effectiveness(args.dataset)
