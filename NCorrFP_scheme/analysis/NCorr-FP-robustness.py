import sys
sys.path.insert(0, '../dissertation')  # make the script standalone for running on server

from datasets import CovertypeSample

import argparse
from itertools import product
from datetime import datetime
from attacks.collusion import *


def collusion(dataset='covertype-sample', save_results='robustness-collusion'):
    """
    Perform analysis on robustness to collusion attack for NCorr-FP
    Args:
        dataset: dataset name; it is expected that it is predefined in the module dataset
        save_results: name extension for the file; in case it is None, the results are not ave into a file

    Returns: pd.DataFrame of the collusion results

    """
    print('NCorr-FP: Robustness - Collusion.\nData: {}'.format(dataset))
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    save_results += f"_{dataset}_{timestamp}.csv"  # out file

    # --- Read data --- #
    data = None
    if dataset == 'covertype-sample':
        data = CovertypeSample()
    if data is None:
        exit('Please provide a valid dataset name ({})'.format(datasets.__all__))
    print(data.dataframe.head(3))

    # --- Define parameters --- #
    fp_params = {'gamma': [32],
                 'k': [300],
                 'fingerprint_length': [128],# 128
                 'n_recipients': [20],
                 'sk': [100]}  # 128
    collusion_params = {'n_colluders': [2, 3, 5, 10],
                        'strategy': ['avg', 'random', 'random_flip'],  # random (where they just choose a random new value)#
                        'threshold': [1]} # 1.2, 0.8], for this I dont need to run it multiple times, just save the accusation scores
    n_experiments = 10
    # --- Initialise the results --- #
    results = {key: [] for key in list(fp_params.keys()) +
                                  list(collusion_params.keys()) +
                                  ['embedding_ratio', 'colluders', 'accusation_scores',
                                   'total_accusations', 'hit_abs', 'hit_rate', 'false_accusation_abs', 'false_accusation_rate', 'recall']}

    # --- Run the collusions --- #
    combinations = list(product(*fp_params.values()))
    folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), dataset + '-fp')
    # Iterate through finerprint parameter settings
    for combination in combinations:

        param = dict(zip(fp_params.keys(), combination))

        # Iterate through collusion parameter settings
        collusion_comb = list(product(*collusion_params.values()))
        for collusion_combination in collusion_comb:
            c_param = dict(zip(collusion_params.keys(), collusion_combination))
            # Run n random experiments with each collusion setting
            for i in range(n_experiments):
                colluders = np.random.choice([i for i in range(param['n_recipients'])],
                                             c_param['n_colluders'], replace=False)
                print(colluders)

                files_to_collude = []
                for c in colluders:
                    param_string = '_'.join(f"{key}{value}" for key, value in param.items())
                    file_name = data.name + "_" + param_string + '_id' + str(c) + '.csv'
                    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), data.name + "-fp", file_name)
                    files_to_collude.append(file_path)
                    # todo append the results with colluder ids

                # Run collusion
                if c_param['strategy'] == 'avg':
                    colluded_ds = collude_datasetes_by_avg(files_to_collude)
                elif c_param['strategy'] == 'random':
                    colluded_ds = collude_datasetes_by_random(files_to_collude)
                elif c_param['strategy'] == 'random_flip':
                    colluded_ds = collude_datasets_by_random_and_flipping(files_to_collude)
                else:
                    exit(f"Invalid collusion strategy ({collusion_params['strategy']})")

                # Run detection
                scheme = NCorrFP(gamma=param['gamma'], fingerprint_bit_length=param['fingerprint_length'], k=param['k'],
                                 number_of_recipients=param['n_recipients'], fingerprint_code_type='hash')
                detected_fp, votes, suspect_probvec = scheme.detection(colluded_ds, secret_key=param['sk'],
                                                                       primary_key='Id',
                                                                       correlated_attributes=data.correlated_attributes,
                                                                       original_columns=list(data.columns))
                print(suspect_probvec)
                accusation, scores = scheme.detect_colluders(detected_fp, secret_key=param['sk'], threshold=c_param['threshold'])
                print(accusation, scores)

                # Record the parameters for the results
                for key, values in param.items():
                    results[key].append(values)
                results['embedding_ratio'].append(1.0 / param['gamma'])
                for key, values in c_param.items():
                    results[key].append(values)
                results['colluders'].append(colluders)
                results['accusation_scores'].append(scores)

                # --- Record the collusion measures --- #
                # --- Number of total accused colluders --- #
                print(f'Total accusations: {len(accusation)}')
                results['total_accusations'].append(len(accusation))

                # --- Total number of correctly accused colluders
                hits_abs = len(set(accusation) & set(colluders))
                print(f'Hits (abs): {hits_abs}')
                results['hit_abs'].append(hits_abs)

                # --- Hit rate: Correctly accused colluders / total accused (ideally, 1.0) --- #
                hit_rate = hits_abs / len(accusation)
                print(f'Hit rate: {hit_rate}')
                results['hit_rate'].append(hit_rate)

                # --- Total number of falsely accused recipients --- #
                print(f'False accusation (abs): {len(accusation) - hits_abs}')
                results['false_accusation_abs'].append(len(accusation) - hits_abs)

                # --- False accusation rate: falsely accused / total accused (ideally, 0) --- #
                print(f'False accusation rate: {1.0 - hit_rate}')
                results['false_accusation_rate'].append(1.0-hit_rate)

                # --- Recall: correctly accused / total colluders (how many colluders got accused?; ideally, 1.0) --- #
                recall = 1.0 - ((len(colluders) - hits_abs) / len(colluders))
                print(f'Recall: {recall}')
                results['recall'].append(recall)

    results_frame = pd.DataFrame(results)
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

    collusion(args.dataset)
    # todo: 1-user attacks
