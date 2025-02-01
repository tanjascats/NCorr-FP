import sys
from pprint import pprint
import pandas as pd

sys.path.insert(0, '../dissertation')  # make the script standalone for running on server

from datasets import Adult, CovertypeSample

import argparse
from itertools import product
from datetime import datetime
from attacks.collusion import *
from attacks.horizontal_subset_attack import *
from attacks.vertical_subset_attack import *
from attacks.bit_flipping_attack import *
import utils
from NCorr_FP_fidelity import *


def collusion(dataset='adult', save_results='robustness-collusion'):
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
    if dataset == 'adult':
        data = Adult()
    if data is None:
        exit('Please provide a valid dataset name ({})'.format(datasets.__all__))
    print(data.dataframe.head(3))

    # --- Define parameters --- #
    fp_params = {'gamma': [8], # 8
                 'k': [300],
                 'fingerprint_length': [128, 256, 512, 1024, 64],#128, 256, 512, 1024, 64],# 128
                 'n_recipients': [20],
                 'sk': [100]}
    collusion_params = {'n_colluders': [2, 3, 5, 10],#, 3],#, 5, 10],
                        'strategy': ['random_flip'], #, 'avg', 'random', 'random_flip'],  # random (where they just choose a random new value)#
                        'threshold': [1]} # 1.2, 0.8], for this I dont need to run it multiple times, just save the accusation scores
    n_experiments = 10
    # --- Initialise the results --- #
    results = {key: [] for key in list(fp_params.keys()) +
                                  list(collusion_params.keys()) +
                                  ['embedding_ratio', 'colluders', 'accusation_scores',
                                   'total_accusations', 'hit_abs', 'hit_rate', 'false_accusation_abs',
                                   'false_accusation_rate', 'recall', 'colluders']}

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
                np.random.seed(int(datetime.now().timestamp()))  # needed because another seed is set elsewhere
                colluders = np.random.choice([i for i in range(param['n_recipients'])],
                                             c_param['n_colluders'], replace=False)
                print('Colluders: ', colluders)

                files_to_collude = []
                for c in colluders:
                    param_string = '_'.join(f"{key}{value}" for key, value in param.items())
                    file_name = data.name + "_" + param_string + '_id' + str(c) + '_codetardos.csv'
                    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fp_datasets', 'NCorrFP',
                                             data.name + "-fp", file_name)
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
                                 number_of_recipients=param['n_recipients'], fingerprint_code_type='tardos')
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

    print(results)
    results_frame = pd.DataFrame(results)
    if save_results is not None:
        results_frame.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), save_results), index=False)
        print(f"Results saved in {os.path.join(os.path.dirname(os.path.abspath(__file__)), save_results)}")
    return results_frame


def vertical_attack(dataset='adult', save_results='robustness-vertical'):
    print('NCorr-FP: Robustness - Verical subset attack.\nData: {}'.format(dataset))
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    save_results += f"_{dataset}_{timestamp}.csv"  # out file

    # --- Read data --- #
    data = None
    if dataset == 'covertype-sample':
        data = CovertypeSample()
    if dataset == 'adult':
        data = Adult()
    if data is None:
        exit('Please provide a valid dataset name ({})'.format(datasets.__all__))
    print(data.dataframe.head(3))
    n_columns = len(data.dataframe.drop(['Id'], axis=1).columns)

    correlated_pairs_dict = utils.extract_mutually_correlated_pairs(data.dataframe,
                                                                    threshold_num=0.70, threshold_cat=0.45,
                                                                    threshold_numcat=0.14)
    correlated_pairs_string = ["_".join(list(a)) for a in list(correlated_pairs_dict.keys())]

    fp_params = {'gamma': [32],  # 4, 8, 16, 32],
                 'k': [300, 450],
                 'fingerprint_length': [128],  # 128
                 'n_recipients': [20],
                 'sk': [100 + i for i in range(1, 10)], # todo: revert to range(10)
                 'id': [0],
                 'code': ['tardos']}  # 128
    attack_strength = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    n_experiments = 3
    results = {key: [] for key in ['gamma', 'k', 'fingerprint_length', 'n_recipients', 'sk', 'id', 'code',
                                   'embedding_ratio',
                                   'experiment_no', 'attack_strength',
                                   'detection_confidence',
                                   'accuracy', 'hellinger_distance', 'hellinger_distance_remaining'] + correlated_pairs_string}

    combinations = list(product(*fp_params.values()))
    # check if all the files are there
    folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fp_datasets', 'NCorrFP', dataset + '-fp')
    file_count = len([file for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file))])
    if file_count < len(combinations):
        print("WARNING: It seems that some fingerprinted datasets are missing. Total file count: ", file_count)
    # Iterate through parameter combinations (datasets)
    for combination in combinations:
        param = dict(zip(fp_params.keys(), combination))

        param_string = '_'.join(f"{key}{value}" for key, value in param.items())
        file_name = data.name + "_" + param_string + '.csv'
        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fp_datasets', 'NCorrFP',
                                 data.name + "-fp", file_name)
        # print('Trying file ', file_path)
        # Check if it's a file (skip folders)
        if os.path.isfile(file_path):
            print(f"Reading file: {file_name}")

            # Open and read the file (assuming csv files here)
            try:
                fingerprinted_data = pd.read_csv(file_path)
            except pd.errors.EmptyDataError:
                print(f"Empty file: {file_name}")
                continue
            for strength in attack_strength:
                for exp in range(n_experiments):
                    scheme = NCorrFP(gamma=param['gamma'], fingerprint_bit_length=param['fingerprint_length'],
                                     k=param['k'],
                                     number_of_recipients=param['n_recipients'], fingerprint_code_type=param['code'])
                    attack = VerticalSubsetAttack()
                    attack_columns = round(strength*n_columns)
                    release_data = attack.run_random(dataset=fingerprinted_data, number_of_columns=attack_columns, random_state=2 * param['sk'] + exp)
                    detected_fp, votes, suspect_probvec = scheme.detection(release_data, secret_key=param['sk'],
                                                                           primary_key='Id',
                                                                           correlated_attributes=data.correlated_attributes,
                                                                           original_columns=list(data.columns))

                    # record the parameters for the results
                    for key, values in param.items():
                        results[key].append(values)
                    results['embedding_ratio'].append(1.0 / param['gamma'])
                    results['experiment_no'].append(exp)
                    results['attack_strength'].append(strength)
                    results['detection_confidence'].append(suspect_probvec[param['id']])

                    print(f"--Detection confidence = ", suspect_probvec[param['id']])
                    # Fidelity of the released data
                    accuracy = 1.0 - (attack_columns / len(data.dataframe.columns))
                    results['accuracy'].append(accuracy)
                    print("--Accuracy ", accuracy)

                    # Hellinger distance of missing attributes is the maximum value of 1
                    common_columns = list(set(data.dataframe.columns).intersection(set(release_data.columns)))
                    missing_columns = list(set(list(data.dataframe.columns)) - set(list(release_data.columns)))
                    hellinger_distances = hellinger_distance(data.dataframe[common_columns], release_data[common_columns])
                    # Hellinger averaged over non-deleted columns
                    hellinger_dist = np.mean(list(hellinger_distances.values()))
                    results['hellinger_distance_remaining'].append(hellinger_dist)
                    print("--Hellinger (remaining) ", hellinger_dist)
                    # Hellinger averaged with 1.0 for missing columns
                    for missing_col in missing_columns:
                        hellinger_distances[missing_col] = 1.0
                    hellinger_dist = np.mean(list(hellinger_distances.values()))
                    results['hellinger_distance'].append(hellinger_dist)
                    print("--Hellinger ", hellinger_dist)

                    # KL divergence upper limit is infinity so we can't deal with this
                    #kl_div = np.mean(list(kl_divergence_kde(data.dataframe, release_data).values()))
                    #results['kl_divergence'].append(kl_div)
                    #print("--KL divergence: ", kl_div)

                    numerical_columns = data.dataframe.select_dtypes(include=['number'])
                    categorical_columns = data.dataframe.select_dtypes(include=['object', 'category'])
                    for i, pair in enumerate(correlated_pairs_dict.keys()):
                        # If one of the attributes is missing we write nan
                        if pair[0] not in release_data.columns or pair[1] not in release_data.columns:
                            results[correlated_pairs_string[i]].append(None)
                        else:
                            if pair[0] in numerical_columns and pair[1] in numerical_columns:
                                # for two numerical calculate pearson's
                                fp_corr = release_data[pair[0]].corr(release_data[pair[1]])
                                delta_corr = abs((correlated_pairs_dict[pair] - fp_corr) / correlated_pairs_dict[pair])
                                results[correlated_pairs_string[i]].append(delta_corr)
                            elif pair[0] in categorical_columns and pair[1] in categorical_columns:
                                # for categorical calculate Cramer's V
                                fp_v = utils.cramers_v(release_data[pair[0]], release_data[pair[1]])
                                delta_v = abs((correlated_pairs_dict[pair] - fp_v) / correlated_pairs_dict[pair])
                                results[correlated_pairs_string[i]].append(delta_v)
                            else:
                                # for categorical x numerical calculate eta squared
                                if pair[0] in categorical_columns:
                                    cat_col = pair[0]
                                    num_col = pair[1]
                                else:
                                    cat_col = pair[1]
                                    num_col = pair[0]
                                fp_eta = utils.eta_squared(release_data, cat_col, num_col)
                                delta_eta = abs((correlated_pairs_dict[pair] - fp_eta) / correlated_pairs_dict[pair])
                                results[correlated_pairs_string[i]].append(delta_eta)
        else:
            print(f"------------\nMissing file: {file_name}\n------------")
    print(results)
    results_frame = pd.DataFrame(results)
    if save_results is not None:
        results_frame.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), save_results), index=False)
        print(f"Results saved in {os.path.join(os.path.dirname(os.path.abspath(__file__)), save_results)}")

    return results_frame


def horizontal_attack(dataset='adult', save_results='robustness-horizontal'):
    print('NCorr-FP: Robustness - Horizontal subset attack.\nData: {}'.format(dataset))
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    save_results += f"_{dataset}_{timestamp}.csv"  # out file

    # --- Read data --- #
    data = None
    if dataset == 'covertype-sample':
        data = CovertypeSample()
    if dataset == 'adult':
        data = Adult()
    if data is None:
        exit('Please provide a valid dataset name ({})'.format(datasets.__all__))
    print(data.dataframe.head(3))

    correlated_pairs_dict = utils.extract_mutually_correlated_pairs(data.dataframe,
                                                                    threshold_num=0.70, threshold_cat=0.45, threshold_numcat=0.14)
    correlated_pairs_string = ["_".join(list(a)) for a in list(correlated_pairs_dict.keys())]

    fp_params = {'gamma': [32],#, 8, 16, 32],#, 4, 8, 16, 32],
                 'k': [300, 450],
                 'fingerprint_length': [128],  # 128
                 'n_recipients': [20],
                 'sk': [100 + i for i in range(1, 10)],# todo: revert to range(10)
                 'id': [0],
                 'code': ['tardos']}  # 128
    attack_strength = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    n_experiments = 3
    results = {key: [] for key in ['gamma', 'k', 'fingerprint_length', 'n_recipients', 'sk', 'id', 'code',
                                   'embedding_ratio',
                                   'experiment_no', 'attack_strength',
               'detection_confidence',
               'accuracy', 'hellinger_distance', 'kl_divergence'] + correlated_pairs_string}

    combinations = list(product(*fp_params.values()))
    # check if all the files are there
    folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fp_datasets', 'NCorrFP', dataset + '-fp')
    file_count = len([file for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file))])
    if file_count < len(combinations):
        print("WARNING: It seems that some fingerprinted datasets are missing. Total file count: ", file_count)
    # Iterate through parameter combinations (datasets)
    for combination in combinations:
        param = dict(zip(fp_params.keys(), combination))

        param_string = '_'.join(f"{key}{value}" for key, value in param.items())
        file_name = data.name + "_" + param_string + '.csv'
        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fp_datasets', 'NCorrFP',
                                 data.name + "-fp", file_name)
        # print('Trying file ', file_path)
        # Check if it's a file (skip folders)
        if os.path.isfile(file_path):
            print(f"Reading file: {file_name}")
            # Open and read the file (assuming csv files here)
            try:
                fingerprinted_data = pd.read_csv(file_path)
            except pd.errors.EmptyDataError:
                print(f"Empty file: {file_name}")
                continue
            for strength in attack_strength:
                for exp in range(n_experiments):
                    scheme = NCorrFP(gamma=param['gamma'], fingerprint_bit_length=param['fingerprint_length'], k=param['k'],
                                     number_of_recipients=param['n_recipients'], fingerprint_code_type=param['code'])
                    attack = HorizontalSubsetAttack()
                    release_data = attack.run(dataset=fingerprinted_data, fraction=1.0-strength, random_state=2*param['sk']+exp)
                    detected_fp, votes, suspect_probvec = scheme.detection(release_data, secret_key=param['sk'],
                                                                           primary_key='Id',
                                                                           correlated_attributes=data.correlated_attributes,
                                                                           original_columns=list(data.columns))

                    # record the parameters for the results
                    for key, values in param.items():
                        results[key].append(values)
                    results['embedding_ratio'].append(1.0 / param['gamma'])
                    results['experiment_no'].append(exp)
                    results['attack_strength'].append(strength)
                    results['detection_confidence'].append(suspect_probvec[param['id']])

                    print(f"--Detection confidence = ", suspect_probvec[param['id']])
                    # Fidelity of the released data
                    accuracy = 1.0 - strength
                    results['accuracy'].append(accuracy)
                    print("--Accuracy ", accuracy)
                    hellinger_dist = np.mean(list(hellinger_distance(data.dataframe, release_data).values()))
                    results['hellinger_distance'].append(hellinger_dist)
                    print("--Hellinger ", hellinger_dist)
                    kl_div = np.mean(list(kl_divergence_kde(data.dataframe, release_data).values()))
                    results['kl_divergence'].append(kl_div)
                    print("--KL divergence: ", kl_div)

                    numerical_columns = data.dataframe.drop(['Id'], axis=1).select_dtypes(include=['number'])
                    categorical_columns = data.dataframe.drop(['Id'], axis=1).select_dtypes(include=['object', 'category'])
                    for i, pair in enumerate(correlated_pairs_dict.keys()):
                        if pair[0] in numerical_columns and pair[1] in numerical_columns:
                            # for two numerical calculate pearson's
                            fp_corr = release_data[pair[0]].corr(release_data[pair[1]])
                            delta_corr = abs((correlated_pairs_dict[pair] - fp_corr) / correlated_pairs_dict[pair])
                            results[correlated_pairs_string[i]].append(delta_corr)
                        elif pair[0] in categorical_columns and pair[1] in categorical_columns:
                            # for categorical calculate Cramer's V
                            fp_v = utils.cramers_v(release_data[pair[0]], release_data[pair[1]])
                            delta_v = abs((correlated_pairs_dict[pair] - fp_v) / correlated_pairs_dict[pair])
                            results[correlated_pairs_string[i]].append(delta_v)
                        else:
                            # for categorical x numerical calculate eta squared
                            if pair[0] in categorical_columns:
                                cat_col = pair[0]
                                num_col = pair[1]
                            else:
                                cat_col = pair[1]
                                num_col = pair[0]
                            fp_eta = utils.eta_squared(release_data, cat_col, num_col)
                            delta_eta = abs((correlated_pairs_dict[pair] - fp_eta) / correlated_pairs_dict[pair])
                            results[correlated_pairs_string[i]].append(delta_eta)
        else:
            print(f"\n------------\nMissing file: {file_name}\n------------\n")
    print(results)
    results_frame = pd.DataFrame(results)
    if save_results is not None:
        results_frame.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), save_results), index=False)
        print(f"Results saved in {os.path.join(os.path.dirname(os.path.abspath(__file__)), save_results)}")
    return results_frame


def flipping_attack(dataset='adult', save_results='robustness-flipping'):
    print('NCorr-FP: Robustness - Flipping attack.\nData: {}'.format(dataset))
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    save_results += f"_{dataset}_{timestamp}.csv"  # out file

    # --- Read data --- #
    data = None
    if dataset == 'covertype-sample':
        data = CovertypeSample()
    if dataset == 'adult':
        data = Adult()
    if data is None:
        exit('Please provide a valid dataset name ({})'.format(datasets.__all__))
    print(data.dataframe.head(3))

    correlated_pairs_dict = utils.extract_mutually_correlated_pairs(data.dataframe,
                                                                    threshold_num=0.70, threshold_cat=0.45,
                                                                    threshold_numcat=0.14)
    correlated_pairs_string = ["_".join(list(a)) for a in list(correlated_pairs_dict.keys())]

    fp_params = {'gamma': [32],#, 16, 32],  # 4, 8, 16, 32],
                 'k': [300, 450],
                 'fingerprint_length': [128],  # 128
                 'n_recipients': [20],
                 'sk': [100 + i for i in range(1, 10)],  # todo: revert to range(10)
                 'id': [0],
                 'code': ['tardos']}  # 128
    print("Parameters: "); pprint(fp_params)
    attack_strength = [0.5, 0.6, 0.7, 0.8, 0.9]
    print("Attack strength: ", attack_strength)
    n_experiments = 3
    results = {key: [] for key in ['gamma', 'k', 'fingerprint_length', 'n_recipients', 'sk', 'id', 'code',
                                   'embedding_ratio',
                                   'experiment_no', 'attack_strength',
                                   'detection_confidence',
                                   'accuracy', 'hellinger_distance', 'kl_divergence'] + correlated_pairs_string}

    combinations = list(product(*fp_params.values()))
    # check if all the files are there
    folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fp_datasets', 'NCorrFP', dataset + '-fp')
    file_count = len([file for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file))])
    if file_count < len(combinations):
        print("WARNING: It seems that some fingerprinted datasets are missing. Total file count: ", file_count)
    # Iterate through parameter combinations (datasets)
    for combination in combinations:
        param = dict(zip(fp_params.keys(), combination))

        param_string = '_'.join(f"{key}{value}" for key, value in param.items())
        file_name = data.name + "_" + param_string + '.csv'
        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fp_datasets', 'NCorrFP',
                                 data.name + "-fp", file_name)
        # print('Trying file ', file_path)
        # Check if it's a file (skip folders)
        if os.path.isfile(file_path):
            print(f"Reading file: {file_name}")

            # Open and read the file (assuming csv files here)
            try:
                fingerprinted_data = pd.read_csv(file_path)
            except pd.errors.EmptyDataError:
                print(f"Empty file: {file_name}")
                continue
            for strength in attack_strength:
                for exp in range(n_experiments):
                    scheme = NCorrFP(gamma=param['gamma'], fingerprint_bit_length=param['fingerprint_length'],
                                     k=param['k'],
                                     number_of_recipients=param['n_recipients'], fingerprint_code_type=param['code'])
                    attack = FlippingAttack()
                    release_data = attack.run(dataset=fingerprinted_data, fraction=strength,
                                              random_state=2 * param['sk'] + exp)
                    detected_fp, votes, suspect_probvec = scheme.detection(release_data, secret_key=param['sk'],
                                                                           primary_key='Id',
                                                                           correlated_attributes=data.correlated_attributes,
                                                                           original_columns=list(data.columns))

                    # record the parameters for the results
                    for key, values in param.items():
                        results[key].append(values)
                    results['embedding_ratio'].append(1.0 / param['gamma'])
                    results['experiment_no'].append(exp)
                    results['attack_strength'].append(strength)
                    results['detection_confidence'].append(suspect_probvec[param['id']])

                    print(f"--Detection confidence = ", suspect_probvec[param['id']])
                    # Fidelity of the released data
                    accuracy = 1.0 - strength
                    results['accuracy'].append(accuracy)
                    print("--Accuracy ", accuracy)
                    hellinger_dist = np.mean(list(hellinger_distance(data.dataframe, release_data).values()))
                    results['hellinger_distance'].append(hellinger_dist)
                    print("--Hellinger ", hellinger_dist)
                    kl_div = np.mean(list(kl_divergence_kde(data.dataframe, release_data).values()))
                    results['kl_divergence'].append(kl_div)
                    print("--KL divergence: ", kl_div)

                    numerical_columns = data.dataframe.drop(['Id'], axis=1).select_dtypes(include=['number'])
                    categorical_columns = data.dataframe.drop(['Id'], axis=1).select_dtypes(
                        include=['object', 'category'])
                    for i, pair in enumerate(correlated_pairs_dict.keys()):
                        if pair[0] in numerical_columns and pair[1] in numerical_columns:
                            # for two numerical calculate pearson's
                            fp_corr = release_data[pair[0]].corr(release_data[pair[1]])
                            delta_corr = abs((correlated_pairs_dict[pair] - fp_corr) / correlated_pairs_dict[pair])
                            results[correlated_pairs_string[i]].append(delta_corr)
                        elif pair[0] in categorical_columns and pair[1] in categorical_columns:
                            # for categorical calculate Cramer's V
                            fp_v = utils.cramers_v(release_data[pair[0]], release_data[pair[1]])
                            delta_v = abs((correlated_pairs_dict[pair] - fp_v) / correlated_pairs_dict[pair])
                            results[correlated_pairs_string[i]].append(delta_v)
                        else:
                            # for categorical x numerical calculate eta squared
                            if pair[0] in categorical_columns:
                                cat_col = pair[0]
                                num_col = pair[1]
                            else:
                                cat_col = pair[1]
                                num_col = pair[0]
                            fp_eta = utils.eta_squared(release_data, cat_col, num_col)
                            delta_eta = abs((correlated_pairs_dict[pair] - fp_eta) / correlated_pairs_dict[pair])
                            results[correlated_pairs_string[i]].append(delta_eta)
        else:
            print(f"------------\nMissing file: {file_name}\n------------")
    print(results)
    results_frame = pd.DataFrame(results)
    if save_results is not None:
        results_frame.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), save_results), index=False)
        print(f"Results saved in {os.path.join(os.path.dirname(os.path.abspath(__file__)), save_results)}")

    return results_frame


def cluster_flipping(dataset='adult', save_results='robustness-clusterflipping'):
    print('NCorr-FP: Robustness - Cluster flipping attack.\nData: {}'.format(dataset))
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    save_results += f"_{dataset}_{timestamp}.csv"  # out file

    # --- Read data --- #
    data = None
    if dataset == 'covertype-sample':
        data = CovertypeSample()
    if dataset == 'adult':
        data = Adult()
    if data is None:
        exit('Please provide a valid dataset name ({})'.format(datasets.__all__))
    print(data.dataframe.head(3))

    correlated_pairs_dict = utils.extract_mutually_correlated_pairs(data.dataframe,
                                                                    threshold_num=0.70, threshold_cat=0.45,
                                                                    threshold_numcat=0.14)
    correlated_pairs_string = ["_".join(list(a)) for a in list(correlated_pairs_dict.keys())]

    fp_params = {'gamma': [32],#8, 16, 32],  # 4, 8, 16, 32],
                 'k': [300, 450],
                 'fingerprint_length': [128],  # 128
                 'n_recipients': [20],
                 'sk': [100 + i for i in range(10)],
                 'id': [0],
                 'code': ['tardos']}  # 128
    attack_strength = [0.3, 0.35, 0.4, 0.45, 0.5]#, 0.4, 0.45, 0.5, 0.55]
    n_experiments = 3
    results = {key: [] for key in ['gamma', 'k', 'fingerprint_length', 'n_recipients', 'sk', 'id', 'code',
                                   'embedding_ratio',
                                   'experiment_no', 'attack_strength',
                                   'detection_confidence',
                                   'accuracy', 'hellinger_distance', 'kl_divergence'] + correlated_pairs_string}

    combinations = list(product(*fp_params.values()))
    # check if all the files are there
    folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fp_datasets', 'NCorrFP', dataset + '-fp')
    file_count = len([file for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file))])
    if file_count < len(combinations):
        print("WARNING: It seems that some fingerprinted datasets are missing. Total file count: ", file_count)
    # Iterate through parameter combinations (datasets)
    for combination in combinations:
        param = dict(zip(fp_params.keys(), combination))

        param_string = '_'.join(f"{key}{value}" for key, value in param.items())
        file_name = data.name + "_" + param_string + '.csv'
        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fp_datasets', 'NCorrFP',
                                 data.name + "-fp", file_name)
        # print('Trying file ', file_path)
        # Check if it's a file (skip folders)
        if os.path.isfile(file_path):
            print(f"Reading file: {file_name}")

            # Open and read the file (assuming csv files here)
            try:
                fingerprinted_data = pd.read_csv(file_path)
            except pd.errors.EmptyDataError:
                print(f"Empty file: {file_name}")
                continue
            buffer = 1.25
            for strength in attack_strength:
                if buffer*strength <= 1:
                    cluster = pd.read_csv('cluster_g1_k325_sk999.csv').head(int(buffer*strength*len(data.dataframe)))
                    cluster.index = cluster['Id']
                else:
                    cluster = pd.read_csv('cluster_g1_k325_sk999.csv')
                    cluster.index = cluster['Id']
                for exp in range(n_experiments):
                    scheme = NCorrFP(gamma=param['gamma'], fingerprint_bit_length=param['fingerprint_length'],
                                     k=param['k'],
                                     number_of_recipients=param['n_recipients'], fingerprint_code_type=param['code'])
                    attack = InfluentialRecordFlippingAttack()
                    release_data, c = attack.run(dataset=fingerprinted_data, fraction=strength, cluster=cluster,
                                              random_state=2 * param['sk'] + exp)
                    detected_fp, votes, suspect_probvec = scheme.detection(release_data, secret_key=param['sk'],
                                                                           primary_key='Id',
                                                                           correlated_attributes=data.correlated_attributes,
                                                                           original_columns=list(data.columns))

                    # record the parameters for the results
                    for key, values in param.items():
                        results[key].append(values)
                    results['embedding_ratio'].append(1.0 / param['gamma'])
                    results['experiment_no'].append(exp)
                    results['attack_strength'].append(strength)
                    results['detection_confidence'].append(suspect_probvec[param['id']])

                    print(f"--Detection confidence = ", suspect_probvec[param['id']])
                    # Fidelity of the released data
                    accuracy = 1.0 - strength
                    results['accuracy'].append(accuracy)
                    print("--Accuracy ", accuracy)
                    hellinger_dist = np.mean(list(hellinger_distance(data.dataframe, release_data).values()))
                    results['hellinger_distance'].append(hellinger_dist)
                    print("--Hellinger ", hellinger_dist)
                    kl_div = np.mean(list(kl_divergence_kde(data.dataframe, release_data).values()))
                    results['kl_divergence'].append(kl_div)
                    print("--KL divergence: ", kl_div)

                    numerical_columns = data.dataframe.drop(['Id'], axis=1).select_dtypes(include=['number'])
                    categorical_columns = data.dataframe.drop(['Id'], axis=1).select_dtypes(
                        include=['object', 'category'])
                    for i, pair in enumerate(correlated_pairs_dict.keys()):
                        if pair[0] in numerical_columns and pair[1] in numerical_columns:
                            # for two numerical calculate pearson's
                            fp_corr = release_data[pair[0]].corr(release_data[pair[1]])
                            delta_corr = abs((correlated_pairs_dict[pair] - fp_corr) / correlated_pairs_dict[pair])
                            results[correlated_pairs_string[i]].append(delta_corr)
                        elif pair[0] in categorical_columns and pair[1] in categorical_columns:
                            # for categorical calculate Cramer's V
                            fp_v = utils.cramers_v(release_data[pair[0]], release_data[pair[1]])
                            delta_v = abs((correlated_pairs_dict[pair] - fp_v) / correlated_pairs_dict[pair])
                            results[correlated_pairs_string[i]].append(delta_v)
                        else:
                            # for categorical x numerical calculate eta squared
                            if pair[0] in categorical_columns:
                                cat_col = pair[0]
                                num_col = pair[1]
                            else:
                                cat_col = pair[1]
                                num_col = pair[0]
                            fp_eta = utils.eta_squared(release_data, cat_col, num_col)
                            delta_eta = abs((correlated_pairs_dict[pair] - fp_eta) / correlated_pairs_dict[pair])
                            results[correlated_pairs_string[i]].append(delta_eta)
        else:
            print(f"------------\nMissing file: {file_name}\n------------")
    print(results)
    results_frame = pd.DataFrame(results)
    if save_results is not None:
        results_frame.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), save_results), index=False)
        print(f"Results saved in {os.path.join(os.path.dirname(os.path.abspath(__file__)), save_results)}")

    return results_frame


def cluster_horizontal(dataset='adult', save_results='robustness-clusterhorizontal'):
    print('NCorr-FP: Robustness - Cluster horizontal attack.\nData: {}'.format(dataset))
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    save_results += f"_{dataset}_{timestamp}.csv"  # out file

    # --- Read data --- #
    data = None
    if dataset == 'covertype-sample':
        data = CovertypeSample()
    if dataset == 'adult':
        data = Adult()
    if data is None:
        exit('Please provide a valid dataset name ({})'.format(datasets.__all__))
    print(data.dataframe.head(3))

    correlated_pairs_dict = utils.extract_mutually_correlated_pairs(data.dataframe,
                                                                    threshold_num=0.70, threshold_cat=0.45,
                                                                    threshold_numcat=0.14)
    correlated_pairs_string = ["_".join(list(a)) for a in list(correlated_pairs_dict.keys())]

    fp_params = {'gamma': [32],#8, 16, 32],  # 4, 8, 16, 32],
                 'k': [300, 450],
                 'fingerprint_length': [128],  # 128
                 'n_recipients': [20],
                 'sk': [100 + i for i in range(10)],
                 'id': [0],
                 'code': ['tardos']}  # 128
    attack_strength = [0.9] #, 0.7, 0.8, 0.9]
    cluster_factor = 1.5
    n_experiments = 3
    results = {key: [] for key in ['gamma', 'k', 'fingerprint_length', 'n_recipients', 'sk', 'id', 'code',
                                   'embedding_ratio',
                                   'experiment_no', 'attack_strength',
                                   'detection_confidence',
                                   'accuracy', 'hellinger_distance', 'kl_divergence'] + correlated_pairs_string}

    combinations = list(product(*fp_params.values()))
    # check if all the files are there
    folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fp_datasets', 'NCorrFP', dataset + '-fp')
    file_count = len([file for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file))])
    if file_count < len(combinations):
        print("WARNING: It seems that some fingerprinted datasets are missing. Total file count: ", file_count)
    # Iterate through parameter combinations (datasets)
    for combination in combinations:
        param = dict(zip(fp_params.keys(), combination))

        param_string = '_'.join(f"{key}{value}" for key, value in param.items())
        file_name = data.name + "_" + param_string + '.csv'
        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fp_datasets', 'NCorrFP',
                                 data.name + "-fp", file_name)
        # print('Trying file ', file_path)
        # Check if it's a file (skip folders)
        if os.path.isfile(file_path):
            print(f"Reading file: {file_name}")

            # Open and read the file (assuming csv files here)
            try:
                fingerprinted_data = pd.read_csv(file_path)
            except pd.errors.EmptyDataError:
                print(f"Empty file: {file_name}")
                continue
            for strength in attack_strength:
                if cluster_factor*strength <= 1:
                    cluster = pd.read_csv('cluster_g1_k325_sk999.csv').head(int(cluster_factor*strength*len(data.dataframe)))
                    cluster.index = cluster['Id']
                else:
                    cluster = pd.read_csv('cluster_g1_k325_sk999.csv')
                    cluster.index = cluster['Id']
                for exp in range(n_experiments):
                    scheme = NCorrFP(gamma=param['gamma'], fingerprint_bit_length=param['fingerprint_length'],
                                     k=param['k'],
                                     number_of_recipients=param['n_recipients'], fingerprint_code_type=param['code'])
                    attack = InfluentialRecordDeletionAttack()
                    release_data = attack.run(dataset=fingerprinted_data, fraction=strength, importance_order=cluster,
                                              random_state=2 * param['sk'] + exp)
                    detected_fp, votes, suspect_probvec = scheme.detection(release_data, secret_key=param['sk'],
                                                                           primary_key='Id',
                                                                           correlated_attributes=data.correlated_attributes,
                                                                           original_columns=list(data.columns))

                    # record the parameters for the results
                    for key, values in param.items():
                        results[key].append(values)
                    results['embedding_ratio'].append(1.0 / param['gamma'])
                    results['experiment_no'].append(exp)
                    results['attack_strength'].append(strength)
                    results['detection_confidence'].append(suspect_probvec[param['id']])

                    print(f"--Detection confidence = ", suspect_probvec[param['id']])
                    # Fidelity of the released data
                    accuracy = 1.0 - strength
                    results['accuracy'].append(accuracy)
                    print("--Accuracy ", accuracy)
                    hellinger_dist = np.mean(list(hellinger_distance(data.dataframe, release_data).values()))
                    results['hellinger_distance'].append(hellinger_dist)
                    print("--Hellinger ", hellinger_dist)
                    kl_div = np.mean(list(kl_divergence_kde(data.dataframe, release_data).values()))
                    results['kl_divergence'].append(kl_div)
                    print("--KL divergence: ", kl_div)

                    numerical_columns = data.dataframe.drop(['Id'], axis=1).select_dtypes(include=['number'])
                    categorical_columns = data.dataframe.drop(['Id'], axis=1).select_dtypes(
                        include=['object', 'category'])
                    for i, pair in enumerate(correlated_pairs_dict.keys()):
                        if pair[0] in numerical_columns and pair[1] in numerical_columns:
                            # for two numerical calculate pearson's
                            fp_corr = release_data[pair[0]].corr(release_data[pair[1]])
                            delta_corr = abs((correlated_pairs_dict[pair] - fp_corr) / correlated_pairs_dict[pair])
                            results[correlated_pairs_string[i]].append(delta_corr)
                        elif pair[0] in categorical_columns and pair[1] in categorical_columns:
                            # for categorical calculate Cramer's V
                            fp_v = utils.cramers_v(release_data[pair[0]], release_data[pair[1]])
                            delta_v = abs((correlated_pairs_dict[pair] - fp_v) / correlated_pairs_dict[pair])
                            results[correlated_pairs_string[i]].append(delta_v)
                        else:
                            # for categorical x numerical calculate eta squared
                            if pair[0] in categorical_columns:
                                cat_col = pair[0]
                                num_col = pair[1]
                            else:
                                cat_col = pair[1]
                                num_col = pair[0]
                            fp_eta = utils.eta_squared(release_data, cat_col, num_col)
                            delta_eta = abs((correlated_pairs_dict[pair] - fp_eta) / correlated_pairs_dict[pair])
                            results[correlated_pairs_string[i]].append(delta_eta)
        else:
            print(f"------------\nMissing file: {file_name}\n------------")
    print(results)
    results_frame = pd.DataFrame(results)
    if save_results is not None:
        results_frame.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), save_results), index=False)
        print(f"Results saved in {os.path.join(os.path.dirname(os.path.abspath(__file__)), save_results)}")

    return results_frame


def cluster_flipping_exact_param(dataset='adult', save_results='robustness-clusterflippingexactparam'):
    print('NCorr-FP: Robustness - Cluster flipping attack with exact param setting as the target dataset.\nData: {}'.format(dataset))
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    save_results += f"_{dataset}_{timestamp}.csv"  # out file

    # --- Read data --- #
    data = None
    if dataset == 'covertype-sample':
        data = CovertypeSample()
    if dataset == 'adult':
        data = Adult()
    if data is None:
        exit('Please provide a valid dataset name ({})'.format(datasets.__all__))
    print(data.dataframe.head(3))

    correlated_pairs_dict = utils.extract_mutually_correlated_pairs(data.dataframe,
                                                                    threshold_num=0.70, threshold_cat=0.45,
                                                                    threshold_numcat=0.14)
    correlated_pairs_string = ["_".join(list(a)) for a in list(correlated_pairs_dict.keys())]

    fp_params = {'gamma': [32],#8, 16, 32],  # 4, 8, 16, 32],
                 'k': [300, 450],
                 'fingerprint_length': [128],  # 128
                 'n_recipients': [20],
                 'sk': [100 + i for i in range(10)],
                 'id': [0],
                 'code': ['tardos']}  # 128
    attack_strength = [0.3, 0.35, 0.4, 0.45, 0.5]
    n_experiments = 3
    results = {key: [] for key in ['gamma', 'k', 'fingerprint_length', 'n_recipients', 'sk', 'id', 'code',
                                   'embedding_ratio',
                                   'experiment_no', 'attack_strength',
                                   'detection_confidence',
                                   'accuracy', 'hellinger_distance', 'kl_divergence'] + correlated_pairs_string}

    combinations = list(product(*fp_params.values()))
    # check if all the files are there
    folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fp_datasets', 'NCorrFP', dataset + '-fp')
    file_count = len([file for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file))])
    if file_count < len(combinations):
        print("WARNING: It seems that some fingerprinted datasets are missing. Total file count: ", file_count)
    # Iterate through parameter combinations (datasets)
    for combination in combinations:
        param = dict(zip(fp_params.keys(), combination))

        param_string = '_'.join(f"{key}{value}" for key, value in param.items())
        file_name = data.name + "_" + param_string + '.csv'
        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fp_datasets', 'NCorrFP',
                                 data.name + "-fp", file_name)
        # print('Trying file ', file_path)
        # Check if it's a file (skip folders)
        if os.path.isfile(file_path):
            print(f"Reading file: {file_name}")

            # Open and read the file (assuming csv files here)
            try:
                fingerprinted_data = pd.read_csv(file_path)
            except pd.errors.EmptyDataError:
                print(f"Empty file: {file_name}")
                continue
            cluster_name = 'cluster_g{}_k{}_sk999.csv'.format(param['gamma'], param['k'])
            buffer = 1.25
            for strength in attack_strength:
                cluster = pd.read_csv(cluster_name)
                if buffer*strength <= 1:  # instead of changing entire selected cluster, give a buffer of +25%
                    cluster = cluster.head(int(buffer*strength*len(data.dataframe)))
                    cluster.index = cluster['Id']
                else:  # if this is exceeding the data size, then limit to the data size
                    cluster.index = cluster['Id']
                for exp in range(n_experiments):
                    scheme = NCorrFP(gamma=param['gamma'], fingerprint_bit_length=param['fingerprint_length'],
                                     k=param['k'],
                                     number_of_recipients=param['n_recipients'], fingerprint_code_type=param['code'])
                    attack = InfluentialRecordFlippingAttack()
                    release_data, c = attack.run(dataset=fingerprinted_data, fraction=strength, cluster=cluster,
                                              random_state=2 * param['sk'] + exp)
                    detected_fp, votes, suspect_probvec = scheme.detection(release_data, secret_key=param['sk'],
                                                                           primary_key='Id',
                                                                           correlated_attributes=data.correlated_attributes,
                                                                           original_columns=list(data.columns))

                    # record the parameters for the results
                    for key, values in param.items():
                        results[key].append(values)
                    results['embedding_ratio'].append(1.0 / param['gamma'])
                    results['experiment_no'].append(exp)
                    results['attack_strength'].append(strength)
                    results['detection_confidence'].append(suspect_probvec[param['id']])

                    print(f"--Detection confidence = ", suspect_probvec[param['id']])
                    # Fidelity of the released data
                    accuracy = 1.0 - strength
                    results['accuracy'].append(accuracy)
 #                   print("--Accuracy ", accuracy)
                    hellinger_dist = np.mean(list(hellinger_distance(data.dataframe, release_data).values()))
                    results['hellinger_distance'].append(hellinger_dist)
#                    print("--Hellinger ", hellinger_dist)
                    kl_div = np.mean(list(kl_divergence_kde(data.dataframe, release_data).values()))
                    results['kl_divergence'].append(kl_div)
 #                   print("--KL divergence: ", kl_div)

                    numerical_columns = data.dataframe.drop(['Id'], axis=1).select_dtypes(include=['number'])
                    categorical_columns = data.dataframe.drop(['Id'], axis=1).select_dtypes(
                        include=['object', 'category'])
                    for i, pair in enumerate(correlated_pairs_dict.keys()):
                        if pair[0] in numerical_columns and pair[1] in numerical_columns:
                            # for two numerical calculate pearson's
                            fp_corr = release_data[pair[0]].corr(release_data[pair[1]])
                            delta_corr = abs((correlated_pairs_dict[pair] - fp_corr) / correlated_pairs_dict[pair])
                            results[correlated_pairs_string[i]].append(delta_corr)
                        elif pair[0] in categorical_columns and pair[1] in categorical_columns:
                            # for categorical calculate Cramer's V
                            fp_v = utils.cramers_v(release_data[pair[0]], release_data[pair[1]])
                            delta_v = abs((correlated_pairs_dict[pair] - fp_v) / correlated_pairs_dict[pair])
                            results[correlated_pairs_string[i]].append(delta_v)
                        else:
                            # for categorical x numerical calculate eta squared
                            if pair[0] in categorical_columns:
                                cat_col = pair[0]
                                num_col = pair[1]
                            else:
                                cat_col = pair[1]
                                num_col = pair[0]
                            fp_eta = utils.eta_squared(release_data, cat_col, num_col)
                            delta_eta = abs((correlated_pairs_dict[pair] - fp_eta) / correlated_pairs_dict[pair])
                            results[correlated_pairs_string[i]].append(delta_eta)
        else:
            print(f"------------\nMissing file: {file_name}\n------------")
    # print(results)
    results_frame = pd.DataFrame(results)
    if save_results is not None:
        results_frame.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), save_results), index=False)
        print(f"Results saved in {os.path.join(os.path.dirname(os.path.abspath(__file__)), save_results)}")

    return results_frame


def cluster_horizontal_exact_param(dataset='adult', save_results='robustness-clusterhorizontalexactparam'):
    print('NCorr-FP: Robustness - Cluster horizontal attack with exact parameter setting as the target dataset.\nData: {}'.format(dataset))
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    save_results += f"_{dataset}_{timestamp}.csv"  # out file

    # --- Read data --- #
    data = None
    if dataset == 'covertype-sample':
        data = CovertypeSample()
    if dataset == 'adult':
        data = Adult()
    if data is None:
        exit('Please provide a valid dataset name ({})'.format(datasets.__all__))
    print(data.dataframe.head(3))

    correlated_pairs_dict = utils.extract_mutually_correlated_pairs(data.dataframe,
                                                                    threshold_num=0.70, threshold_cat=0.45,
                                                                    threshold_numcat=0.14)
    correlated_pairs_string = ["_".join(list(a)) for a in list(correlated_pairs_dict.keys())]

    fp_params = {'gamma': [4],#8, 16, 32],  # 4, 8, 16, 32],
                 'k': [300],#, 450],
                 'fingerprint_length': [128],  # 128
                 'n_recipients': [20],
                 'sk': [100],# + i for i in range(10)],
                 'id': [0],
                 'code': ['tardos']}  # 128
    attack_strength = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    cluster_factor = 1.25
    n_experiments = 3
    results = {key: [] for key in ['gamma', 'k', 'fingerprint_length', 'n_recipients', 'sk', 'id', 'code',
                                   'embedding_ratio',
                                   'experiment_no', 'attack_strength',
                                   'detection_confidence',
                                   'accuracy', 'hellinger_distance', 'kl_divergence'] + correlated_pairs_string}

    combinations = list(product(*fp_params.values()))
    # check if all the files are there
    folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fp_datasets', 'NCorrFP', dataset + '-fp')
    file_count = len([file for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file))])
    if file_count < len(combinations):
        print("WARNING: It seems that some fingerprinted datasets are missing. Total file count: ", file_count)
    # Iterate through parameter combinations (datasets)
    for combination in combinations:
        param = dict(zip(fp_params.keys(), combination))

        param_string = '_'.join(f"{key}{value}" for key, value in param.items())
        file_name = data.name + "_" + param_string + '.csv'
        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fp_datasets', 'NCorrFP',
                                 data.name + "-fp", file_name)
        # print('Trying file ', file_path)
        # Check if it's a file (skip folders)
        if os.path.isfile(file_path):
            print(f"Reading file: {file_name}")

            # Open and read the file (assuming csv files here)
            try:
                fingerprinted_data = pd.read_csv(file_path)
            except pd.errors.EmptyDataError:
                print(f"Empty file: {file_name}")
                continue
            # TODO DELETE the following line
            cluster_file = 'cluster_g{}_k{}_sk100.csv'.format(param['gamma'], param['k'])
            #cluster_file = 'cluster_g{}_k{}_sk999.csv'.format(param['gamma'], param['k'])
            for strength in attack_strength:
                if cluster_factor*strength <= 1:
                    cluster = pd.read_csv(cluster_file).head(int(cluster_factor*strength*len(data.dataframe)))
                    cluster.index = cluster['Id']
                else:
                    cluster = pd.read_csv(cluster_file)
                    cluster.index = cluster['Id']
                for exp in range(n_experiments):
                    scheme = NCorrFP(gamma=param['gamma'], fingerprint_bit_length=param['fingerprint_length'],
                                     k=param['k'],
                                     number_of_recipients=param['n_recipients'], fingerprint_code_type=param['code'])
                    attack = InfluentialRecordDeletionAttack()
                    release_data = attack.run(dataset=fingerprinted_data, fraction=strength, importance_order=cluster,
                                              random_state=2 * param['sk'] + exp)
                    detected_fp, votes, suspect_probvec = scheme.detection(release_data, secret_key=param['sk'],
                                                                           primary_key='Id',
                                                                           correlated_attributes=data.correlated_attributes,
                                                                           original_columns=list(data.columns))

                    # record the parameters for the results
                    for key, values in param.items():
                        results[key].append(values)
                    results['embedding_ratio'].append(1.0 / param['gamma'])
                    results['experiment_no'].append(exp)
                    results['attack_strength'].append(strength)
                    results['detection_confidence'].append(suspect_probvec[param['id']])

                    print(f"--Detection confidence = ", suspect_probvec[param['id']])
                    # Fidelity of the released data
                    accuracy = 1.0 - strength
                    results['accuracy'].append(accuracy)
                    print("--Accuracy ", accuracy)
                    hellinger_dist = np.mean(list(hellinger_distance(data.dataframe, release_data).values()))
                    results['hellinger_distance'].append(hellinger_dist)
                    print("--Hellinger ", hellinger_dist)
                    kl_div = np.mean(list(kl_divergence_kde(data.dataframe, release_data).values()))
                    results['kl_divergence'].append(kl_div)
                    print("--KL divergence: ", kl_div)

                    numerical_columns = data.dataframe.drop(['Id'], axis=1).select_dtypes(include=['number'])
                    categorical_columns = data.dataframe.drop(['Id'], axis=1).select_dtypes(
                        include=['object', 'category'])
                    for i, pair in enumerate(correlated_pairs_dict.keys()):
                        if pair[0] in numerical_columns and pair[1] in numerical_columns:
                            # for two numerical calculate pearson's
                            fp_corr = release_data[pair[0]].corr(release_data[pair[1]])
                            delta_corr = abs((correlated_pairs_dict[pair] - fp_corr) / correlated_pairs_dict[pair])
                            results[correlated_pairs_string[i]].append(delta_corr)
                        elif pair[0] in categorical_columns and pair[1] in categorical_columns:
                            # for categorical calculate Cramer's V
                            fp_v = utils.cramers_v(release_data[pair[0]], release_data[pair[1]])
                            delta_v = abs((correlated_pairs_dict[pair] - fp_v) / correlated_pairs_dict[pair])
                            results[correlated_pairs_string[i]].append(delta_v)
                        else:
                            # for categorical x numerical calculate eta squared
                            if pair[0] in categorical_columns:
                                cat_col = pair[0]
                                num_col = pair[1]
                            else:
                                cat_col = pair[1]
                                num_col = pair[0]
                            fp_eta = utils.eta_squared(release_data, cat_col, num_col)
                            delta_eta = abs((correlated_pairs_dict[pair] - fp_eta) / correlated_pairs_dict[pair])
                            results[correlated_pairs_string[i]].append(delta_eta)
        else:
            print(f"------------\nMissing file: {file_name}\n------------")
    print(results)
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

    # collaboration attacks
    collusion(args.dataset)
    # 1-user attack
#    horizontal_attack()
#    vertical_attack()
#    flipping_attack()
#    cluster_flipping()  # the second  most disruptive
#    cluster_horizontal()
#    cluster_flipping_exact_param()  # the most disruptive
#    cluster_horizontal_exact_param()
