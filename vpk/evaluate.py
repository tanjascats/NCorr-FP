from NCorrFP.NCorrFP import *
import insert_vpk
from datetime import datetime
import utils
from datasets import *
from attacks import *
from itertools import product


def vpk_exclusiveness(data_path):
    """
    Calculates the exclusiveness of vpk in data.
    This indicates that regular content-based VPKs cannot be distinguished for this number of records.
    Returns:

    """
    data_with_vpk = insert_vpk.add_virtual_primary_key(data_path)
    unique = len(data_with_vpk['Id'].unique())
    total = len(data_with_vpk['Id'])
    return round(unique/total, 2)


def record_exclusiveness(data_path):
    """
        Calculates the exclusiveness of records in data.
        This indicates that regular content-based VPKs cannot be distinguished for this number of records.
        Returns:

        """
    data = pd.read_csv(data_path)
    unique = data.drop_duplicates()
    return round(len(unique) / len(data), 2)


def detection_confidence(runs=10):
    """
    Evaluates the detection confidence on a clean (non-manipulated/non-attacked) fingerprinted dataset when vpk is used.

    Returns: mean detection confidence

    """
    scheme = NCorrFP(gamma=2, fingerprint_bit_length=128)
    recipient_id = 4
    avg_conf = 0
    for run in range(runs):
        random_sate = 101+run
        fingerprinted_data = scheme.insertion_vpk('adult-vpk', secret_key=random_sate, recipient_id=recipient_id,
                                                  outfile='adult_vpk_{}.csv'.format(random_sate))
        fingerprint_template, count, exact_match_scores = scheme.detection_vpk(fingerprinted_data, secret_key=random_sate,
                                       correlated_attributes=['relationship', 'marital-status', 'occupation',
                                                              'workclass', 'education-num'])
        avg_conf += exact_match_scores[recipient_id]

    avg_conf = avg_conf/runs
    return avg_conf


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
#                    hellinger_distances = hellinger_distance(data.dataframe[common_columns], release_data[common_columns])
#                    # Hellinger averaged over non-deleted columns
#                    hellinger_dist = np.mean(list(hellinger_distances.values()))
#                    results['hellinger_distance_remaining'].append(hellinger_dist)
#                    print("--Hellinger (remaining) ", hellinger_dist)
                    # Hellinger averaged with 1.0 for missing columns
#                    for missing_col in missing_columns:
#                        hellinger_distances[missing_col] = 1.0
#                    hellinger_dist = np.mean(list(hellinger_distances.values()))
#                    results['hellinger_distance'].append(hellinger_dist)
#                    print("--Hellinger ", hellinger_dist)

                    # KL divergence upper limit is infinity so we can't deal with this
                    #kl_div = np.mean(list(kl_divergence_kde(data.dataframe, release_data).values()))
                    #results['kl_divergence'].append(kl_div)
                    #print("--KL divergence: ", kl_div)

#                    numerical_columns = data.dataframe.select_dtypes(include=['number'])
#                    categorical_columns = data.dataframe.select_dtypes(include=['object', 'category'])
#                    for i, pair in enumerate(correlated_pairs_dict.keys()):
#                        # If one of the attributes is missing we write nan
#                        if pair[0] not in release_data.columns or pair[1] not in release_data.columns:
#                            results[correlated_pairs_string[i]].append(None)
#                        else:
#                            if pair[0] in numerical_columns and pair[1] in numerical_columns:
#                                # for two numerical calculate pearson's
#                                fp_corr = release_data[pair[0]].corr(release_data[pair[1]])
#                                delta_corr = abs((correlated_pairs_dict[pair] - fp_corr) / correlated_pairs_dict[pair])
#                                results[correlated_pairs_string[i]].append(delta_corr)
#                            elif pair[0] in categorical_columns and pair[1] in categorical_columns:
#                                # for categorical calculate Cramer's V
#                                fp_v = utils.cramers_v(release_data[pair[0]], release_data[pair[1]])
#                                delta_v = abs((correlated_pairs_dict[pair] - fp_v) / correlated_pairs_dict[pair])
#                                results[correlated_pairs_string[i]].append(delta_v)
#                            else:
#                                # for categorical x numerical calculate eta squared
#                                if pair[0] in categorical_columns:
#                                    cat_col = pair[0]
#                                    num_col = pair[1]
#                                else:
#                                    cat_col = pair[1]
#                                    num_col = pair[0]
#                                fp_eta = utils.eta_squared(release_data, cat_col, num_col)
#                                delta_eta = abs((correlated_pairs_dict[pair] - fp_eta) / correlated_pairs_dict[pair])
#                                results[correlated_pairs_string[i]].append(delta_eta)
        else:
            print(f"------------\nMissing file: {file_name}\n------------")
    print(results)
    results_frame = pd.DataFrame(results)
    if save_results is not None:
        results_frame.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), save_results), index=False)
        print(f"Results saved in {os.path.join(os.path.dirname(os.path.abspath(__file__)), save_results)}")

    return results_frame


if __name__ == '__main__':
    print("Uniqueness of Adult census data")
    print(record_exclusiveness("../datasets/adult_train.csv"))
    print("VPK uniqueness")
    print(vpk_exclusiveness("../datasets/adult_train.csv"))
    print("Detection confidence (ovg confidence on clean fp dataset when vpk is used):")
    print(detection_confidence())

