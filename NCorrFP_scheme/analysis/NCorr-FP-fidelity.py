import sys
sys.path.insert(0, '../dissertation')  # make the script standalone for running on server

import datasets
from datasets import CovertypeSample
from NCorrFP_scheme.NCorrFP import NCorrFP

import pandas as pd
import argparse
import os
from itertools import product
from datetime import datetime
import numpy as np


def get_delta_mean_std(df1, df2):
    """
    Calculated relative differences between mean and std values for each attribute between two datasets.
    Args:
        df1 (pandas.DataFrame):
        df2 (pandas.DataFrame):

    Returns: pandas.DataFrame of the results

    """
    # Calculate descriptive statistics for each dataset
    stats1 = df1.describe().transpose()
    stats2 = df2.describe().transpose()

    # Initialize DataFrame for results
    results = pd.DataFrame(index=stats1.index, columns=['rel_delta_mean', 'rel_delta_std'])

    # Calculate relative differences
    for column in stats1.index:
        # Calculate range for each attribute
        range_attr = stats1.loc[column, 'max'] - stats1.loc[column, 'min']

        # Calculate relative difference in mean
        diff_mean = abs(stats1.loc[column, 'mean'] - stats2.loc[column, 'mean'])
        relative_diff_mean = diff_mean / range_attr if range_attr != 0 else 0

        # Calculate relative difference in standard deviation
        diff_std = abs(stats1.loc[column, 'std'] - stats2.loc[column, 'std'])
        relative_diff_std = diff_std / stats1.loc[column, 'std'] if stats1.loc[column, 'std'] != 0 else 0

        # Store results
        results.loc[column, 'rel_delta_mean'] = relative_diff_mean
        results.loc[column, 'rel_delta_std'] = relative_diff_std

    return results


def extract_high_correlations(correlation_matrix, threshold=0.55):
    """
    Extract pairs of attributes with a correlation greater than the threshold in absolute value.

    Args:
    - correlation_matrix (pd.DataFrame): The correlation matrix of a DataFrame.
    - threshold (float): The minimum absolute correlation value to consider.

    Returns:
    - Dictionary (attribute pair, correlation): Each key contains (attribute1, attribute2) and the value is correlation
    """
    # Select the upper triangle of the correlation matrix without the diagonal
    upper_tri = correlation_matrix.where(
        np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
    )

    # Find attribute pairs with correlation above the threshold
    high_corr_pairs = dict()
    for col in upper_tri.columns:
        if col == 'Id':
            continue
        for idx in upper_tri.index:
            if idx == 'Id':
                continue
            if abs(upper_tri.loc[idx, col]) > threshold:
                high_corr_pairs[(idx, col)] = upper_tri.loc[idx, col]

    return high_corr_pairs


def fidelity(dataset='covertype-sample', save_results='fidelity'):
    """
    Perform fidelity analysis of NCorr-FP
        1. dataset value accuracy (data similarity)
        2. Univariate stats
        3. Bivariate stats
    Args:
        dataset (str): dataset name; it's expected that it's predefined in the module dataset
        save_results: name extension for the file; in case it's None, the results are not saved into a file

    Returns: pd.DataFrame of the fidelity results

    """
    print('NCorr-FP: Effectiveness.\nData: {}'.format(dataset))
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    save_bivar = save_results + f"_bivariate_{dataset}_{timestamp}.csv"
    save_results += f"_univariate_{dataset}_{timestamp}.csv"  # out file

    # --- Read data --- #
    data = None
    if dataset == 'covertype-sample':
        data = CovertypeSample()
    if data is None:
        exit('Please provide a valid dataset name ({})'.format(datasets.__all__))
    print(data.dataframe.head(3))

    # --- Measure the statistics of the original dataset --- #
    # todo: the general problem with these is that I usually need a numeric value to describe and compare
    # todo: univariate descriptive stats per attribute
    # 1. Mean
    # 2. Median
    # 3. Std
    # 4. Min, Max
    # 5. Quartiles (25th, 50th, 75th percentiles)
    descriptive_stats_original = data.dataframe.describe().transpose()
    # todo: Frequency and disributions
    # 1. Unique values per column
    # 2. Frequency counts (density function/ binned frequencies)
    # todo: correlation analysis
    # 1. Pearson correlation matrix for numerical cols
    correlation_matrix_original = data.dataframe.corr()
    correlated_pairs = extract_high_correlations(correlation_matrix_original, threshold=0.55)
    correlated_pairs_string = ["-".join(list(a)) for a in list(correlated_pairs.keys())]

    # --- Define parameters --- #
    params = {'gamma': [2, 4, 8, 16, 32],
              'k': [300, 500],
              'fingerprint_length': [64, 128, 256], #, 64 128, 256],
              'n_recipients': [20],
              'sk': [100 + i for i in range(10)], #10)]}  # #sk-s = #experiments
              'id': [i for i in range(20)]}

    # --- Initialise the results --- #
    # todo: add all metrics
    # Univariate
    results_univar = {key: [] for key in list(params.keys()) +
               ['embedding_ratio', 'recipient_id', 'attribute', 'rel_delta_mean', 'rel_delta_std']}
    # Bivariate
    results_bivar = {key: [] for key in list(params.keys()) +
               ['embedding_ratio', 'recipient_id'] + correlated_pairs_string}

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
    # Iterate through parameter combinations (datasets)
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

            # todo: do stuff on fingerprinted data
            # -- Calculate delta mean and std for each attribute --- #
            stats_fp = get_delta_mean_std(fingerprinted_data, data.dataframe)
            print(stats_fp)
            print(stats_fp['rel_delta_mean']['Slope'])
            for attribute in stats_fp.index:
                # record the parameters for the results
                for key, values in param.items():
                    results_univar[key].append(values)
                results_univar['embedding_ratio'].append(1.0 / param['gamma'])
                results_univar['recipient_id'].append(param['id'])

                # add the stat results
                results_univar['attribute'].append(attribute)
                results_univar['rel_delta_mean'].append(stats_fp['rel_delta_mean'][attribute])
                results_univar['rel_delta_std'].append(stats_fp['rel_delta_std'][attribute])

            # -- Calculated delta corr for highly correlated pairs -- #
            # record the parameters for the results
            for key, values in param.items():
                results_bivar[key].append(values)
            results_bivar['embedding_ratio'].append(1.0 / param['gamma'])
            results_bivar['recipient_id'].append(param['id'])

            # add the stat results
            for i, pair in enumerate(correlated_pairs):
                fp_corr = fingerprinted_data[pair[0]].corr(fingerprinted_data[pair[1]])
                delta_corr = abs((correlated_pairs[pair] - fp_corr)/correlated_pairs[pair])
                results_bivar[correlated_pairs_string[i]].append(delta_corr)

    print(results_bivar)
    results_univar_frame = pd.DataFrame(results_univar)
    results_bivar_frame = pd.DataFrame(results_bivar)
    if save_results is not None:
        results_univar_frame.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), save_results), index=False)
        print(f"Results saved in {os.path.join(os.path.dirname(os.path.abspath(__file__)), save_results)}")
        results_bivar_frame.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), save_bivar), index=False)

    return results_univar_frame, results_bivar_frame


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Load a dataset with optional configurations.")
    # Required positional argument
    parser.add_argument("dataset", type=str, help="Dataset name.")
    # Parse arguments
    args = parser.parse_args()

    fidelity(args.dataset)
