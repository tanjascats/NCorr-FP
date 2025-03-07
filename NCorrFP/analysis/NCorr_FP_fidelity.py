import sys
sys.path.insert(0, '../dissertation')  # make the script standalone for running on server

import datasets
from datasets import CovertypeSample, Adult

import pandas as pd
import argparse
import os
from itertools import product
from datetime import datetime
import numpy as np
from scipy.stats import entropy, gaussian_kde, wasserstein_distance, ks_2samp
from scipy.spatial.distance import cdist
from math import sqrt
import utils


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


def kl_divergence_kde(df1, df2, num_points=100):
    """
    Calculate KL divergence for each column between two DataFrames using Kernel Density Estimation.

    Args:
    - df1 (pd.DataFrame): First DataFrame (reference distribution).
    - df2 (pd.DataFrame): Second DataFrame (comparison distribution).
    - num_points (int): Number of points for evaluating the KDE.

    Returns:
    - dict: KL divergence values for each column.
    """
    kl_divergences = {}

    numerical_columns = df1.drop(['Id'], axis=1).select_dtypes(include=['number'])
    categorical_columns = df1.drop(['Id'], axis=1).select_dtypes(include=['object', 'category'])

    for column in numerical_columns:
        # Compute KDE for each DataFrame's feature column
        kde1 = gaussian_kde(df1[column], bw_method='scott')
        kde2 = gaussian_kde(df2[column], bw_method='scott')

        # Define a range for evaluation based on both DataFrames' data in the column
        min_value = min(df1[column].min(), df2[column].min())
        max_value = max(df1[column].max(), df2[column].max())
        x = np.linspace(min_value, max_value, num_points)

        # Evaluate the KDE for each dataset
        p = kde1(x)
        q = kde2(x)

        # Add a small constant to avoid division by zero or log(0)
        p += 1e-10
        q += 1e-10

        # Normalize the distributions
        p /= p.sum()
        q /= q.sum()

        # Calculate KL divergence for this feature
        kl_divergences[column] = entropy(p, q)

    # Categorical attributes
    for col in categorical_columns:
        freq1 = df1[col].value_counts(normalize=True)
        freq2 = df2[col].value_counts(normalize=True)

        # Align categories (fill missing categories with 0)
        all_categories = set(freq1.index).union(set(freq2.index))
        freq1 = freq1.reindex(all_categories, fill_value=0)
        freq2 = freq2.reindex(all_categories, fill_value=0)

        # Compute KL divergence
        freq1 = np.array(freq1)
        freq2 = np.array(freq2)

        # Avoid division by zero or log of zero by adding a small epsilon
        epsilon = 1e-10
        freq1 = np.clip(freq1, epsilon, 1)
        freq2 = np.clip(freq2, epsilon, 1)

        kl_divergences[col] = np.sum(freq1 * np.log(freq1 / freq2))

    return kl_divergences


def kl_divergence(df1, df2, bins=10):
    """
    Calculate KL divergence for each column between two DataFrames.

    Args:
    - df1 (pd.DataFrame): First DataFrame (reference distribution).
    - df2 (pd.DataFrame): Second DataFrame (comparison distribution).
    - bins (int): Number of bins for discretizing continuous data.

    Returns:
    - dict: KL divergence values for each column.
    """
    kl_divergences = {}

    for column in df1.columns:
        # Create histograms for each feature in both DataFrames
        p, bin_edges = np.histogram(df1[column], bins=bins, density=True)
        q, _ = np.histogram(df2[column], bins=bin_edges, density=True)

        # Add a small constant to avoid division by zero or log(0)
        p += 1e-10
        q += 1e-10

        # Normalize the distributions
        p /= p.sum()
        q /= q.sum()

        # Calculate KL divergence for this feature
        kl_divergences[column] = entropy(p, q)

    return kl_divergences


def emd(df1, df2):
    """
    Calculate the Earth Mover's Distance (EMD) between corresponding columns in two DataFrames.
    Lower EMD values indicate similar distributions, while higher values suggest greater differences.
    Args:
    - df1 (pd.DataFrame): First DataFrame.
    - df2 (pd.DataFrame): Second DataFrame.

    Returns:
    - dict: EMD values for each column.
    """
    # Ensure the DataFrames have the same columns
    if not all(df1.columns == df2.columns):
        raise ValueError("DataFrames must have the same columns to compute EMD.")

    emd_distances = {}

    numerical_columns = df1.drop(['Id'], axis=1).select_dtypes(include=['number'])
    categorical_columns = df1.drop(['Id'], axis=1).select_dtypes(include=['object', 'category'])

    for column in numerical_columns:
        # Calculate EMD (Wasserstein distance) for each column
        emd_distances[column] = wasserstein_distance(df1[column], df2[column])

        # Calculate EMD for categorical attributes
    for column in categorical_columns:
        # Get unique values and their frequencies for each dataset
        freq1 = df1[column].value_counts(normalize=True).sort_index()
        freq2 = df2[column].value_counts(normalize=True).sort_index()

        # Align the distributions to include all unique values
        all_categories = set(freq1.index).union(set(freq2.index))
        freq1 = freq1.reindex(all_categories, fill_value=0)
        freq2 = freq2.reindex(all_categories, fill_value=0)

        # Use a simple distance matrix for categories (e.g., 0 if same, 1 if different)
        distance_matrix = np.ones((len(all_categories), len(all_categories))) - np.eye(len(all_categories))

        # Calculate the Earth Mover's Distance using frequency distributions
        emd_distances[column] = np.sum(
            cdist(freq1.values.reshape(-1, 1), freq2.values.reshape(-1, 1), metric='euclidean'))

    return emd_distances


def ks_statistic(df1, df2):
    """
    Calculate the Kolmogorov-Smirnov (KS) statistic for each column between two DataFrames.
    KS Statistic: Indicates the degree of difference between distributions (closer to 0 means more similar).
    P-value: Tests the null hypothesis that both distributions are the same. A small p-value (e.g., <0.05) indicates a significant difference between the distributions.

    Args:
    - df1 (pd.DataFrame): First DataFrame.
    - df2 (pd.DataFrame): Second DataFrame.

    Returns:
    - dict: KS statistic values for each column.
    """
    # Ensure the DataFrames have the same columns
    if not all(df1.columns == df2.columns):
        raise ValueError("DataFrames must have the same columns to compute KS statistic.")

    ks_statistics = {}

    for column in df1.columns:
        # Calculate the KS statistic and p-value for each column
        ks_stat, p_value = ks_2samp(df1[column], df2[column])
        ks_statistics[column] = {'ks_statistic': ks_stat, 'p_value': p_value}

    return ks_statistics


def hellinger_distance(df1, df2, num_points=100):
    """
    Calculate the Hellinger Distance for each column between two DataFrames.
    A smaller distance (closer to 0) indicates similar distributions for a given feature, while values closer to 1 suggest more significant differences. This metric is particularly useful for comparing how well distributions align across two datasets.
    Args:
    - df1 (pd.DataFrame): First DataFrame.
    - df2 (pd.DataFrame): Second DataFrame.
    - num_points (int): Number of points for evaluating the KDE.

    Returns:
    - dict: Hellinger Distance values for each column.
    """
    # Ensure the DataFrames have the same columns
    if not all(df1.columns == df2.columns):
        raise ValueError("DataFrames must have the same columns to compute Hellinger Distance.")

    hellinger_distances = {}

    if 'Id' in df1:
        numerical_columns = df1.drop(['Id'], axis=1).select_dtypes(include=['number'])
        categorical_columns = df1.drop(['Id'], axis=1).select_dtypes(include=['object', 'category'])
    else:
        numerical_columns = df1.select_dtypes(include=['number'])
        categorical_columns = df1.select_dtypes(include=['object', 'category'])

    # Numerical
    for column in numerical_columns:
        # Compute KDE for each column in both DataFrames
        kde1 = gaussian_kde(df1[column], bw_method='scott')
        kde2 = gaussian_kde(df2[column], bw_method='scott')

        # Define a range for evaluation based on the combined range of both DataFrames
        min_value = min(df1[column].min(), df2[column].min())
        max_value = max(df1[column].max(), df2[column].max())
        x = np.linspace(min_value, max_value, num_points)

        # Evaluate the KDEs to get probability densities
        p = kde1(x)
        q = kde2(x)

        # Normalize the densities to make them probability distributions
        p /= p.sum()
        q /= q.sum()

        # Calculate Hellinger Distance
        hellinger_dist = sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2))
        hellinger_distances[column] = hellinger_dist

    # Categorical attributes
    for col in categorical_columns:
        freq1 = df1[col].value_counts(normalize=True)
        freq2 = df2[col].value_counts(normalize=True)

        # Align categories (fill missing categories with 0)
        all_categories = set(freq1.index).union(set(freq2.index))
        freq1 = freq1.reindex(all_categories, fill_value=0)
        freq2 = freq2.reindex(all_categories, fill_value=0)

        # Compute Hellinger distance
        distance = (1 / np.sqrt(2)) * np.sqrt(np.sum((np.sqrt(freq1) - np.sqrt(freq2)) ** 2))
        hellinger_distances[col] = distance

    return hellinger_distances


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
    print('NCorr-FP: Fidelity.\nData: {}'.format(dataset))
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    save_bivar = save_results + f"_bivariate_{dataset}_{timestamp}.csv"
    save_results += f"_univariate_{dataset}_{timestamp}.csv"  # out file

    # --- Read data --- #
    data = None
    if dataset == 'covertype-sample':
        data = CovertypeSample()
    elif dataset == 'adult':
        data = Adult()
    if data is None:
        exit('Please provide a valid dataset name ({})'.format(datasets.__all__))
    print(data.dataframe.head(3))
    numerical_columns = data.dataframe.drop(['Id'], axis=1).select_dtypes(include=['number'])
    categorical_columns = data.dataframe.drop(['Id'], axis=1).select_dtypes(include=['object', 'category'])

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
    # todo: correlation analysis (pearson, cramer's V, ets squared)
    # 1. Pearson correlation matrix for numerical cols
#    correlation_matrix_original = data.dataframe.corr()
#    correlated_pairs = extract_high_correlations(correlation_matrix_original, threshold=0.55)
    correlated_pairs_dict = utils.extract_mutually_correlated_pairs(data.dataframe,
                                                                    threshold_num=0.70, threshold_cat=0.45, threshold_numcat=0.14)
    correlated_pairs_string = ["-".join(list(a)) for a in list(correlated_pairs_dict.keys())]

    # --- Define parameters --- #
    params = {'gamma': [2, 4, 8, 16, 32], #, 4, 8, 16, 32],
              'k': [300, 450],#, 450],
              'fingerprint_length': [128, 256, 512], #, 128, 256, 512], #, 64 128, 256],
              'n_recipients': [20],
              'sk': [100 + i for i in range(10)], #10)]}  # #sk-s = #experiments
              'id': [i for i in range(20)],
              'code': ['tardos']} #,i for i in range(20)]}

    # --- Initialise the results --- #
    # Univariate
    results_univar = {key: [] for key in list(params.keys()) +
                      ['embedding_ratio', 'recipient_id', 'attribute', 'rel_delta_mean', 'rel_delta_std',
                       'hellinger_distance', 'kl_divergence', 'emd', 'ks', 'p_value']}
    # Bivariate
    results_bivar = {key: [] for key in list(params.keys()) +
               ['embedding_ratio', 'recipient_id', 'accuracy'] + correlated_pairs_string}

    # --- Run the detection and count: --- #
    #   1. wrong votes
    #   2. fingerprint detection confidence
    #   3. detection confidence for the wrong recipients
    combinations = list(product(*params.values()))
    # check if all the files are there
    folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fp_datasets', 'NCorrFP', dataset + '-fp')
    file_count = len([file for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file))])
    if file_count < len(combinations):
        print("WARNING: It seems that some fingerprinted datasets are missing. Total file count: ", file_count)
    # Iterate through parameter combinations (datasets)
    for combination in combinations:
        param = dict(zip(params.keys(), combination))

        param_string = '_'.join(f"{key}{value}" for key, value in param.items())
        file_name = data.name + "_" + param_string + '.csv'
        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fp_datasets', 'NCorrFP', data.name + "-fp", file_name)

        # Check if it's a file (skip folders)
        if os.path.isfile(file_path):
            print(f"Reading file: {file_name}")

            # Open and read the file (assuming csv files here)
            try:
                fingerprinted_data = pd.read_csv(file_path)
            except pd.errors.EmptyDataError:
                print(f"Empty file: {file_name}")
                continue


            # -- Calculate delta mean and std for each (numerical) attribute --- #
            delta_mean_std = get_delta_mean_std(fingerprinted_data, data.dataframe)
#            print(delta_mean_std.index)
#            print(data.dataframe.columns); exit()
            # -- Calculate: (i) hellinger dist, (ii) kl divergence, (iii) emd, (iv) ks (numerical)
            hellinger_dist = hellinger_distance(data.dataframe, fingerprinted_data)
            kl_div = kl_divergence_kde(data.dataframe, fingerprinted_data)
            emd_score = emd(data.dataframe, fingerprinted_data)
            ks = ks_statistic(data.dataframe, fingerprinted_data)  # returns ks and p-value
            for attribute in data.dataframe.columns:
                if attribute == 'Id':
                    continue
                # record the parameters for the results
                for key, values in param.items():
                    results_univar[key].append(values)
                results_univar['embedding_ratio'].append(1.0 / param['gamma'])
                results_univar['recipient_id'].append(param['id'])

                # add the stat results
                results_univar['attribute'].append(attribute)
                if attribute in delta_mean_std.index:
                    results_univar['rel_delta_mean'].append(delta_mean_std['rel_delta_mean'][attribute])
                    results_univar['rel_delta_std'].append(delta_mean_std['rel_delta_std'][attribute])
                    results_univar['ks'].append(ks[attribute]['ks_statistic'])
                    results_univar['p_value'].append(ks[attribute]['p_value'])
                else:  # categorical attributes
                    results_univar['rel_delta_mean'].append(None)
                    results_univar['rel_delta_std'].append(None)
                    results_univar['ks'].append(None)
                    results_univar['p_value'].append(None)

                results_univar['hellinger_distance'].append(hellinger_dist[attribute])
                results_univar['kl_divergence'].append(kl_div[attribute])
                results_univar['emd'].append(emd_score[attribute])


            # -- Calculated delta corr for highly correlated pairs -- #
            # record the parameters for the results
            for key, values in param.items():
                results_bivar[key].append(values)
            results_bivar['embedding_ratio'].append(1.0 / param['gamma'])
            results_bivar['recipient_id'].append(param['id'])

            # add the stat results
            # data accuracy (% changed values)
            accuracy = (fingerprinted_data != data.dataframe).sum().sum() / \
                       np.prod(data.dataframe.drop(['Id'], axis=1).shape)
            results_bivar['accuracy'].append(accuracy)
            # correlations
            for i, pair in enumerate(correlated_pairs_dict.keys()):
                if pair[0] in numerical_columns and pair[1] in numerical_columns:
                    # for two numerical calculate pearson's
                    fp_corr = fingerprinted_data[pair[0]].corr(fingerprinted_data[pair[1]])
                    delta_corr = abs((correlated_pairs_dict[pair] - fp_corr)/correlated_pairs_dict[pair])
                    results_bivar[correlated_pairs_string[i]].append(delta_corr)
                elif pair[0] in categorical_columns and pair[1] in categorical_columns:
                    # for categorical calculate Cramer's V
                    fp_v = utils.cramers_v(fingerprinted_data[pair[0]], fingerprinted_data[pair[1]])
                    delta_v = abs((correlated_pairs_dict[pair] - fp_v)/correlated_pairs_dict[pair])
                    results_bivar[correlated_pairs_string[i]].append(delta_v)
                else:
                    # for categorical x numerical calculate eta squared
                    if pair[0] in categorical_columns:
                        cat_col = pair[0]
                        num_col = pair[1]
                    else:
                        cat_col = pair[1]
                        num_col = pair[0]
                    fp_eta = utils.eta_squared(fingerprinted_data, cat_col, num_col)
                    delta_eta = abs((correlated_pairs_dict[pair] - fp_eta)/correlated_pairs_dict[pair])
                    results_bivar[correlated_pairs_string[i]].append(delta_eta)
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
