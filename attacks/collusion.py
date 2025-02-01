import datetime

from NCorrFP.NCorrFP import NCorrFP
from fp_codes import tardos
import pandas as pd
import numpy as np
from utils import *
from collections import Counter


# testing naive approaches to collusion attack
def insert_fingerprints(n_recipients, secret_key, gamma=1, fp_bit_len=16, code_type='tardos'):
    scheme = NCorrFP(gamma=gamma, fingerprint_bit_length=fp_bit_len, k=100, fingerprint_code_type=code_type,
                     number_of_recipients=n_recipients)
    original_path = "NCorrFP/test/test_data/synthetic_1000_3_continuous.csv"
    correlated_attributes = ['X', 'Y']
    for r in range(n_recipients):
        scheme.insertion(original_path, primary_key_name='Id', secret_key=secret_key, recipient_id=r,
                         correlated_attributes=correlated_attributes, save_computation=True,
                         outfile='attacks/collusion/test_data/test_id{}.csv'.format(r))
    return scheme


# Step 1: consider 2 colluding attackers; merge 2 datasets by averaging on disagreeing values
def collude_2_datasetes_by_avg(dataset_path_1=None, dataset_path_2=None):
    """
    Scenario with 2 colluding attackers.
    Merging 2 fingerprinted datasets by averaging on their disagreeing values.
    Args:
        dataset_path_1 (str): rel or abs paths to one colluding dataset
        dataset_path_2 (str): rel or abs path to another colluding dataset
    Returns:

    """
    fp_data_1 = pd.read_csv(dataset_path_1)
    fp_data_2 = pd.read_csv(dataset_path_2)

    # Check if both dataframes have the same shape and columns
    if fp_data_1.shape != fp_data_2.shape or not (fp_data_1.columns == fp_data_2.columns).all():
        raise ValueError("Datasets must have the same shape and column names")

    # Calculate the merged DataFrame by averaging only where values disagree
    merged_df = fp_data_1.copy()
    disagreement_mask = fp_data_1 != fp_data_2  # True where values disagree
    averaged_values = (fp_data_1 + fp_data_2) / 2  # Average the values

    # Assign averaged values, casting them to the original data type in df1
    for column in fp_data_1.columns:
        if disagreement_mask[column].any():  # Only process if there are disagreements in the column
            merged_df.loc[disagreement_mask[column], column] = averaged_values[column].astype(fp_data_1[column].dtype)

    return merged_df


def collude_datasetes_by_avg(dataset_paths):

    """
    Merges multiple DataFrames by averaging values where they disagree among any of the provided DataFrames.
    For numerical data the average is calculated; for categorical the mode is calculated.

    Args:
        dataset_paths (list): Arbitrary number of paths to datasets to collude.

    Returns:
        pd.DataFrame: Merged DataFrame with averaged values where any disagreements occur.
    """
    print('Collusion strategy: avg')
    if len(dataset_paths) < 2:
        raise ValueError("At least two DataFrames are required for comparison.")

    dataframes = [pd.read_csv(ds) for ds in dataset_paths]

    # Check if all DataFrames have the same shape and columns
    ref_shape = dataframes[0].shape
    ref_columns = dataframes[0].columns
    if not all(df.shape == ref_shape and df.columns.equals(ref_columns) for df in dataframes):
        raise ValueError("All DataFrames must have the same shape and columns.")

    # Identify numerical and categorical columns
    numerical_columns = dataframes[0].select_dtypes(include=['number']).columns
    categorical_columns = dataframes[0].select_dtypes(include=['object', 'category']).columns

    # Stack all DataFrames along a new axis to allow comparison across all values
    stacked = np.stack([df.values for df in dataframes])

    # Initiate the colluded dataset
    merged_df = pd.DataFrame(stacked[0], columns=ref_columns).copy(deep=True)

    # Handle numerical attributes
    for col in numerical_columns:
        col_idx = ref_columns.get_loc(col)
        # Extract the column values across datasets
        column_values = stacked[:, :, col_idx]
        # Find disagreements
        disagreements = np.any(column_values != column_values[0], axis=0)
        # Calculate the mean along the new axis for numerical values
        mean_values = np.mean(column_values, axis=0)
        # Replace disagreements with the mean
        merged_df.loc[disagreements, col] = mean_values[disagreements]

    # Handle categorical attributes
    for col in categorical_columns:
        col_idx = ref_columns.get_loc(col)
        # Extract the column values across datasets
        column_values = stacked[:, :, col_idx]
        # For each row, determine the most frequent value
        for row_idx in range(column_values.shape[1]):
            value_counts = Counter(column_values[:, row_idx])  # biased towards the first dataset in order if counts are tied (always the case in 2-collusion) so we need to shuffle
            max_count = max(value_counts.values())
            top_candidates = [val for val, count in value_counts.items() if count == max_count]
            if len(top_candidates) > 1:  # shuffle top-candidates if there's a tie
                np.random.shuffle(top_candidates)
            # Select the first candidate (either the only candidate or one randomly chosen among the tied ones)
            most_common_value = top_candidates[0]
            merged_df.loc[row_idx, col] = most_common_value

    return merged_df


def collude_datasetes_by_random(dataset_paths):

    """
    Merges multiple DataFrames by replacing disagreements with random values from the attribute's domain.

    Args:
        dataset_paths (list): Arbitrary number of paths to datasets to collude.

    Returns:
        pd.DataFrame: Merged DataFrame with random new values where any disagreements occur.
    """
    if len(dataset_paths) < 2:
        raise ValueError("At least two DataFrames are required for comparison.")

    dataframes = [pd.read_csv(ds) for ds in dataset_paths]

    # Check if all DataFrames have the same shape and columns
    ref_shape = dataframes[0].shape
    ref_columns = dataframes[0].columns
    if not all(df.shape == ref_shape and df.columns.equals(ref_columns) for df in dataframes):
        raise ValueError("All DataFrames must have the same shape and columns.")

    # Stack all DataFrames along a new axis to allow comparison across all values
    stacked = np.stack([df.values for df in dataframes])

    # Calculate where disagreements occur by checking if all values along the new axis are the same
    disagreements = np.any(stacked != stacked[0, :, :], axis=0)

    # Create the merged DataFrame
    merged_df = pd.DataFrame(stacked[0], columns=ref_columns).copy()

    # Replace disagreements with random values from the attribute's domain
    for col in ref_columns:
        if disagreements[:, ref_columns.get_loc(col)].any():
            # Get unique values for the attribute across all DataFrames (the domain)
            attribute_domain = np.unique(stacked[:, :, ref_columns.get_loc(col)])

            # Generate random values from the domain for the disagreement positions
            np.random.seed(int(datetime.datetime.now().timestamp()))  # np.random is seeded elsewhere
            random_values = np.random.choice(attribute_domain, size=disagreements[:, ref_columns.get_loc(col)].sum())

            # Insert random values in positions where disagreements occur
            merged_df.loc[disagreements[:, ref_columns.get_loc(col)], col] = random_values

    return merged_df


def collude_datasets_by_random_and_flipping(dataset_paths):
    """
        Merges multiple DataFrames by replacing disagreements with random values from the attribute's domain.
        Additionally, modifies a random selection of values in the merged DataFrame to random values from their respective attribute domains.

        Args:
            dataset_paths (list): Arbitrary number of paths to datasets to collude.

        Returns:
            pd.DataFrame: Merged DataFrame with disagreements and additional random modifications.
        """
    if len(dataset_paths) < 2:
        raise ValueError("At least two DataFrames are required for comparison.")

    dataframes = [pd.read_csv(ds) for ds in dataset_paths]

    # Check if all DataFrames have the same shape and columns
    ref_shape = dataframes[0].shape
    ref_columns = dataframes[0].columns
    if not all(df.shape == ref_shape and df.columns.equals(ref_columns) for df in dataframes):
        raise ValueError("All DataFrames must have the same shape and columns.")

    # Stack all DataFrames along a new axis to allow comparison across all values
    stacked = np.stack([df.values for df in dataframes])

    # Identify disagreements
    disagreements = np.any(stacked != stacked[0, :, :], axis=0)
    num_disagreements = np.sum(disagreements)

    # Create the merged DataFrame and replace disagreements with random values
    merged_df = pd.DataFrame(stacked[0], columns=ref_columns).copy()
    for col in ref_columns:
        if disagreements[:, ref_columns.get_loc(col)].any():
            attribute_domain = np.unique(stacked[:, :, ref_columns.get_loc(col)])
            random_values = np.random.choice(attribute_domain, size=disagreements[:, ref_columns.get_loc(col)].sum())
            merged_df.loc[disagreements[:, ref_columns.get_loc(col)], col] = random_values

    # Randomly modify additional values in the DataFrame
    if num_disagreements > 0:
        rows, cols = merged_df.shape
        total_elements = rows * cols

        # Select random positions for additional modifications
        random_indices = np.random.choice(total_elements, size=num_disagreements, replace=False)
        for index in random_indices:
            row, col_idx = divmod(index, cols)
            col_name = merged_df.columns[col_idx]
            attribute_domain = np.unique(stacked[:, :, col_idx])
            merged_df.iloc[row, col_idx] = np.random.choice(attribute_domain)

    return merged_df


def collude_3_datasets_by_avg(dataset_path_1, dataset_path_2, dataset_path_3):
    fp_data_1 = pd.read_csv(dataset_path_1)
    fp_data_2 = pd.read_csv(dataset_path_2)
    fp_data_3 = pd.read_csv(dataset_path_3)
    # Merge the datasets on the 'Id' column to align the rows
    merged_df = pd.merge(fp_data_1, fp_data_2, on='Id', how='outer', suffixes=('_df1', '_df2'))
    merged_df = pd.merge(merged_df, fp_data_3, on='Id', how='outer', suffixes=('', '_df3'))
    print('Attack: collusion of 3 by avg.')
    # Identify differences by comparing the corresponding columns
#    differences = (merged['X_df1'] != merged['X_df2']) | \
#                  (merged['Y_df1'] != merged['Y_df2']) | \
#                  (merged['Z_df1'] != merged['Z_df2'])
    # Count the total number of differing rows
#    total_differences = differences.sum()
#    print(f"Total differences: {total_differences}")
    # Create a new dataset that averages the differences
    merged_df['X'] = merged_df[['X_df1', 'X_df2', 'X']].mean(axis=1)
    merged_df['Y'] = merged_df[['Y_df1', 'Y_df2', 'Y']].mean(axis=1)
    merged_df['Z'] = merged_df[['Z_df1', 'Z_df2', 'Z']].mean(axis=1)
    final_merged_df = merged_df[['Id', 'X', 'Y', 'Z']].round(0).astype(int)

    final_merged_df.to_csv('attacks/collusion/out_data/merged_dataset_3colluders.csv', index=False)
    return final_merged_df


def bitstring_to_array(bitstring):
    return [int(bit) for bit in bitstring]


def collusion_resolution_tardos():
    # insert a few fingerprints
    gamma = 1
    fp_bit_len = 128
    for i in range(5):
        fingerprints = insert_fingerprint(i, secret_key=101, gamma=gamma, fp_bit_len=fp_bit_len)
    s = tardos_codes.check_exact_matching(string_to_array("00011010010010111000010011001110"), 101, 5)
    print('Sanity check ', s)

    # collude 2 datasets by averaging values with already embedded tardos codes; secret key 101
    colluded_ds = collude_2_datasetes_by_avg(dataset_path_1='attacks/collusion/test_data/test_id2.csv',
                                             dataset_path_2='attacks/collusion/test_data/test_id4.csv')

    scheme = NCorrFP(gamma=gamma, fingerprint_bit_length=fp_bit_len, k=100)
    suspect = scheme.detection(colluded_ds, secret_key=101, primary_key='Id',
                               correlated_attributes=['X', 'Y'],
                               original_columns=["X", 'Y', 'Z'])
    sus = tardos_codes.check_exact_matching(list_to_string(scheme.detected_fp), 101, scheme.number_of_recipients)
    # print(fingerprints)
    print(list_to_string(scheme.detected_fp))
    print("Exact match? ", sus)

    #    code1 = string_to_array("11011010110011101100110111011111")
    #    code2 = string_to_array("00011001110100101110100011011110")
    #    marked_code = np.where(code1 == code2, code1, 1)
    #    print("Marked code: ", marked_code)
    print("Detected co: ", scheme.detected_fp)
    #    colluders_check = tardos_codes.detect_colluders(marked_code, secret_key=101, total_n_recipients=5)
    #    print("Colluders ", colluders_check)
    colluders = tardos_codes.detect_colluders(scheme.detected_fp, secret_key=101, total_n_recipients=5)
    print("Colluders ", colluders)


def collusion_resolution_hash():
    # insert a few fingerprints
    gamma = 1
    fp_bit_len = 64
    n_recipients = 20
    scheme = NCorrFP(gamma=gamma, fingerprint_bit_length=fp_bit_len, k=100, fingerprint_code_type='hash',
                     number_of_recipients=n_recipients)
    # uncomment line below to replicate the fingerprint embedding
    # scheme = insert_fingerprints(n_recipients, secret_key=101, gamma=gamma, fp_bit_len=fp_bit_len, code_type='hash')
    # sanity check
    sus_data = pd.read_csv('attacks/collusion/test_data/test_id2.csv')
    scheme.detection(sus_data, secret_key=101, primary_key='Id', correlated_attributes=['X', 'Y'],
                     original_columns=['X', 'Y', 'Z'])

    # collude 2 datasets by averaging values with already embedded hash codes; secret key 101
    colluded_ds_2 = collude_2_datasetes_by_avg(dataset_path_1='attacks/collusion/test_data/test_id2.csv',
                                               dataset_path_2='attacks/collusion/test_data/test_id9.csv')

    suspect = scheme.detection(colluded_ds_2, secret_key=101, primary_key='Id',
                               correlated_attributes=['X', 'Y'],
                               original_columns=["X", 'Y', 'Z'])

    # collude 3 datasets by averaging values with already embedded hash codes; secret key 101
    colluded_ds_3 = collude_3_datasets_by_avg(dataset_path_1='attacks/collusion/test_data/test_id2.csv',
                                              dataset_path_2='attacks/collusion/test_data/test_id9.csv',
                                              dataset_path_3='attacks/collusion/test_data/test_id3.csv')

    suspect = scheme.detection(colluded_ds_3, secret_key=101, primary_key='Id',
                               correlated_attributes=['X', 'Y'],
                               original_columns=["X", 'Y', 'Z'])


if __name__ == '__main__':
    collusion_resolution_hash()
