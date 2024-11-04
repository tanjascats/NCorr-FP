from NCorrFP_scheme.NCorrFP import NCorrFP
from fp_codes import tardos
import pandas as pd
import numpy as np
from utils import *


# testing naive approaches to collusion attack
def insert_fingerprints(n_recipients, secret_key, gamma=1, fp_bit_len=16, code_type='tardos'):
    scheme = NCorrFP(gamma=gamma, fingerprint_bit_length=fp_bit_len, k=100, fingerprint_code_type=code_type,
                     number_of_recipients=n_recipients)
    original_path = "NCorrFP_scheme/test/test_data/synthetic_1000_3_continuous.csv"
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
    # Merge the datasets on the 'Id' column to align the rows
    merged = pd.merge(fp_data_1, fp_data_2,
                      on="Id", suffixes=('_df1', '_df2'))
    print('Attack: collusion of 2 by avg.')
    # Identify differences by comparing the corresponding columns
    differences = (merged['X_df1'] != merged['X_df2']) | \
                  (merged['Y_df1'] != merged['Y_df2']) | \
                  (merged['Z_df1'] != merged['Z_df2'])
    # Count the total number of differing rows
    total_differences = differences.sum()
    print(f"Total differences: {total_differences}")
    # Create a new dataset that averages the differences
    merged['X'] = merged[['X_df1', 'X_df2']].mean(axis=1)
    merged['Y'] = merged[['Y_df1', 'Y_df2']].mean(axis=1)
    merged['Z'] = merged[['Z_df1', 'Z_df2']].mean(axis=1)
    final_merged_df = merged[['Id', 'X', 'Y', 'Z']].round(0).astype(int)

    # Save the final merged DataFrame to CSV
    # final_merged_df.to_csv(output_file_path, index=False)
    # print(f"Merged dataset saved to: {output_file_path}")

    # Check the detection on merged dataset
#    suspect = scheme.detection(final_merged_df, secret_key=101, primary_key='Id',
#                               correlated_attributes=correlated_attributes,
#                               original_columns=["X", 'Y', 'Z'])
#    sus = tardos_codes.check_exact_matching(list_to_string(scheme.detected_fp), 101, scheme.number_of_recipients)
#    print(fingerprints)
#    print(list_to_string(scheme.detected_fp))
#    print("Exact match? ", sus)
    final_merged_df.to_csv('attacks/collusion/out_data/merged_dataset.csv', index=False)
    return final_merged_df


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
