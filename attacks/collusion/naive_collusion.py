from NCorrFP_scheme.NCorrFP import NCorrFP
import tardos_codes
import pandas as pd
import numpy as np

# testing naive approaches to collusion attack

# Step 1: consider 2 colluding attackers; merge dataset and average the differences
def collude_2_avg(dataset_paths=None):
    # fingerprint two dataset and distribute to 2 recipients
    scheme = NCorrFP(gamma=2, fingerprint_bit_length=32, k=100)
    original_path = "NCorrFP_scheme/test/test_data/synthetic_1000_3_continuous.csv"
    correlated_attributes = ['X', 'Y']
    recipients = [2, 4]
    fp_data = dict()
    if dataset_paths is not None and len(dataset_paths) == 2:
        fp_data[2] = pd.read_csv(dataset_paths[0])
        fp_data[4] = pd.read_csv(dataset_paths[1])
    fingerprints = []
    for r in recipients:
        if dataset_paths is None:  # if datasets are not provided
            fp_data[r] = scheme.insertion(original_path, primary_key_name='Id', secret_key=101, recipient_id=r,
                                                  correlated_attributes=correlated_attributes, save_computation=True,
                                                  outfile='attacks/collusion/test_data/test_id{}.csv'.format(r))
        # check that fingerprint is inserted correctly
        if scheme.detection(fp_data[r], secret_key=101, primary_key='Id',
                            correlated_attributes=correlated_attributes,
                            original_columns=["X", 'Y', 'Z']) != r:
            exit('Fingerprint detection fail for recipient {}'.format(r))
        fingerprints.append(scheme.detected_fp)

    # Merge the datasets on the 'Id' column to align the rows
    merged = pd.merge(fp_data[recipients[0]], fp_data[recipients[1]],
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
    suspect = scheme.detection(final_merged_df, secret_key=101, primary_key='Id',
                               correlated_attributes=correlated_attributes,
                               original_columns=["X", 'Y', 'Z'])
    print(fingerprints)
    print(scheme.detected_fp)


def test_tardos(codes, marked_code):
    p = np.random.beta(0.5, 0.5, size=32)
    suspected_colluders = tardos_codes._detect_colluders_old(codes, marked_code, p, k=0.9)  # , threshold)
    return suspected_colluders

def bitstring_to_array(bitstring):
    return [int(bit) for bit in bitstring]

if __name__ == '__main__':
    #collude_2_avg(['NCorrFP_scheme/test/out_data/test_id2.csv',
    #               'NCorrFP_scheme/test/out_data/test_id4.csv'])
    test_tardos([bitstring_to_array('01011100001111100110001010110100'),
                 bitstring_to_array('00000100011110101000001110000100')],
                bitstring_to_array("00001100001111101120001110000100"))
