from knn_scheme.scheme import CategoricalNeighbourhood
from knn_scheme.experimental.blind_scheme import BlindNNScheme


def test_knn():
#    scheme = CategoricalNeighbourhood(gamma=1)
    scheme = BlindNNScheme(gamma=1, fingerprint_bit_length=16)
    data = "datasets/breast_cancer_full.csv"
    fingerprinted_data = scheme.insertion('breast-cancer', primary_key='Id', secret_key=601, recipient_id=4,
                                          outfile='knn_scheme/outfiles/fp_data_blind_corr_all_attributes.csv',
                                          correlated_attributes=['age', 'menopause', 'inv-nodes', 'node-caps'])
#
#    suspect = scheme.detection(fingerprinted_data, secret_key=100, original_data=data)
    #print(fingerprinted_data)
    #columns = ['Id'] + list(fingerprinted_data.columns)
    #fingerprinted_data['Id'] = fingerprinted_data.index
    #fingerprinted_data = fingerprinted_data[columns]
    #print(fingerprinted_data)
    suspect = scheme.detection(fingerprinted_data, secret_key=601, primary_key='Id',
                               correlated_attributes=['age', 'menopause', 'inv-nodes', 'node-caps'])


def knn_adult_census():
    scheme = BlindNNScheme(gamma=10, fingerprint_bit_length=32)

    fingerprinted_data = scheme.insertion('adult', primary_key='Id', secret_key=100, recipient_id=4,
                                          outfile='knn_scheme/outfiles/adult_fp_acc_wc_en_100.csv',
                                          correlated_attributes=['relationship', 'marital-status', 'occupation', 'workclass', 'education-num'])
    suspect = scheme.detection(fingerprinted_data, secret_key=100, primary_key='Id',
                               correlated_attributes=['relationship', 'marital-status', 'occupation', 'workclass',
                                                      'education-num'])


if __name__ == '__main__':
    knn_adult_census()
