import unittest
import itertools

import numpy as np

from ..NCorrFP import *
import cProfile

warnings.simplefilter(action='ignore', category=FutureWarning)


class TestNCorrFP(unittest.TestCase):
    def test_insertion_single_correlation(self):
        scheme = NCorrFP(gamma=1, fingerprint_bit_length=16)
        original = pd.read_csv("datasets/breast_cancer_full.csv")
        correlated_attributes = ['inv-nodes', 'node-caps']
        fingerprinted_data = scheme.insertion('breast-cancer', primary_key_name='Id', secret_key=101, recipient_id=4,
                                              correlated_attributes=correlated_attributes)
        values1_fp = fingerprinted_data['inv-nodes'].unique()
        values2_fp = fingerprinted_data['node-caps'].unique()
        # Generate all combinations of the two attribute values
        combinations_fp = list(itertools.product(values1_fp, values2_fp))
        values1 = original['inv-nodes'].unique()
        values2 = original['node-caps'].unique()
        # Generate all combinations of the two attribute values
        combinations = list(itertools.product(values1, values2))
        difference = [item for item in combinations_fp if item not in combinations]
        self.assertIs(len(difference), 0, msg='SUCCESS. No new occurences of value combinations for correlated '
                                              'attributes.')

    def test_detection_single_correlation(self):
        secret_key = 106
        scheme = NCorrFP(gamma=1, fingerprint_bit_length=16, k=20)
        correlated_attributes = ['inv-nodes', 'node-caps']
        fingerprinted_data = scheme.insertion('breast-cancer', primary_key_name='Id', secret_key=secret_key, recipient_id=4,
                                              correlated_attributes=correlated_attributes)
        suspect = scheme.detection(fingerprinted_data, secret_key=secret_key, primary_key='Id',
                                   correlated_attributes=['inv-nodes', 'node-caps'],
                                   original_columns=["age", "menopause", "tumor-size", "inv-nodes", "node-caps",
                                                     "deg-malig", "breast", "breast-quad",
                                                     "irradiat", "recurrence"])
        self.assertIs(suspect, 4, msg="SUCCESS. The detected recipient is correct.")

    def test_insertion_multi_correlation(self):
        scheme = NCorrFP(gamma=1, fingerprint_bit_length=16, k=20)
        original = pd.read_csv("datasets/breast_cancer_full.csv")
        correlated_attributes = [['inv-nodes', 'node-caps'], ['age', 'menopause']]
        fingerprinted_data = scheme.insertion('breast-cancer', primary_key_name='Id', secret_key=101, recipient_id=4,
                                              correlated_attributes=correlated_attributes)
        values1_fp = fingerprinted_data['inv-nodes'].unique()
        values2_fp = fingerprinted_data['node-caps'].unique()
        values3_fp = fingerprinted_data['age'].unique()
        values4_fp = fingerprinted_data['menopause'].unique()
        # Generate all combinations of the two attribute values
        combinations_fp_corr1 = list(itertools.product(values1_fp, values2_fp))
        combinations_fp_corr2 = list(itertools.product(values3_fp, values4_fp))

        values1 = original['inv-nodes'].unique()
        values2 = original['node-caps'].unique()
        values3 = original['age'].unique()
        values4 = original['menopause'].unique()
        # Generate all combinations of the two attribute values
        combinations_corr1 = list(itertools.product(values1, values2))
        combinations_corr2 = list(itertools.product(values3, values4))

        difference1 = [item for item in combinations_fp_corr1 if item not in combinations_corr1]
        difference2 = [item for item in combinations_fp_corr2 if item not in combinations_corr2]

        self.assertIs(len(difference1)+len(difference2), 0,
                      msg='SUCCESS. No new occurences of value combinations for correlated attributes.')

    def test_detection_multi_correlation(self):
        secret_key = 1101
        scheme = NCorrFP(gamma=1, fingerprint_bit_length=16, k=20)
        correlated_attributes = [['age', 'menopause'], ['inv-nodes', 'node-caps']]
        fingerprinted_data = scheme.insertion('breast-cancer', primary_key_name='Id', secret_key=secret_key,
                                              recipient_id=4,
                                              correlated_attributes=correlated_attributes)
        suspect = scheme.detection(fingerprinted_data, secret_key=secret_key, primary_key='Id',
                                   correlated_attributes=correlated_attributes,
                                   original_columns=["age", "menopause", "tumor-size", "inv-nodes", "node-caps",
                                                     "deg-malig", "breast", "breast-quad",
                                                     "irradiat", "recurrence"])
        self.assertIs(suspect, 4, msg="SUCCESS. The detected recipient is correct.")

    def test_parse_correlated_attrs_list(self):
        relation, primary_key = import_dataset_from_file('breast-cancer', 'Id')
        correlated_attributes = ['inv-nodes', 'node-caps']
        print(parse_correlated_attrs(correlated_attributes, relation))
        self.assertEqual(len(parse_correlated_attrs(correlated_attributes, relation)), 1)

    def test_parse_correlated_attrs_list_of_lists(self):
        relation, primary_key = import_dataset_from_file('breast-cancer', 'Id')
        correlated_attributes = [['inv-nodes', 'node-caps'], ['age', 'menopause']]
        print(parse_correlated_attrs(correlated_attributes, relation))
        self.assertEqual(len(parse_correlated_attrs(correlated_attributes, relation)), 2)

    def test_parse_correlated_attrs_list_of_lists_multi(self):
        relation, primary_key = import_dataset_from_file('breast-cancer', 'Id')
        correlated_attributes = [['inv-nodes', 'node-caps'], ['age', 'menopause', 'inv-nodes']]
        print(parse_correlated_attrs(correlated_attributes, relation))
        self.assertEqual(len(parse_correlated_attrs(correlated_attributes, relation)), 1)

    def test_parse_correlated_attrs_none(self):
        relation, primary_key = import_dataset_from_file('breast-cancer', 'Id')
        print(parse_correlated_attrs(None, relation))
        self.assertEqual(len(parse_correlated_attrs(None, relation)), 1)

    def test_parse_correlated_attrs_wrong(self):
        relation, primary_key = import_dataset_from_file('breast-cancer', 'Id')
        correlated_attributes = 'inv-nodes'
        self.assertRaises(ValueError, parse_correlated_attrs, correlated_attributes, relation)

    def test_init_balltrees_single_corr(self):
        original = pd.read_csv("datasets/breast_cancer_full.csv").drop('Id', axis=1)
        correlated_attributes = [['inv-nodes', 'node-caps']]

        # label encoder
        categorical_attributes = original.select_dtypes(include='object').columns
        label_encoders = dict()
        for cat in categorical_attributes:
            label_enc = LabelEncoder()
            original[cat] = label_enc.fit_transform(original[cat])
            label_encoders[cat] = label_enc

        balltree = init_balltrees(correlated_attributes, original,
                                  categorical_attributes=categorical_attributes)
        print(balltree)

    def test_init_balltrees_multi_corr(self):
        original = pd.read_csv("datasets/breast_cancer_full.csv").drop('Id', axis=1)
        correlated_attributes = [['inv-nodes', 'node-caps'], ['age', 'menopause']]

        # label encoder
        categorical_attributes = original.select_dtypes(include='object').columns
        label_encoders = dict()
        for cat in categorical_attributes:
            label_enc = LabelEncoder()
            original[cat] = label_enc.fit_transform(original[cat])
            label_encoders[cat] = label_enc

        balltree = init_balltrees(correlated_attributes, original, categorical_attributes=categorical_attributes)
        print(balltree)

    def test_mark_continuous_value_mark1(self):
        neighbours = [16,  9, 11, 12,  1,  1,  2,  4,  7, 11,
                      18,  7,  3,  7, 14,  0, 14, 12,  3, 14,
                      3,  7, 18,  0,  0,  9,  0, 19,  7,  7,
                      4, 5, 5, 18, 15]   # Generating some example data
        mark_bit = 1
        marked_attribute = mark_continuous_value(neighbours, mark_bit=mark_bit, plot=True)
        print(marked_attribute)
        self.assertTrue(marked_attribute in [3, 4, 5, 7])

    def test_mark_continuous_value_mark0(self):
        neighbours = [16, 9, 11, 12, 1, 1, 2, 4, 7, 11,
                      18, 7, 3, 7, 14, 0, 14, 12, 3, 14,
                      3, 7, 18, 0, 0, 9, 0, 19, 7, 7,
                      4, 5, 5, 18, 15]  # Generating some example data
        mark_bit = 0
        marked_attribute = mark_continuous_value(neighbours, mark_bit=mark_bit, plot=True)
        print(marked_attribute)
        self.assertTrue(marked_attribute in [0, 1, 2, 9, 11, 12, 14, 15, 16, 18, 19])

    def test_insertion_continuous(self):
        scheme = NCorrFP(gamma=3, fingerprint_bit_length=16,
                         distance_metric_continuous='minkowski', k=10)
        original_path = "NCorrFP/test/test_data/synthetic_300_continuous.csv"
        original = pd.read_csv(original_path)
        correlated_attributes = ['X', 'Y']
        correlation_original = original['X'].corr(original['Y'])
        fingerprinted_data = scheme.insertion(original_path, primary_key_name='Id', secret_key=101, recipient_id=4,
                                              correlated_attributes=correlated_attributes)
        correlation_fingerprinted = fingerprinted_data['X'].corr(fingerprinted_data['Y'])
        delta = 0.04
        message = 'Original and fingerprinted correlations are almost equal within {}'.format(delta)
        self.assertAlmostEqual(correlation_fingerprinted, correlation_original, None, message, delta)

    def test_insertion_continuous_3_correlated(self):
        scheme = NCorrFP(gamma=3, fingerprint_bit_length=16,
                         distance_metric_continuous='minkowski', k=10)
        original_path = "NCorrFP/test/test_data/synthetic_300_3_continuous.csv"
        original = pd.read_csv(original_path)
        correlated_attributes = ['X', 'Y', 'Z']
        correlation_original = original['Y'].corr(original['Z'])
        fingerprinted_data = scheme.insertion(original_path, primary_key_name='Id', secret_key=101, recipient_id=4,
                                              correlated_attributes=correlated_attributes)
        correlation_fingerprinted = fingerprinted_data['Y'].corr(fingerprinted_data['Z'])
        delta = 0.03
        message = 'Original and fingerprinted correlations are almost equal within {}'.format(delta)
        self.assertAlmostEqual(correlation_fingerprinted, correlation_original, None, message, delta)

    def test_detection_continuous(self):
        scheme = NCorrFP(gamma=1, fingerprint_bit_length=16, k=30)
        original_path = "NCorrFP/test/test_data/synthetic_300_3_continuous.csv"
        original = pd.read_csv(original_path)
        correlated_attributes = ['X', 'Y']
        fingerprinted_data = scheme.insertion(original_path, primary_key_name='Id', secret_key=101, recipient_id=4,
                                              correlated_attributes=correlated_attributes)
        suspect = scheme.detection(fingerprinted_data, secret_key=101, primary_key='Id',
                                   correlated_attributes=correlated_attributes,
                                   original_columns=["X", 'Y', 'Z'])
        self.assertIs(suspect, 4, msg="SUCCESS. The detected recipient is correct.")

    def test_categorical_minkowski(self):
        secret_key = 1010
        distance_metric = 'minkowski'
        scheme = NCorrFP(gamma=1, fingerprint_bit_length=16, distance_metric_discrete=distance_metric, k=20)
        original = pd.read_csv("datasets/breast_cancer_full.csv")
        correlated_attributes = ['inv-nodes', 'node-caps']
        fingerprinted_data = scheme.insertion('breast-cancer', primary_key_name='Id', secret_key=secret_key, recipient_id=4,
                                              correlated_attributes=correlated_attributes)
        suspect = scheme.detection(fingerprinted_data, secret_key=secret_key, primary_key='Id',
                                   correlated_attributes=correlated_attributes,
                                   original_columns=["age", "menopause", "tumor-size", "inv-nodes", "node-caps",
                                                     "deg-malig", "breast", "breast-quad",
                                                     "irradiat", "recurrence"])
        self.assertIs(suspect, 4, msg="SUCCESS. The detected recipient is correct.")

    def test_runtime(self):
        # done: weak points? (hypothesis: reading and writing to pandas df is slow; querying ball-trees is expensive...)
        # done: testing the full runtime over #records -- linearity?
        # done: optimisation
        # done: runtime estimation
        # done: same drill for insertion and detection
        scheme = NCorrFP(gamma=1, fingerprint_bit_length=16, k=10)
        original_path = "NCorrFP/test/test_data/synthetic_1000_3_continuous.csv"
        correlated_attributes = ['X', 'Y']
        start_insertion = time.time()
        fingerprinted_data = scheme.insertion(original_path, primary_key_name='Id', secret_key=101, recipient_id=4,
                                              correlated_attributes=correlated_attributes, save_computation=True)
        end_insertion = time.time()
        time_insertion = end_insertion - start_insertion
        start_detection = time.time()
        suspect = scheme.detection(fingerprinted_data, secret_key=101, primary_key='Id',
                                   correlated_attributes=correlated_attributes,
                                   original_columns=["X", 'Y', 'Z'])
        time_detection = time.time() - start_detection
        print('Runtime:\n\t-Insertion in {} seconds.'.format(round(time_insertion, 4)))
        print('Runtime:\n\t-Detection in {} seconds.'.format(time_detection))
        #self.assertIs(suspect, 4, msg="SUCCESS. The detected recipient is correct.")

    def test_detection_gamma_float(self):
        scheme = NCorrFP(gamma=1/0.7, fingerprint_bit_length=16, k=100)
        original_path = "NCorrFP/test/test_data/synthetic_1000_3_continuous.csv"
        original = pd.read_csv(original_path)
        correlated_attributes = ['X', 'Y']
        fingerprinted_data = scheme.insertion(original_path, primary_key_name='Id', secret_key=101, recipient_id=4,
                                              correlated_attributes=correlated_attributes)
        suspect = scheme.detection(fingerprinted_data, secret_key=101, primary_key='Id',
                                   correlated_attributes=correlated_attributes,
                                   original_columns=["X", 'Y', 'Z'])
        self.assertIs(suspect, 4, msg="SUCCESS. The detected recipient is correct.")

    def test_generate_fingerprint_tardos_consistency(self):
        secret_key = 101
        rid = 0
        scheme = NCorrFP(fingerprint_bit_length=16, fingerprint_code_type='tardos')
        fingerprint_1 = scheme.create_fingerprint(recipient_id=rid, secret_key=secret_key)

        fingerprint_2 = scheme.create_fingerprint(recipient_id=rid, secret_key=secret_key)
        self.assertTrue(np.array_equal(fingerprint_2, fingerprint_1))

    def test_generate_fingerprint_tardos_diversity(self):
        secret_key = 101
        scheme = NCorrFP(fingerprint_bit_length=16, fingerprint_code_type='tardos')
        fingerprints = []
        for i in range(16):
            fingerprints.append(scheme.create_fingerprint(recipient_id=i, secret_key=secret_key))
        unique_fps = {tuple(fp) for fp in fingerprints}
        self.assertEqual(len(unique_fps), len(fingerprints))

