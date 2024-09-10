import unittest
import itertools
from ..NCorrFP import *

warnings.simplefilter(action='ignore', category=FutureWarning)


class TestNCorrFP(unittest.TestCase):
    def test_insertion_single_correlation(self):
        scheme = NCorrFP(gamma=1, fingerprint_bit_length=16)
        original = pd.read_csv("datasets/breast_cancer_full.csv")
        correlated_attributes = ['inv-nodes', 'node-caps']
        fingerprinted_data = scheme.insertion('breast-cancer', primary_key='Id', secret_key=101, recipient_id=4,
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
        scheme = NCorrFP(gamma=1, fingerprint_bit_length=16)
        original = pd.read_csv("datasets/breast_cancer_full.csv")
        correlated_attributes = ['inv-nodes', 'node-caps']
        fingerprinted_data = scheme.insertion('breast-cancer', primary_key='Id', secret_key=101, recipient_id=4,
                                              correlated_attributes=correlated_attributes)
        suspect = scheme.detection(fingerprinted_data, secret_key=101, primary_key='Id',
                                   correlated_attributes=['inv-nodes', 'node-caps'],
                                   original_columns=["age", "menopause", "tumor-size", "inv-nodes", "node-caps",
                                                     "deg-malig", "breast", "breast-quad",
                                                     "irradiat", "recurrence"])
        self.assertIs(suspect, 4, msg="SUCCESS. The detected recipient is correct.")

    def test_insertion_multi_correlation(self):
        # todo
        pass

    def test_init_balltrees(self):
        original = pd.read_csv("datasets/breast_cancer_full.csv").drop('Id', axis=1)
        correlated_attributes = ['inv-nodes', 'node-caps']

        # label encoder
        categorical_attributes = original.select_dtypes(include='object').columns
        label_encoders = dict()
        for cat in categorical_attributes:
            label_enc = LabelEncoder()
            original[cat] = label_enc.fit_transform(original[cat])
            label_encoders[cat] = label_enc

        balltree = init_balltrees(correlated_attributes, original)
        print(balltree)