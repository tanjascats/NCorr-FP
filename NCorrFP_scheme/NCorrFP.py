import numpy.random as random
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import BallTree
from bitstring import BitArray
import hashlib
import bitstring
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from utils import *
from utils import _read_data

_MAXINT = 2**31 - 1


def init_balltrees(correlated_attributes, relation):
    """
    Initialises balltrees for neighbourhood search.
    Balltrees for correlated attributes are created from the attribute's correlated attributes.
    Balltrees for other attributes are created from all other attributes.
    Args:
        correlated_attributes: (strictly) a list of lists (groups) of correlated attributes
        relation: the DataFrame of the dataset

    Returns: a dictionary (attribute name: balltree)

    """
    start_training_balltrees = time.time()
    # ball trees from all-except-one attribute and all attributes
    balltree = dict()
    for attr in relation.columns:
        # get the index of a list of correlated attributes to attr; if attr is not correlated then return None
        index = next((i for i, sublist in enumerate(correlated_attributes) if attr in sublist), None)
        if index is not None:  # if attr is part of any group of correlations
            balltree_i = BallTree(relation[correlated_attributes[index]].drop(attr, axis=1), metric="hamming")
        else:  # if attr is not correlated to anything
            balltree_i = BallTree(relation.drop(attr, axis=1), metric='hamming')
        balltree[attr] = balltree_i
    print("Training balltrees in: " + str(round(time.time() - start_training_balltrees, 2)) + " sec.")
    return balltree


def merge_mutually_correlated_groups(megalist):
    # Iteratively merge lists that have any common elements
    merged = []

    for sublist in megalist:
        # Find all existing lists in merged that share elements with the current sublist
        to_merge = [m for m in merged if any(elem in m for elem in sublist)]

        # Remove the lists that will be merged from merged
        for m in to_merge:
            merged.remove(m)

        # Merge the current sublist with the found lists and add back to merged
        merged.append(set(sublist).union(*to_merge))

    # Convert sets back to lists
    return [list(group) for group in merged]


def parse_correlated_attrs(correlated_attributes, relation):
    """
    Checks the validity of passed arguments for correlated attributes. They can be either a list, list of lists or None.
    Args:
        correlated_attributes: argument provided by the user
        relation: pandas DataFrame dataset

    Returns: list of groups (lists) of correlated attributes

    """
    # correlated attributes are treated always as a list of lists even if there is only one group of corr. attributes
    if correlated_attributes is None:
        if 'Id' in relation.columns:
            relation = relation.drop('Id', axis=1)
        correlated_attributes = [relation.columns[:]]  # everything is correlated if not otherwise specified
        # Check if the input is a list of lists
    elif isinstance(correlated_attributes, list) and all(isinstance(i, list) for i in correlated_attributes):
        # It's a list of lists; multiple groups of correlated attributes
        # If there are multiple correlation groups with the same attribute, we consider them all mutually correlated
        correlated_attributes = merge_mutually_correlated_groups(correlated_attributes)
        correlated_attributes = [pd.Index(corr_group) for corr_group in correlated_attributes]
    elif isinstance(correlated_attributes, list):
        # It's a single list
        correlated_attributes = [pd.Index(correlated_attributes)]
    else:
        raise ValueError("Input correlated_attributes must be either a list or a list of lists")
    return correlated_attributes


class NCorrFP():
    # todo: add support for looking at frequencies/distributions of continuous attributes
    # supports the dataset size of up to 1,048,576 entries
    __primary_key_len = 20

    def __init__(self, gamma, xi=1, fingerprint_bit_length=32, number_of_recipients=100, distance_based=False,
                 d=0, k=10):
        self.gamma = gamma
        self.xi = xi
        self.fingerprint_bit_length = fingerprint_bit_length
        self.number_of_recipients = number_of_recipients
        self.distance_based = distance_based  # if False, then fixed-size-neighbourhood-based with k=10 - default
        if distance_based:
            self.d = d
        else:
            self.k = k
        self._INIT_MESSAGE = "NCorrFP - initialised.\n\t(Correlation-preserving NN-based fingerprinting scheme.)\nEmbedding started...\n" \
                             "\tgamma: " + str(self.gamma) + "\n\tfingerprint length: " + \
                             str(self.fingerprint_bit_length) + "\n\tdistance based: " + str(self.distance_based)
        self.count = None  # the most recent fingerprint bit-wise counts
        self.detected_fp = None  # the msot recently detected fingerprint

    def create_fingerprint(self, recipient_id, secret_key):
        """
        Creates a fingerprint for a recipient with the given ID
        :param recipient_id: identifier of a data copy recipient
        :param secret_key: owner's secret key used to fingerprint the data
        :return: fingerprint (BitArray)
        """
        if recipient_id < 0 or recipient_id >= self.number_of_recipients:
            print("Please specify valid recipient id")
            exit()

        # seed is generated by concatenating secret key with recipients id
        shift = 10
        # seed is 42 bit long
        seed = (secret_key << shift) + recipient_id
        b = hashlib.blake2b(key=seed.to_bytes(6, 'little'), digest_size=int(self.fingerprint_bit_length / 8))
        fingerprint = BitArray(hex=b.hexdigest())
        fp_msg = "\nGenerated fingerprint for recipient " + str(recipient_id) + ": " + fingerprint.bin
        print(fp_msg)
        return fingerprint

    def detect_potential_traitor(self, fingerprint, secret_key):
        """
        Detects a suspect from the extracted fingerprint
        :param fingerprint: string of characters describing binary representation of a fingerprint or a bitstring
        :param secret_key: owner's secret key used to fingerprint the data
        :return: id of a suspect or -1 if no suspect is detected
        """
        if isinstance(fingerprint, bitstring.BitArray):
            fingerprint = fingerprint.bin

        shift = 10
        # for each recipient
        for recipient_id in range(self.number_of_recipients):
            recipient_seed = (secret_key << shift) + recipient_id
            b = hashlib.blake2b(key=recipient_seed.to_bytes(6, 'little'),
                                digest_size=int(self.fingerprint_bit_length / 8))
            recipient_fp = BitArray(hex=b.hexdigest())
            recipient_fp = recipient_fp.bin
            if recipient_fp == fingerprint:
                return recipient_id
        return -1


    def insertion(self, dataset_name, recipient_id, secret_key, primary_key=None, outfile=None,
                  correlated_attributes=None):
        """
        Embeds a fingerprint into the data using NCorrFP algorithm.
        Args:
            dataset_name: string name of the predefined test dataset
            recipient_id: unique identifier of the recipient
            secret_key: owner's secret key
            primary_key: optional - name of the primary key attribute. If not specified, index is used
            outfile:  optional - name of the output file for writing fingerprinted dataset
            correlated_attributes: optional - list of names of correlated attributes. If multiple groups of correlated attributes exist, they collectivelly need to be passed as a list of lists. If not specified, all attributes will be used for neighbourhood search.

        Returns: fingerprinted dataset (pandas.DataFrame)

        """
        print("Start the NCorr fingerprint insertion algorithm...")
        print("\tgamma: " + str(self.gamma) + "\n\tcorrelated attributes: " + str(correlated_attributes))
        if secret_key is not None:
            self.secret_key = secret_key
        # it is assumed that the first column in the dataset is the primary key
        relation, primary_key = import_dataset_from_file(dataset_name, primary_key)
        # number of numerical attributes minus primary key
        number_of_num_attributes = len(relation.select_dtypes(exclude='object').columns) - 1
        # number of non-numerical attributes
        number_of_cat_attributes = len(relation.select_dtypes(include='object').columns)
        # total number of attributes
        tot_attributes = number_of_num_attributes + number_of_cat_attributes

        fingerprint = self.create_fingerprint(recipient_id, secret_key)
        print("\nGenerated fingerprint for recipient " + str(recipient_id) + ": " + fingerprint.bin)
        print("Inserting the fingerprint...\n")

        start = time.time()

        # label encoder
        categorical_attributes = relation.select_dtypes(include='object').columns
        label_encoders = dict()
        for cat in categorical_attributes:
            label_enc = LabelEncoder()
            relation[cat] = label_enc.fit_transform(relation[cat])
            label_encoders[cat] = label_enc

        correlated_attributes = parse_correlated_attrs(correlated_attributes, relation)
        # ball trees from user-specified correlated attributes
        balltree = init_balltrees(correlated_attributes, relation.drop('Id', axis=1))

        fingerprinted_relation = relation.copy()
        for r in relation.iterrows():
            # r[0] is an index of a row = primary key
            # seed = concat(secret_key, primary_key)
            seed = (self.secret_key << self.__primary_key_len) + r[1][0]  # first column must be primary ke
            random.seed(seed)

            # selecting the tuple
            if random.randint(0, _MAXINT) % self.gamma == 0:
                # selecting the attribute
                attr_idx = random.randint(0, _MAXINT) % tot_attributes + 1  # +1 to skip the prim key
                attr_name = r[1].index[attr_idx]
                attribute_val = r[1][attr_idx]

                # select fingerprint bit
                fingerprint_idx = random.randint(0, _MAXINT) % self.fingerprint_bit_length
                fingerprint_bit = fingerprint[fingerprint_idx]
                # select mask and calculate the mark bit
                mask_bit = random.randint(0, _MAXINT) % 2
                mark_bit = (mask_bit + fingerprint_bit) % 2

                marked_attribute = attribute_val
                # fp information: if mark_bit = fp_bit xor mask_bit is 1 then take the most frequent value,
                # # # otherwise one of the less frequent ones

                # selecting attributes for knn search -> this is user specified
                # get the index of a group of correlated attributes to attr; if attr is not correlated then return None
                corr_group_index = next((i for i, sublist in enumerate(correlated_attributes) if attr_name in sublist), None)
                # if attr_name in correlated_attributes:
                if corr_group_index is not None:
                    other_attributes = correlated_attributes[corr_group_index].tolist().copy()
                    other_attributes.remove(attr_name)
                    bt = balltree[attr_name]
                else:
                    other_attributes = r[1].index.tolist().copy()
                    other_attributes.remove(attr_name)
                    other_attributes.remove('Id')
                    bt = balltree[attr_name]
                if self.distance_based:
                    neighbours, dist = bt.query_radius([relation[other_attributes].loc[r[0]]], r=self.d,
                                                       return_distance=True, sort_results=True)
                else:
                    # nondeterminism - non chosen tuples with max distance
                    dist, neighbours = bt.query([relation[other_attributes].loc[r[0]]], k=self.k + 1)
                    dist = dist[0].tolist()
                    dist.remove(dist[0])
                    neighbours, dist = bt.query_radius(
                        [relation[other_attributes].loc[r[0]]], r=max(dist), return_distance=True,
                        sort_results=True)  # the list of neighbours is first (and only) element of the returned list
                    neighbours = neighbours[0].tolist()
                    neighbours.remove(neighbours[0])
                dist = dist[0].tolist()
                dist.remove(dist[0])

                # check the frequencies of the values
                possible_values = []
                for neighb in neighbours:
                    possible_values.append(relation.at[neighb, r[1].keys()[attr_idx]])
                frequencies = dict()
                if len(possible_values) != 0:
                    for value in set(possible_values):
                        f = possible_values.count(value) / len(possible_values)
                        frequencies[value] = f
                    # sort the values by their frequency
                    frequencies = {k: v for k, v in sorted(frequencies.items(), key=lambda item: item[1], reverse=True)}
                    if mark_bit == 0 and len(frequencies.keys()) > 1:
                        # choose among less frequent values, weighted by their frequencies
                        norm_freq = list(frequencies.values())[1:]/np.sum(list(frequencies.values())[1:])
                        marked_attribute = random.choice(list(frequencies.keys())[1:], 1,
                                                         p=norm_freq)[0]
                    else:  # choose the most frequent value
                        marked_attribute = list(frequencies.keys())[0]

                fingerprinted_relation.at[r[0], r[1].keys()[attr_idx]] = marked_attribute

        # delabeling
        for cat in categorical_attributes:
            fingerprinted_relation[cat] = label_encoders[cat].inverse_transform(fingerprinted_relation[cat])

        print("Fingerprint inserted.")
        if secret_key is None:
            write_dataset(fingerprinted_relation, "categorical_neighbourhood", "blind/" + dataset_name,
                          [self.gamma, self.xi],
                          recipient_id)
        runtime = time.time() - start
        if runtime < 1:
            print("Runtime: " + str(round(runtime*1000, 2)) + " ms.")
        else:
            print("Runtime: " + str(round(runtime, 2)) + " sec.")
        if outfile is not None:
            fingerprinted_relation.to_csv(outfile, index=False)
        return fingerprinted_relation

    def detection(self, dataset, secret_key, primary_key=None, correlated_attributes=None, original_columns=None):
        """

        Args:
            dataset:
            secret_key:
            primary_key:
            correlated_attributes:
            original_columns:

        Returns:
            recipient_no: identification of the recipient of detected fingeprint
            detected_fp: detected fingerprint bitstring
            count: array of bit-wise fingerprint votes

        """
        print("Start NCorr fingerprint detection algorithm ...")
        print("\tgamma: " + str(self.gamma) + "\n\tcorrelated attributes: " + str(correlated_attributes))

        relation_fp = _read_data(dataset)
        indices = list(relation_fp.dataframe.index)
        # number of numerical attributes minus primary key
        number_of_num_attributes = len(relation_fp.dataframe.select_dtypes(exclude='object').columns) - 1
        number_of_cat_attributes = len(relation_fp.dataframe.select_dtypes(include='object').columns)
        tot_attributes = number_of_num_attributes + number_of_cat_attributes
        categorical_attributes = relation_fp.dataframe.select_dtypes(include='object').columns

        attacked_columns = []
        if original_columns is not None:  # aligning with original schema (in case of vertical attacks)
            if "Id" in original_columns:
                original_columns.remove("Id")
            for orig_col in original_columns:
                if orig_col not in relation_fp.columns:
                    # fill in
                    relation_fp.dataframe[orig_col] = 0
                    attacked_columns.append(orig_col)
            # rearrange in the original order
            relation_fp.dataframe = relation_fp.dataframe[["Id"] + original_columns]
            tot_attributes += len(attacked_columns)

        # encode the categorical values
        label_encoders = dict()
        for cat in categorical_attributes:
            label_enc = LabelEncoder()
            relation_fp.dataframe[cat] = label_enc.fit_transform(relation_fp.dataframe[cat])
            label_encoders[cat] = label_enc

        start = time.time()
        correlated_attributes = parse_correlated_attrs(correlated_attributes, relation_fp.dataframe)
        balltree = init_balltrees(correlated_attributes, relation_fp.dataframe.drop('Id', axis=1))

        count = [[0, 0] for x in range(self.fingerprint_bit_length)]

        for r in relation_fp.dataframe.iterrows():
            seed = (self.secret_key << self.__primary_key_len) + r[1][0]  # primary key must be the first column
            random.seed(seed)
            # this tuple was marked
            if random.randint(0, _MAXINT) % self.gamma == 0:
                # this attribute was marked (skip the primary key)
                attr_idx = random.randint(0, _MAXINT) % tot_attributes + 1  # add 1 to skip the primary key
                attr_name = r[1].index[attr_idx]
                if attr_name in attacked_columns:  # if this columns was deleted by VA, don't collect the votes
                    continue
                attribute_val = r[1][attr_idx]
                # fingerprint bit
                fingerprint_idx = random.randint(0, _MAXINT) % self.fingerprint_bit_length
                # mask
                mask_bit = random.randint(0, _MAXINT) % 2

                corr_group_index = next((i for i, sublist in enumerate(correlated_attributes) if attr_name in sublist), None)
                if corr_group_index is not None:  # if attr_name is in correlated attributes
                    other_attributes = correlated_attributes[corr_group_index].tolist().copy()
                    other_attributes.remove(attr_name)
                    bt = balltree[attr_name]
                else:
                    other_attributes = r[1].index.tolist().copy()
                    other_attributes.remove(attr_name)
                    other_attributes.remove('Id')
                    bt = balltree[attr_name]
                if self.distance_based:
                    neighbours, dist = bt.query_radius([relation_fp[other_attributes].loc[r[1][0]]], r=self.d,
                                                       return_distance=True, sort_results=True)
                else:
                    # nondeterminism - non chosen tuples with max distance
                    dist, neighbours = bt.query([relation_fp.dataframe[other_attributes].loc[r[1][0]]], k=self.k + 1)
                    dist = dist[0].tolist()
                    dist.remove(dist[0])
                    neighbours, dist = bt.query_radius(
                        [relation_fp.dataframe[other_attributes].loc[r[0]]], r=max(dist), return_distance=True,
                        sort_results=True)  # the list of neighbours is first (and only) element of the returned list
                    neighbours = neighbours[0].tolist()
                    neighbours.remove(neighbours[0])
                dist = dist[0].tolist()
                dist.remove(dist[0])

                # check the frequencies of the values
                possible_values = []
                for neighb in neighbours:
                    neighb = indices[neighb]  # balltree resets the index so querying by index only fails for horizontal attacks, so we have to keep track of indices like this
                    possible_values.append(relation_fp.dataframe.at[neighb, r[1].keys()[attr_idx]])
                frequencies = dict()
                if len(possible_values) != 0:
                    for value in set(possible_values):
                        f = possible_values.count(value) / len(possible_values)
                        frequencies[value] = f
                    # sort the values by their frequency
                    frequencies = {k: v for k, v in
                                   sorted(frequencies.items(), key=lambda item: item[1], reverse=True)}
                if attribute_val == list(frequencies.keys())[0]:
                    mark_bit = 1
                else:
                    mark_bit = 0
                fingerprint_bit = (mark_bit + mask_bit) % 2
                count[fingerprint_idx][fingerprint_bit] += 1

        # this fingerprint template will be upside-down from the real binary representation
        fingerprint_template = [2] * self.fingerprint_bit_length
        # recover fingerprint
        for i in range(self.fingerprint_bit_length):
            # certainty of a fingerprint value
            T = 0.50
            if count[i][0] + count[i][1] != 0:
                if count[i][0] / (count[i][0] + count[i][1]) > T:
                    fingerprint_template[i] = 0
                elif count[i][1] / (count[i][0] + count[i][1]) > T:
                    fingerprint_template[i] = 1

        fingerprint_template_str = ''.join(map(str, fingerprint_template))
        self.detected_fp = list_to_string(fingerprint_template)
        print("Fingerprint detected: " + self.detected_fp)
        self.count = count
        print(count)

        recipient_no = self.detect_potential_traitor(fingerprint_template_str, secret_key)
        if recipient_no >= 0:
            print("Fingerprint belongs to Recipient " + str(recipient_no))
        else:
            print("None suspected.")
        runtime = time.time() - start
        if runtime < 1:
            print("Runtime: " + str(int(runtime)*1000) + " ms.")
        else:
            print("Runtime: " + str(round(runtime, 2)) + " sec.")
        return recipient_no

    def demo_insertion(self, dataset_name, recipient_id, secret_key, primary_key=None, outfile=None,
                       correlated_attributes=None):
        print("Start the demo NCorr fingerprint insertion algorithm...")
        print("\tgamma: " + str(self.gamma) + "\n\tcorrelated attributes: " + str(correlated_attributes))
        if secret_key is not None:
            self.secret_key = secret_key
        # it is assumed that the first column in the dataset is the primary key
        relation, primary_key = import_dataset_from_file(dataset_name, primary_key)
        # number of numerical attributes minus primary key
        number_of_num_attributes = len(relation.select_dtypes(exclude='object').columns) - 1
        # number of non-numerical attributes
        number_of_cat_attributes = len(relation.select_dtypes(include='object').columns)
        # total number of attributes
        tot_attributes = number_of_num_attributes + number_of_cat_attributes

        fingerprint = self.create_fingerprint(recipient_id, secret_key)
        print("\nGenerated fingerprint for recipient " + str(recipient_id) + ": " + fingerprint.bin)
        print("Inserting the fingerprint...\n")

        start = time.time()

        # label encoder
        categorical_attributes = relation.select_dtypes(include='object').columns
        label_encoders = dict()
        for cat in categorical_attributes:
            label_enc = LabelEncoder()
            relation[cat] = label_enc.fit_transform(relation[cat])
            label_encoders[cat] = label_enc

        # ball trees from user-specified correlated attributes
        if correlated_attributes is None:
            correlated_attributes = relation.columns[:]  # everything is correlated if not otherwise specified
        else:
            correlated_attributes = pd.Index(correlated_attributes)

        start_training_balltrees = time.time()
        # ball trees from all-except-one attribute and all attributes
        balltree = dict()
        for i in range(len(correlated_attributes)):
            balltree_i = BallTree(relation[correlated_attributes[:i].append(correlated_attributes[(i + 1):])],
                                  metric="hamming")
            balltree[correlated_attributes[i]] = balltree_i
        balltree_all = BallTree(relation[correlated_attributes], metric="hamming")
        balltree["all"] = balltree_all
        print("Training balltrees in: " + str(round(time.time() - start_training_balltrees, 2)) + " sec.")

        fingerprinted_relation = relation.copy()
        iter_log = []
        for r in relation.iterrows():
            # r[0] is an index of a row = primary key
            # seed = concat(secret_key, primary_key)
            seed = (self.secret_key << self.__primary_key_len) + r[1][0]  # first column must be primary ke
            random.seed(seed)

            # selecting the tuple
            if random.randint(0, _MAXINT) % self.gamma == 0:
                iteration = {'row_index': r[1][0]}
                # selecting the attribute
                attr_idx = random.randint(0, _MAXINT) % tot_attributes + 1 # +1 to skip the prim key
                attr_name = r[1].index[attr_idx]
                attribute_val = r[1][attr_idx]
                iteration['attribute'] = attr_name

                # select fingerprint bit
                fingerprint_idx = random.randint(0, _MAXINT) % self.fingerprint_bit_length
                fingerprint_bit = fingerprint[fingerprint_idx]
                # select mask and calculate the mark bit
                mask_bit = random.randint(0, _MAXINT) % 2
                mark_bit = (mask_bit + fingerprint_bit) % 2
                iteration['mark_bit'] = mark_bit

                marked_attribute = attribute_val
                # fp information: if mark_bit = fp_bit xor mask_bit is 1 then take the most frequent value,
                # # # otherwise the second most frequent

                # selecting attributes for knn search -> this is user specified
                if attr_name in correlated_attributes:
                    other_attributes = correlated_attributes.tolist().copy()
                    other_attributes.remove(attr_name)
                    bt = balltree[attr_name]
                else:
                    other_attributes = correlated_attributes.tolist().copy()
                    bt = balltree["all"]
                if self.distance_based:
                    neighbours, dist = bt.query_radius([relation[other_attributes].loc[r[0]]], r=self.d,
                                                       return_distance=True, sort_results=True)
                else:
                    # nondeterminism - non chosen tuples with max distance
                    dist, neighbours = bt.query([relation[other_attributes].loc[r[0]]], k=self.k + 1)
                # for demo:
                # print("Neighbors: " + str(neighbours) + " at distance " + str(dist))
                    dist = dist[0].tolist()
                    dist.remove(dist[0])
                    # resolve the non-determinism take all the tuples with max distance
                    neighbours, dist = bt.query_radius(
                        [relation[other_attributes].loc[r[0]]], r=max(dist), return_distance=True,
                        sort_results=True)
                    neighbours = neighbours[0].tolist()
                    neighbours.remove(neighbours[0])
                dist = dist[0].tolist()
                dist.remove(dist[0])
                iteration['neighbors'] = neighbours
                iteration['dist'] = dist

                # check the frequencies of the values
                possible_values = []
                for neighb in neighbours:
                    possible_values.append(relation.at[neighb, r[1].keys()[attr_idx]])
                frequencies = dict()
                if len(possible_values) != 0:
                    for value in set(possible_values):
                        f = possible_values.count(value) / len(possible_values)
                        frequencies[value] = f
                    # sort the values by their frequency
                    frequencies = {k: v for k, v in sorted(frequencies.items(), key=lambda item: item[1], reverse=True)}
                    if mark_bit == 0 and len(frequencies.keys()) > 1:
                        # choose among less frequent values, weighted by their frequencies
                        norm_freq = list(frequencies.values())[1:] / np.sum(list(frequencies.values())[1:])
                        marked_attribute = random.choice(list(frequencies.keys())[1:], 1,
                                                         p=norm_freq)[0]
                    else:  # choose the most frequent value
                        marked_attribute = list(frequencies.keys())[0]

                if attr_name in categorical_attributes:
                    iteration['frequencies'] = dict()
                    for (k, val) in frequencies.items():
                        decoded_k = label_encoders[attr_name].inverse_transform([k])
                        iteration['frequencies'][decoded_k[0]] = val
                else:
                    iteration['frequencies'] = frequencies
                iteration['new_value'] = marked_attribute
                # print("Index " + str(r[0]) + ", attribute " + str(r[1].keys()[attr_idx]) + ", from " +
                #      str(attribute_val) + " to " + str(marked_attribute))
                fingerprinted_relation.at[r[0], r[1].keys()[attr_idx]] = marked_attribute
                iter_log.append(iteration)
                #   STOPPING DEMO
                # if iteration['id'] == 1:
                #    exit()

        # delabeling
        for cat in categorical_attributes:
            fingerprinted_relation[cat] = label_encoders[cat].inverse_transform(fingerprinted_relation[cat])

        print("Fingerprint inserted.")
        if secret_key is None:
            write_dataset(fingerprinted_relation, "categorical_neighbourhood", "blind/" + dataset_name,
                          [self.gamma, self.xi],
                          recipient_id)
        runtime = time.time() - start
        if runtime < 1:
            print("Runtime: " + str(int(runtime) * 1000) + " ms.")
        else:
            print("Runtime: " + str(round(runtime, 2)) + " sec.")
        if outfile is not None:
            fingerprinted_relation.to_csv(outfile, index=False)
        return fingerprinted_relation, iter_log

    def demo_detection(self, dataset, secret_key, primary_key=None, correlated_attributes=None, original_columns=None):
        print("Start demo NCorr fingerprint detection algorithm ...")
        print("\tgamma: " + str(self.gamma) + "\n\tcorrelated attributes: " + str(correlated_attributes))

        relation_fp = _read_data(dataset)
        indices = list(relation_fp.dataframe.index)
        # number of numerical attributes minus primary key
        number_of_num_attributes = len(relation_fp.dataframe.select_dtypes(exclude='object').columns) - 1
        number_of_cat_attributes = len(relation_fp.dataframe.select_dtypes(include='object').columns)
        tot_attributes = number_of_num_attributes + number_of_cat_attributes
        categorical_attributes = relation_fp.dataframe.select_dtypes(include='object').columns

        attacked_columns = []
        if original_columns is not None:  # aligning with original schema (in case of vertical attacks)
            if "Id" in original_columns:
                original_columns.remove("Id")  # just in case
            for orig_col in original_columns:
                if orig_col not in relation_fp.columns:
                    # fill in
                    relation_fp.dataframe[orig_col] = 0
                    attacked_columns.append(orig_col)
            # rearrange in the original order
            relation_fp.dataframe = relation_fp.dataframe[["Id"] + original_columns]
            tot_attributes += len(attacked_columns)

        # encode the categorical values
        label_encoders = dict()
        for cat in categorical_attributes:
            label_enc = LabelEncoder()
            relation_fp.dataframe[cat] = label_enc.fit_transform(relation_fp.dataframe[cat])
            label_encoders[cat] = label_enc

        start = time.time()

        start_balling = time.time()
        # ball trees from all-except-one attribute and all attributes
        if correlated_attributes is None:
            correlated_attributes = relation_fp.columns[:]  # everything is correlated if not otherwise specified
        else:
            correlated_attributes = pd.Index(correlated_attributes)
        balltree = dict()
        for i in range(len(correlated_attributes)):
            balltree_i = BallTree(relation_fp.dataframe[correlated_attributes[:i].append(correlated_attributes[(i + 1):])],
                                  metric="hamming")
            balltree[correlated_attributes[i]] = balltree_i
        balltree_all = BallTree(relation_fp.dataframe[correlated_attributes], metric="hamming")
        balltree["all"] = balltree_all

        count = [[0, 0] for x in range(self.fingerprint_bit_length)]
        iter_log = []
        for r in relation_fp.dataframe.iterrows():
            seed = (self.secret_key << self.__primary_key_len) + r[1][0]  # primary key must be the first column
            random.seed(seed)
            # this tuple was marked
            if random.randint(0, _MAXINT) % self.gamma == 0:
                iteration = {'row_index': r[1][0]}
                # this attribute was marked (skip the primary key)
                attr_idx = random.randint(0, _MAXINT) % tot_attributes + 1  # add 1 to skip the primary key
                attr_name = r[1].index[attr_idx]
                if attr_name in attacked_columns:  # if this columns was deleted by VA, don't collect the votes
                    continue
                iteration['attribute'] = attr_name
                attribute_val = r[1][attr_idx]
                # fingerprint bit
                fingerprint_idx = random.randint(0, _MAXINT) % self.fingerprint_bit_length
                iteration['fingerprint_idx'] = fingerprint_idx
                # mask
                mask_bit = random.randint(0, _MAXINT) % 2

                if attr_name in correlated_attributes:
                    other_attributes = correlated_attributes.tolist().copy()
                    other_attributes.remove(attr_name)
                    bt = balltree[attr_name]
                else:
                    other_attributes = correlated_attributes.tolist().copy()
                    bt = balltree["all"]
                if self.distance_based:
                    neighbours, dist = bt.query_radius([relation_fp[other_attributes].loc[r[1][0]]], r=self.d,
                                                       return_distance=True, sort_results=True)
                else:
                    # nondeterminism - non chosen tuples with max distance
                    dist, neighbours = bt.query([relation_fp.dataframe[other_attributes].loc[r[1][0]]], k=self.k + 1)
                # excluding the observed tuple - todo: dont exclude
                # neighbours.remove(neighbours[0])
                    dist = dist[0].tolist()
                    dist.remove(dist[0])
                    # resolve the non-determinism take all the tuples with max distance
                    neighbours, dist = bt.query_radius(
                        [relation_fp.dataframe[other_attributes].loc[r[0]]], r=max(dist), return_distance=True,
                        sort_results=True)
                    neighbours = neighbours[0].tolist()
                    neighbours.remove(neighbours[0])
                dist = dist[0].tolist()
                dist.remove(dist[0])
                iteration['neighbors'] = neighbours
                iteration['dist'] = dist

                # check the frequencies of the values
                possible_values = []
                for neighb in neighbours:
                    neighb = indices[neighb]  # balltree resets the index so querying by index only fails for horizontal attacks, so we have to keep track of indices like this
                    possible_values.append(relation_fp.dataframe.at[neighb, r[1].keys()[attr_idx]])
                frequencies = dict()
                if len(possible_values) != 0:
                    for value in set(possible_values):
                        f = possible_values.count(value) / len(possible_values)
                        frequencies[value] = f
                    # sort the values by their frequency
                    frequencies = {k: v for k, v in
                                   sorted(frequencies.items(), key=lambda item: item[1], reverse=True)}
                if attribute_val == list(frequencies.keys())[0]:
                    mark_bit = 1
                else:
                    mark_bit = 0

                if attr_name in categorical_attributes:
                    iteration['frequencies'] = dict()
                    for (k, val) in frequencies.items():
                        decoded_k = label_encoders[attr_name].inverse_transform([k])
                        iteration['frequencies'][decoded_k[0]] = val
                else:
                    iteration['frequencies'] = frequencies
                iteration['mark_bit'] = mark_bit

                fingerprint_bit = (mark_bit + mask_bit) % 2
                count[fingerprint_idx][fingerprint_bit] += 1
                if r[1][0] in [49, 99, 199, 299, 399, 499, 599, 699]:
                    print(count)
#                print(count)
                iteration['count_state'] = count  # this returns the final counts for each step ??
#                print(iteration['count_state'])
                iteration['mask_bit'] = mask_bit
                iteration['fingerprint_bit'] = fingerprint_bit

                iter_log.append(iteration)

        # this fingerprint template will be upside-down from the real binary representation
        fingerprint_template = [2] * self.fingerprint_bit_length
        # recover fingerprint
        for i in range(self.fingerprint_bit_length):
            # certainty of a fingerprint value
            T = 0.50
            if count[i][0] + count[i][1] != 0:
                if count[i][0] / (count[i][0] + count[i][1]) > T:
                    fingerprint_template[i] = 0
                elif count[i][1] / (count[i][0] + count[i][1]) > T:
                    fingerprint_template[i] = 1

        fingerprint_template_str = ''.join(map(str, fingerprint_template))
        print("Fingerprint detected: " + list_to_string(fingerprint_template))
        # print(count)

        recipient_no = self.detect_potential_traitor(fingerprint_template_str, secret_key)
        if recipient_no >= 0:
            print("Fingerprint belongs to Recipient " + str(recipient_no))
        else:
            print("None suspected.")
        runtime = time.time() - start
        if runtime < 1:
            print("Runtime: " + str(int(runtime) * 1000) + " ms.")
        else:
            print("Runtime: " + str(round(runtime, 2)) + " sec.")
        return recipient_no, iter_log

    def detection_temp(self, dataset, secret_key, primary_key=None, correlated_attributes=None, original_columns=None):
        """

        Args:
            dataset:
            secret_key:
            primary_key:
            correlated_attributes:
            original_columns:

        Returns:
            recipient_no: identification of the recipient of detected fingeprint
            detected_fp: detected fingerprint bitstring
            count: array of bit-wise fingerprint votes

        """
        print("Start NCorr fingerprint detection algorithm ...")
        print("\tgamma: " + str(self.gamma) + "\n\tcorrelated attributes: " + str(correlated_attributes))

        relation_fp = _read_data(dataset)
        indices = list(relation_fp.dataframe.index)
        # number of numerical attributes minus primary key
        number_of_num_attributes = len(relation_fp.dataframe.select_dtypes(exclude='object').columns) - 1
        number_of_cat_attributes = len(relation_fp.dataframe.select_dtypes(include='object').columns)
        tot_attributes = number_of_num_attributes + number_of_cat_attributes
        categorical_attributes = relation_fp.dataframe.select_dtypes(include='object').columns

        start = time.time()

        start_balling = time.time()
        # ball trees from all-except-one attribute and all attributes

        for r in relation_fp.dataframe.iterrows():
            seed = (self.secret_key << self.__primary_key_len) + r[1][0]  # primary key must be the first column
            random.seed(seed)
            # this tuple was marked
            if random.randint(0, _MAXINT) % self.gamma == 0:
                # this attribute was marked (skip the primary key)
                attr_idx = random.randint(0, _MAXINT) % tot_attributes + 1  # add 1 to skip the primary key

        runtime = time.time() - start
        if runtime < 1:
            print("Runtime: " + str(int(runtime)*1000) + " ms.")
        else:
            print("Runtime: " + str(round(runtime, 2)) + " sec.")
        return True