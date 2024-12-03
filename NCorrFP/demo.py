from utils import *
from .NCorrFP import *
from sklearn.preprocessing import LabelEncoder
from matplotlib import colors

_MAXINT = 2 ** 31 - 1


class Demo():
    __primary_key_len = 20

    def __init__(self, scheme):
        self.scheme = scheme
        self.secret_key = scheme.secret_key
        self.recipient_id = scheme.recipient_id
        self.fingerprint = scheme.fingerprint
        self.fingerprinted_data = scheme.relation_fp.dataframe

        self.insertion_iter_log = scheme.insertion_iter_log
        self.detection_iter_log = scheme.detection_iter_log

        self.data = scheme.dataset
        self.primary_key_name = scheme.dataset.primary_key_attribute
        self.dataset_name = scheme.dataset.name  # or path
        self.correlated_attributes = scheme.dataset.correlated_attributes
        self.original_columns = scheme.dataset.columns
        self.relation = scheme.dataset.dataframe

    def eval(self, data, correlated_attributes, primary_key_name='Id', secret_key=101, recipient_id=4,
             show_messages=True):
        if self.scheme is not None:
            self.dataset_name = data
            self.correlated_attributes = correlated_attributes
            self.primary_key_name = primary_key_name
            self.secret_key = secret_key
            self.recipient_id = recipient_id

            self.insertion(show_messages)
            self.detection(show_messages)
        if show_messages:
            print('DONE!')

    def insertion(self, show_messagess=True, optimise=True):
        if show_messagess:
            print("Start the demo NCorr fingerprint insertion algorithm...")
            print("\tgamma: " + str(self.scheme.gamma) + "\n\tcorrelated attributes: " + str(self.correlated_attributes))
        # it is assumed that the first column in the dataset is the primary key
        self.relation, primary_key = import_dataset_from_file(self.dataset_name, self.primary_key_name)
        relation = self.relation
        # number of numerical attributes minus primary key
        number_of_num_attributes = len(relation.select_dtypes(exclude='object').columns) - 1
        # number of non-numerical attributes
        number_of_cat_attributes = len(relation.select_dtypes(include='object').columns)
        # total number of attributes
        tot_attributes = number_of_num_attributes + number_of_cat_attributes

        fingerprint = self.scheme.create_fingerprint(self.recipient_id, self.secret_key)
        if show_messagess:
            print("\nGenerated fingerprint for recipient " + str(self.recipient_id) + ": " + fingerprint.bin)
            print("Inserting the fingerprint...\n")

        start = time.time()
        self.original_columns = relation.drop('Id', axis=1).columns
        # label encoder
        categorical_attributes = relation.select_dtypes(include='object').columns
        label_encoders = dict()
        for cat in categorical_attributes:
            label_enc = LabelEncoder()
            relation[cat] = label_enc.fit_transform(relation[cat])
            label_encoders[cat] = label_enc

        correlated_attributes = parse_correlated_attrs(self.correlated_attributes, relation)
        # ball trees from user-specified correlated attributes
        balltree = init_balltrees(correlated_attributes, relation.drop('Id', axis=1),
                                  self.scheme.dist_metric_discrete, self.scheme.dist_metric_continuous,
                                  categorical_attributes)

        fingerprinted_relation = relation.copy()
        iter_log = []
        embeddings_count = 0
        for r in relation.iterrows():
            # r[0] is an index of a row = primary key
            # seed = concat(secret_key, primary_key)
            seed = (self.secret_key << self.__primary_key_len) + r[1].iloc[0]  # first column must be primary key
            random.seed(seed)
            # selecting the tuple
            if random.choices([0, 1], [1/self.scheme.gamma, 1-1/self.scheme.gamma]) == [0]:
                iteration = {'seed': seed, 'row_index': r[1].iloc[0]}
                # selecting the attribute
                attr_idx = random.randint(0, _MAXINT) % tot_attributes + 1  # +1 to skip the prim key
                attr_name = r[1].index[attr_idx]
                attribute_val = r[1][attr_idx]
                iteration['attribute'] = attr_name

                # select fingerprint bit
                fingerprint_idx = random.randint(0, _MAXINT) % self.scheme.fingerprint_bit_length
                fingerprint_bit = fingerprint[fingerprint_idx]
                # select mask and calculate the mark bit
                mask_bit = random.randint(0, _MAXINT) % 2
                iteration['mask_bit'] = mask_bit
                mark_bit = (mask_bit + fingerprint_bit) % 2
                iteration['mark_bit'] = mark_bit

                marked_attribute = attribute_val
                # fp information: if mark_bit = fp_bit xor mask_bit is 1 then take the most frequent value,
                # # # otherwise the second most frequent

                # selecting attributes for knn search -> this is user specified
                # get the index of a group of correlated attributes to attr; if attr is not correlated then return None
                corr_group_index = next((i for i, sublist in enumerate(correlated_attributes) if attr_name in sublist),
                                        None)
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
                if self.scheme.distance_based:
                    neighbours, dist = bt.query_radius([relation[other_attributes].loc[r[0]]], r=self.scheme.d,
                                                       return_distance=True, sort_results=True)

                else:
                    if not optimise:
                        # nondeterminism - non chosen tuples with max distance
                        dist, neighbours = bt.query([relation[other_attributes].loc[r[0]]], k=self.scheme.k)
                        dist = dist[0].tolist()
                        radius = np.ceil(max(dist) * 10 ** 6) / 10 ** 6  # ceil the float max dist to 6th decimal
                        neighbours, dist = bt.query_radius(
                            [relation[other_attributes].loc[r[0]]], r=radius, return_distance=True,
                            sort_results=True)  # the list of neighbours is first (and only) element of the returned list
                        neighbours = neighbours[0].tolist()
                    else:
                        dist, neighbours = bt.query([relation[other_attributes].loc[r[0]]], k=3*self.scheme.k)  # query with extra neighbourhs
                        k_dist = dist[0][self.scheme.k - 1]  # Distance of the kth nearest neighbor
                        neighbours = neighbours[0][dist[0] <= k_dist]  # Get k neighbours plus the ties

                dist = dist[0].tolist()
                iteration['neighbors'] = neighbours
                iteration['dist'] = dist

                neighbourhood = relation.iloc[neighbours][attr_name].tolist()
                if attr_name in categorical_attributes:
                    marked_attribute = mark_categorical_value(neighbourhood, mark_bit)
                else:
                    marked_attribute = mark_continuous_value(neighbourhood, mark_bit, seed=seed)

                #                if attr_name in categorical_attributes:
                #                    iteration['frequencies'] = dict()
                #                    for (k, val) in frequencies.items():
                #                        decoded_k = label_encoders[attr_name].inverse_transform([k])
                #                        iteration['frequencies'][decoded_k[0]] = val
                #                else:
                #                    iteration['frequencies'] = frequencies
                iteration['new_value'] = marked_attribute
                # print("Index " + str(r[0]) + ", attribute " + str(r[1].keys()[attr_idx]) + ", from " +
                #      str(attribute_val) + " to " + str(marked_attribute))
                fingerprinted_relation.at[r[0], r[1].keys()[attr_idx]] = marked_attribute
                iter_log.append(iteration)
                #   STOPPING DEMO
                # if iteration['id'] == 1:
                #    exit()
                embeddings_count += 1

        # delabeling
        for cat in categorical_attributes:
            fingerprinted_relation[cat] = label_encoders[cat].inverse_transform(fingerprinted_relation[cat])

        if show_messagess:
            print("Fingerprint inserted.")
        if self.secret_key is None:
            write_dataset(fingerprinted_relation, "categorical_neighbourhood", "blind/" + self.dataset_name,
                          [self.scheme.gamma, self.scheme.xi],
                          self.recipient_id)
        runtime = time.time() - start
        if show_messagess:
            if runtime < 1:
                print("Runtime: " + str(int(runtime) * 1000) + " ms.")
            else:
                print("Runtime: " + str(round(runtime, 2)) + " sec.")
        #if outfile is not None:
        #    fingerprinted_relation.to_csv(outfile, index=False)
        self.fingerprinted_data = fingerprinted_relation
        self.iter_log = iter_log
        self.fingerprint = [int(bit) for bit in fingerprint.bin]
        self.embeddings_count = embeddings_count
        return fingerprinted_relation, iter_log


    def detection(self, show_messages=True, optimise=True):
        if show_messages:
            print("Start demo NCorr fingerprint detection algorithm ...")
            print("\tgamma: " + str(self.scheme.gamma) + "\n\tcorrelated attributes: " + str(self.correlated_attributes))

        relation_fp = self.fingerprinted_data
        # indices = list(relation_fp.dataframe.index)
        # number of numerical attributes minus primary key
        number_of_num_attributes = len(relation_fp.select_dtypes(exclude='object').columns) - 1
        number_of_cat_attributes = len(relation_fp.select_dtypes(include='object').columns)
        tot_attributes = number_of_num_attributes + number_of_cat_attributes
        categorical_attributes = relation_fp.select_dtypes(include='object').columns

        attacked_columns = []
        if self.original_columns is not None:  # aligning with original schema (in case of vertical attacks)
            if "Id" in self.original_columns:
                self.original_columns.remove("Id")  # just in case
            for orig_col in self.original_columns:
                if orig_col not in relation_fp.columns:
                    # fill in
                    relation_fp[orig_col] = 0
                    attacked_columns.append(orig_col)
            # rearrange in the original order
            relation_fp = relation_fp[["Id"] + list(self.original_columns)]
            tot_attributes += len(attacked_columns)

        # encode the categorical values
        label_encoders = dict()
        for cat in categorical_attributes:
            label_enc = LabelEncoder()
            relation_fp[cat] = label_enc.fit_transform(relation_fp[cat])
            label_encoders[cat] = label_enc

        start = time.time()
        correlated_attributes = parse_correlated_attrs(self.correlated_attributes, relation_fp)
        balltree = init_balltrees(correlated_attributes, relation_fp.drop('Id', axis=1),
                                  self.scheme.dist_metric_discrete, self.scheme.dist_metric_continuous,
                                  categorical_attributes)

        count = [[0, 0] for x in range(self.scheme.fingerprint_bit_length)]
        iter_log = []
        for r in relation_fp.iterrows():
            seed = (self.secret_key << self.__primary_key_len) + r[1].iloc[0]  # primary key must be the first column
            random.seed(seed)
            # this tuple was marked
            if random.randint(0, _MAXINT) % self.scheme.gamma == 0:
                iteration = {'seed': seed, 'row_index': r[1].iloc[0]}
                # this attribute was marked (skip the primary key)
                attr_idx = random.randint(0, _MAXINT) % tot_attributes + 1  # add 1 to skip the primary key
                attr_name = r[1].index[attr_idx]
                if attr_name in attacked_columns:  # if this columns was deleted by VA, don't collect the votes
                    continue
                iteration['attribute'] = attr_name
                attribute_val = r[1][attr_idx]
                iteration['attribute_val'] = attribute_val
                # fingerprint bit
                fingerprint_idx = random.randint(0, _MAXINT) % self.scheme.fingerprint_bit_length
                iteration['fingerprint_idx'] = fingerprint_idx
                # mask
                mask_bit = random.randint(0, _MAXINT) % 2
                iteration['mask_bit'] = mask_bit

                corr_group_index = next((i for i, sublist in enumerate(correlated_attributes) if attr_name in sublist),
                                        None)
                if corr_group_index is not None:  # if attr_name is in correlated attributes
                    other_attributes = correlated_attributes[corr_group_index].tolist().copy()
                    other_attributes.remove(attr_name)
                    bt = balltree[attr_name]
                else:
                    other_attributes = r[1].index.tolist().copy()
                    other_attributes.remove(attr_name)
                    other_attributes.remove('Id')
                    bt = balltree[attr_name]
                if self.scheme.distance_based:
                    neighbours, dist = bt.query_radius([relation_fp[other_attributes].loc[r[1].iloc[0]]], r=self.scheme.d,
                                                       return_distance=True, sort_results=True)
                else:
                    if not optimise:
                        # find the neighborhood of cardinality k (non-deterministic)
                        dist, neighbours = bt.query([relation_fp[other_attributes].loc[r[1].iloc[0]]], k=self.scheme.k)
                        dist = dist[0].tolist()
                        # solve nondeterminism: get all other elements of max distance in the neighbourhood
                        radius = np.ceil(max(dist) * 10 ** 6) / 10 ** 6  # ceil the float max dist to 6th decimal
                        neighbours, dist = bt.query_radius(
                            [relation_fp[other_attributes].loc[r[1].iloc[0]]], r=radius, return_distance=True,
                            sort_results=True)
                        neighbours = neighbours[
                            0].tolist()  # the list of neighbours was first (and only) element of the returned list
                        dist = dist[0].tolist()
                    else:
                        dist, neighbours = bt.query([relation_fp[other_attributes].loc[r[0]]], k=3*self.scheme.k)
                        k_dist = dist[0][self.scheme.k - 1]
                        neighbours = neighbours[0][dist[0] <= k_dist]  # get k neighbours plus the ties

                iteration['neighbors'] = neighbours
                iteration['dist'] = dist

                # check the frequencies of the values
                neighbourhood = relation_fp.iloc[neighbours][attr_name].tolist()
                mark_bit = get_mark_bit(is_categorical=(attr_name in categorical_attributes),
                                        attribute_val=attribute_val, neighbours=neighbourhood,
                                        relation_fp=relation_fp, attr_name=attr_name)

                fingerprint_bit = (mark_bit + mask_bit) % 2
                count[fingerprint_idx][fingerprint_bit] += 1

                iteration['count_state'] = copy.deepcopy(count)  # this returns the final counts for each step ??
                iteration['mark_bit'] = mark_bit
                iteration['fingerprint_bit'] = fingerprint_bit

                iter_log.append(iteration)

        # this fingerprint template will be upside-down from the real binary representation
        fingerprint_template = [2] * self.scheme.fingerprint_bit_length
        # recover fingerprint
        for i in range(self.scheme.fingerprint_bit_length):
            # certainty of a fingerprint value
            T = 0.50
            if count[i][0] + count[i][1] != 0:
                if count[i][0] / (count[i][0] + count[i][1]) > T:
                    fingerprint_template[i] = 0
                elif count[i][1] / (count[i][0] + count[i][1]) > T:
                    fingerprint_template[i] = 1

        fingerprint_template_str = ''.join(map(str, fingerprint_template))
        if show_messages:
            print("Fingerprint detected: " + list_to_string(fingerprint_template))
        # print(count)

        recipient_no = self.scheme.detect_potential_traitor(fingerprint_template_str, self.secret_key)
        if show_messages:
            if recipient_no >= 0:
                print("Fingerprint belongs to Recipient " + str(recipient_no))
            else:
                print("None suspected.")
        runtime = time.time() - start
        if show_messages:
            if runtime < 1:
                print("Runtime: " + str(int(runtime) * 1000) + " ms.")
            else:
                print("Runtime: " + str(round(runtime, 2)) + " sec.")
        self.det_iter_log = iter_log
        return recipient_no, iter_log

    def show_embedding_iteration(self, iter):
        print("Marking record no. " + str(self.insertion_iter_log[iter]['row_index']))
        print("Marking attribute: " + str(self.insertion_iter_log[iter]['attribute']))
        print("The record to mark: \n" + str(self.relation.iloc[[iter]]))
        print(
            '------------------------------------------------------------------------------------------------------------------')
        if self.insertion_iter_log[iter]['attribute'] in self.correlated_attributes:
            other = list(self.correlated_attributes)
            other.remove(self.insertion_iter_log[iter]['attribute'])
            print('Neighbourhood: ' + str(self.insertion_iter_log[iter]['attribute']) + ' is correlated to ' + str(
                other) + ' so we are finding the records with most similar values to ' + str(other) + '=' + str(
                self.relation.iloc[iter][other[0]]))
        else:
            print('Neighbourhood: ' + str(self.insertion_iter_log[iter][
                                              'attribute']) + ' is not a correlated attribute, so we are including all attributes to find the closest neighbourhood.')
        print('Neighbours idx: ' + str(self.insertion_iter_log[iter]['neighbors']))
        print('Neighbours dist: ' + str(self.insertion_iter_log[iter]['dist']))
        #print('Neighbours values:' + str(list(self.relation.iloc[self.insertion_iter_log[iter]['neighbors']]['X'])))
        print('\nNow we look at the values of attribute ' + str(
            self.insertion_iter_log[iter]['attribute']) + ' in this neighbourhood, and among these is our potential new value.')
        print('Target values:' + str(list(self.relation.iloc[self.insertion_iter_log[iter]['neighbors']][self.insertion_iter_log[iter]['attribute']])))
        print(
            'For this we estimate the distribution of these target values (see the plot below) before sampling one new value.')
        print(
            'There are generally two outcomes for the sampled value:\n\t-mark bit is 1 (50%) - the new value is sampled from the most dense areas of a distribution of the target variable in the neighbourhood\n\t-mark bit is 0(50%) - the new value is sampled from the tails of distribution of the target value in the neighbourhood')
        print('Mark bit is generated pseudorandomly via PRNG.\n')
        mark_bit = self.insertion_iter_log[iter]['mark_bit']
        if mark_bit == 1:
            print(
                'In this case, mark bit is {}, therefore we sample from the dense part of distribution of the variable {} in the neighbourhood. The thresholds are set by an arbitrary percentile (here we use 75th)'.format(
                    mark_bit, self.insertion_iter_log[iter]['attribute']))
        else:
            print(
                'In this case, mark bit is {}, therefore we sample from tails of distribution of the variable {} in the neighbourhood. The thresholds are set by an arbitrary percentile (here we use 75th)'.format(
                    mark_bit, self.insertion_iter_log[iter]['attribute']))
        sample_from_area(data=list(self.relation.iloc[self.insertion_iter_log[iter]['neighbors']][self.insertion_iter_log[iter]['attribute']]), percent=0.75,
                         dense=mark_bit, plot=True, seed=self.insertion_iter_log[iter]['seed'])
        print(
            "The sampled continuous value is rounded to the closest existing value from the data (to avoid perceptibility of marks) and is: " + str(
                self.insertion_iter_log[iter]['new_value']))
        print("The fingerprinted record is:")
        print(self.fingerprinted_data.iloc[[iter]])

    def values_for_plot_distribution(self, target_values):
        data = target_values
        kde = gaussian_kde(data)
        # Create a range of values to evaluate the PDF
        x = np.linspace(min(data), max(data), 1000)
        pdf_values = kde(x)

        threshold = np.percentile(pdf_values, 0.75 * 100)

        mask = (pdf_values >= threshold)
        # Re-normalize the masked PDF and CDF
        masked_pdf = np.where(mask, pdf_values, 0)
        masked_cdf = np.cumsum(masked_pdf)
        masked_cdf /= masked_cdf[-1]

        return x, pdf_values, masked_pdf


    # for converting fingerprint to color pattern for plotting the count state
    def binary_to_pairs(self, binary_list):
        # Initialize an empty list to store the result
        result = []

        # Iterate over each element in the binary list
        for value in binary_list:
            if value == 0:
                # If the input value is 0, create the pair (1, 0)
                pair = (1, 0)
            else:
                # If the input value is 1, create the pair (0, 1)
                pair = (0, 1)
            # Append the pair to the result list
            result.append(pair)

        return result

    def plot_count_updates(self, count_state, fingerprint):
        # Extract the first and second values of each pair for the rows
        first_row = [pair[0] for pair in count_state]
        second_row = [pair[1] for pair in count_state]

        # Extract the color flags for each row
        color_flags = self.binary_to_pairs(fingerprint)  # 1 = green, 0 = red
        first_row_flags = [flag[0] for flag in color_flags]
        second_row_flags = [flag[1] for flag in color_flags]

        # Normalize values for color intensity
        all_values = first_row + second_row
        norm = colors.Normalize(vmin=min(all_values), vmax=max(all_values))

        # Create figure and axis
        fig, ax = plt.subplots(figsize=(len(count_state), 2))

        # Create the table with two rows (first row, second row)
        table_data = [first_row, second_row]
        table = ax.table(cellText=table_data, loc='center', cellLoc='center', colWidths=[0.05] * len(count_state))

        # Hide the axes
        ax.axis('off')

        # Iterate over the table cells to set the colors based on value and color flag
        for i in range(2):  # Two rows
            for j in range(len(count_state)):
                # Get the value for the cell
                value = table_data[i][j]

                # Determine the color based on the color flags
                if i == 0:  # First row
                    if first_row_flags[j] == 1:
                        color = plt.cm.Greens(norm(value))  # Green colormap if flag is 1
                    else:
                        color = plt.cm.Reds(norm(value))  # Red colormap if flag is 0
                else:  # Second row
                    if second_row_flags[j] == 1:
                        color = plt.cm.Greens(norm(value))  # Green colormap if flag is 1
                    else:
                        color = plt.cm.Reds(norm(value))  # Red colormap if flag is 0

                # Set the color of the cell
                cell = table[(i, j)]
                cell.set_facecolor(color)
                cell.set_text_props(color='black')

        # Display the plot
        plt.show()


    def show_detection_iteration(self, iteration):
        print("Detecting from record at idx: " + str(self.detection_iter_log[iteration]['row_index']))
        print("Detecting from attribute: " + str(self.detection_iter_log[iteration]['attribute']))
        print(self.fingerprinted_data.iloc[[iteration]])
        fingerprinted_value = self.fingerprinted_data.iloc[self.detection_iter_log[iteration]['row_index']][
            self.detection_iter_log[iteration]['attribute']]
        print('Fingerpritned value: ' + str(fingerprinted_value))
        target_values_det = self.fingerprinted_data.iloc[self.detection_iter_log[iteration]['neighbors']][
            self.detection_iter_log[iteration]['attribute']].tolist()
        print("----------------------------------------------------------")
 #       print("Obtaining neighbourhood....")
 #       print('Target values:' + str(target_values_det))
 #       print("----------------------------------------------------------")

        x_det, pdf_values_det, masked_pdf_det = self.values_for_plot_distribution(target_values_det)
        target_values_ins = list(self.relation.iloc[self.insertion_iter_log[iteration]['neighbors']][self.insertion_iter_log[iteration]['attribute']])
        x_ins, pdf_values_ins, masked_pdf_ins = self.values_for_plot_distribution(target_values_ins)

        fig, axs = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(10, 4))

        axs[0].plot(x_det, pdf_values_det, label='Original PDF\n (estimate)')
        axs[0].plot(x_det, masked_pdf_det, label='Modified PDF\n ({}th percentile)'.format(int(100 * 0.75)))
        axs[0].hist(target_values_det, bins=10, density=True, alpha=0.3, label='Neighbourhood\n data points')
        axs[0].scatter(fingerprinted_value, 0, color='red', label='Marked value', zorder=5)
        axs[0].set_ylabel('Density')
        axs[0].set_xlabel('Values')
        axs[0].set_title('Fingerprinted data')
        axs[0].legend(prop={'size': 8})

        axs[1].plot(x_ins, pdf_values_ins, label='Original PDF\n (estimate)')
        axs[1].plot(x_ins, masked_pdf_ins, label='Modified PDF\n ({}th percentile)'.format(int(100 * 0.75)))
        axs[1].hist(target_values_ins, bins=10, density=True, alpha=0.3, label='Neighbourhood\n data points')
        axs[1].scatter(self.insertion_iter_log[iteration]['new_value'], 0, color='red', label='Marked value', zorder=5)
        axs[1].set_ylabel('Density')
        axs[1].set_xlabel('Values')
        axs[1].set_title('Original data')
        axs[1].legend(prop={'size': 8})

        print(
            "\n--> Observing the distribution of target attribute {} below...".format(self.detection_iter_log[iteration]['attribute']))
        message = ' (i.e. tails of distribution)' if self.detection_iter_log[iteration]['mark_bit'] == 0 else ' (i.e. in densest area)'
        print("Mark bit (where in distribution falls the target value?): " + str(
            self.detection_iter_log[iteration]['mark_bit']) + message)
        print("Mask bit (from PRNG): " + str(self.detection_iter_log[iteration]['mask_bit']))
        print("Fingerprint bit index (from PRNG): " + str(self.detection_iter_log[iteration]['fingerprint_idx']))
        print("Fingerprint bit value (mark bit xor mask bit): " + str(self.detection_iter_log[iteration]['fingerprint_bit']))

        if self.detection_iter_log[iteration]['fingerprint_bit'] == self.fingerprint[self.detection_iter_log[iteration]['fingerprint_idx']]:
            print('\nFingerprint bit CORRECT :)')
        else:
            print('\nFingerprint bit FALSE :( (it is just a wrong vote)')

        self.plot_count_updates(self.detection_iter_log[iteration]['count_state'], self.fingerprint)
        print(
            'Table: fingerprint count updates after this iteration (iteration {}). Each column is one fingerprint bit position (e.g. 16-bit --> 16 columns), and each row represents votes for either 0 or 1 being the value of that bit. The final decision is made at the end af the detection algorithm according to majority vote.'.format(
                iteration))

    def get_error_iterations(self):
        # find all iterations with errors in detection
        errors = []
        for iteration in range(len(self.detection_iter_log)):
            if self.detection_iter_log[iteration]['fingerprint_bit'] != self.fingerprint[self.detection_iter_log[iteration]['fingerprint_idx']]:
                errors.append(iteration)
        return errors

    def total_errors(self):
        return len(self.get_error_iterations())

    def error_rate(self):
        return self.total_errors()/self.embeddings_count

    def show_wrong_detections(self, batch=1):
        # find all iterations with errors in detection
        errors = self.get_error_iterations()
        if batch=='all' or batch>len(errors):
            batch = len(errors)

        iter_count = 0
        for iteration in errors:
            iter_count += 1
            if iter_count > batch:  # exit criterium
                break

            print("\nDetecting from attribute: " + str(self.detection_iter_log[iteration]['attribute']))
            fingerprinted_value = self.fingerprinted_data.iloc[self.detection_iter_log[iteration]['row_index']][
                self.detection_iter_log[iteration]['attribute']]
            print('Fingerprinted value: ' + str(fingerprinted_value))
            target_values_det = self.fingerprinted_data.iloc[self.detection_iter_log[iteration]['neighbors']][
                self.detection_iter_log[iteration]['attribute']].tolist()
            message = ' (i.e. tails of distribution)' if self.detection_iter_log[iteration][
                                                             'mark_bit'] == 0 else ' (i.e. in densest area)'
            print("Mark bit (where in distribution falls the target value?): " + str(
                self.detection_iter_log[iteration]['mark_bit']) + message)

            x_det, pdf_values_det, masked_pdf_det = self.values_for_plot_distribution(target_values_det)
            target_values_ins = list(self.relation.iloc[self.insertion_iter_log[iteration]['neighbors']][self.insertion_iter_log[iteration]['attribute']])
            x_ins, pdf_values_ins, masked_pdf_ins = self.values_for_plot_distribution(target_values_ins)

            fig, axs = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(10, 4))

            axs[0].plot(x_det, pdf_values_det, label='Original PDF\n (estimate)')
            axs[0].plot(x_det, masked_pdf_det, label='Modified PDF\n ({}th percentile)'.format(int(100 * 0.75)))
            axs[0].hist(target_values_det, bins=10, density=True, alpha=0.3, label='Neighbourhood\n data points')
            axs[0].scatter(fingerprinted_value, 0, color='red', label='Marked value', zorder=5)
            axs[0].set_ylabel('Density')
            axs[0].set_xlabel('Values')
            axs[0].set_title('Fingerprinted data')
            axs[0].legend(prop={'size': 8})

            axs[1].plot(x_ins, pdf_values_ins, label='Original PDF\n (estimate)')
            axs[1].plot(x_ins, masked_pdf_ins, label='Modified PDF\n ({}th percentile)'.format(int(100 * 0.75)))
            axs[1].hist(target_values_ins, bins=10, density=True, alpha=0.3, label='Neighbourhood\n data points')
            axs[1].scatter(self.insertion_iter_log[iteration]['new_value'], 0, color='red', label='Marked value', zorder=5)
            axs[1].set_ylabel('Density')
            axs[1].set_xlabel('Values {}'.format(self.detection_iter_log[iteration]['attribute']))
            axs[1].set_title('Original data')
            axs[1].legend(prop={'size': 8})


def demo(scheme):
    '''

    Args:
        scheme: the scheme instance, class NCorrFP

    Returns:

    '''
    demo_scheme = Demo(scheme=scheme)
    return demo_scheme

