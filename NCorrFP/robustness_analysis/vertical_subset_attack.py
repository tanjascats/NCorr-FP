# this is a script that runs the experiments of a vertical subset attack on
# the scheme for categorical data based on a neighbourhood search
# approach
import random
from datetime import datetime
import numpy as np
import json
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import attacks.vertical_subset_attack
from NCorrFP.NCorrFP import NCorrFP


def run(config_file=None):
    #path = "NCorrFP/robustness_analysis/"  # for runs from terminal -- add to config file
    if config_file is None:
        config_file = "config/vertical.json"
    with open(config_file) as infile:
        config = json.load(infile)
    # n_experiments = 10  # (20) number of times we attack the same fingerprinted file
    # n_fp_experiments = 30  # (50) number of times we run fp insertion

    size_of_subset = np.array(config['size_of_subset'])  # number of columns to be DELETED
    # size_of_subset = np.array([i for i in range(20)])  # number of columns to be DELETED
    results = []
    # gamma = 3; xi = 2; fingerprint_bit_length = 8

    scheme = NCorrFP(gamma=config['gamma'],
                     xi=config['xi'],
                     fingerprint_bit_length=config['fingerprint_bit_length'])
    attack = attacks.vertical_subset_attack.VerticalSubsetAttack()
    # data = 'german_credit'

    timestamp = str(datetime.fromtimestamp(int(datetime.timestamp(datetime.now())))).replace(' ', '-').replace(':', '-')
    f = open("log/"
             "vertical_subset_attack_{}_{}.txt".format(config['data'], timestamp), "a+")

    df = dict()
    done = False
    for size in size_of_subset:
        if not done:
            # for reproducibility
            seed = config['seed']
            random.seed(seed)

            correct, misdiagnosis = 0, 0
            detected_fingerprints = []
            counts = []
            for i in range(config['n_fp_experiments']):
                # fingerprint the data
                # random.seed(seed + i)
                secret_key = random.randint(0, 1000)
                fp_dataset = scheme.insertion(dataset_name=config['data'],
                                              recipient_id=1, secret_key=secret_key,
                                              correlated_attributes=config['correlated_attributes'])

                for j in range(config['n_experiments']):
                    # perform the attack
                    release_data = attack.run_random(dataset=fp_dataset, number_of_columns=size,
                                                     random_state=secret_key + j)
                    # try to extract the fingerprint
                    suspect = scheme.detection(release_data,
                                               primary_key='Id',
                                               secret_key=secret_key,
                                               correlated_attributes=config['correlated_attributes'],
                                               original_columns=config['original_columns'])
                    if suspect == 1:
                        correct += 1
                    elif suspect != -1:
                        misdiagnosis += 1

                    detected_fingerprints.append(scheme.detected_fp)
                    counts.append(scheme.count)

            if str(config['gamma']) not in df.keys():
                df[str(config['gamma'])] = dict()
            df[str(config['gamma'])][str(size)] = detected_fingerprints
            print("\n\n--------------------------------------------------------------\n\n")
            print("Data: " + config['data'])
            print("(size of subset, gamma, xi, length of a fingerprint): " + str(
                (size, config['gamma'], config['xi'], config['fingerprint_bit_length'])))
            print("Correct: " + str(correct) + "/" + str(config['n_experiments'] * config['n_fp_experiments']))
            print("Wrong: " + str(config['n_experiments'] * config['n_fp_experiments'] - correct) + "/" + str(
                config['n_experiments'] * config['n_fp_experiments'])
                  + "\n\t- out of which misdiagnosed: " + str(misdiagnosis))

            # write to log file
            f.write(str(datetime.fromtimestamp(int(datetime.timestamp(datetime.now())))))
            f.write("\nseed: " + str(seed))
            f.write("\nData: " + config['data'])
            f.write("\n(size of subset, gamma, xi, length of a fingerprint): " + str(
                (size, config['gamma'], config['xi'], config['fingerprint_bit_length'])))
            f.write("\nCorrect: " + str(correct) + "/" + str(config['n_experiments'] * config['n_fp_experiments']))
            f.write("\nWrong: " + str(config['n_experiments'] * config['n_fp_experiments'] - correct) + "/" + str(
                config['n_experiments'] * config['n_fp_experiments'])
                    + "\n\t- out of which misdiagnosed: " + str(misdiagnosis))
            f.write("\nDetected fingerprints: " + str(detected_fingerprints))
            f.write("\nVotes: " + str(counts))
            f.write("\n\n--------------------------------------------------------------\n\n")

            results.append(correct)

            # if correct == 0:  # for expected bad robustness
            if correct == int(config['n_fp_experiments'] * config['n_experiments']):  # for expected good robustness
                done = True
        else:
            # skipping unnecessary calculations since the false miss has already reached the max/min
            # results.append(0)  # for expected bad robustness
            results.append(int(config['n_fp_experiments'] * config['n_experiments']))  # for expected good robustness

    f.write("SUMMARY\n")
    f.write(str(datetime.fromtimestamp(int(datetime.timestamp(datetime.now())))))
    f.write("\n(gamma, xi, length of a fingerprint): " + str(
        (config['gamma'], config['xi'], config['fingerprint_bit_length'])))
    f.write("\nCorrect: " + str(results) + "\n\t/" + str(str(config['n_experiments'] * config['n_fp_experiments'])))

    f.write("\n\n--------------------------------------------------------------\n\n")
    f.close()

    with open('detection_error_analysis/'
              'vertical_attack/'
              'detected_fingerprints_test.json', 'w+') as outfile:
        json.dump(df, outfile)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        config_file_path = sys.argv[1]
        run(config_file_path)
    else:
        run()
