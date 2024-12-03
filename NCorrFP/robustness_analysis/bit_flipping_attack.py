# this is a script that runs the experiments of a bit-flipping attack on
# the scheme for categorical data based on a neighbourhood search
# approach
import random
from datetime import datetime
import numpy as np
import json
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from attacks.bit_flipping_attack import BitFlippingAttack
from NCorrFP.NCorrFP import NCorrFP


def run(config_file=None):
    results = []
    if config_file is None:
        config_file = "config/flipping.json"
    with open(config_file) as infile:
        config = json.load(infile)

    scheme = NCorrFP(gamma=config['gamma'],
                     xi=config['xi'],
                     fingerprint_bit_length=config['fingerprint_bit_length'])
    attack = BitFlippingAttack()

    fractions = np.array(config['size_of_subset'])
    timestamp = str(datetime.fromtimestamp(int(datetime.timestamp(datetime.now())))).replace(' ', '-').replace(':', '-')
    df = dict()
    done = False
    for size in fractions:
        if not done:
            # for reproducibility
            seed = config['seed']
            random.seed(seed)

            correct, misdiagnosis = 0, 0
            detected_fingerprints = []
            counts = []
            for i in range(config['n_fp_experiments']):
                # fingerprint the data
                random.seed(seed + i)
                secret_key = random.randint(0, 1000)
                fp_dataset = scheme.insertion(dataset_name=config['data'],
                                              recipient_id=1, secret_key=secret_key,
                                              correlated_attributes=config['correlated_attributes'])
                for j in range(config['n_experiments']):
                    # perform the attack
                    release_data = attack.run(dataset=fp_dataset, fraction=size)
                    # try to extract the fingerprint
                    suspect = scheme.detection(release_data,
                                               primary_key='Id',
                                               secret_key=secret_key,
                                               correlated_attributes=config['correlated_attributes'])
                    if suspect == 1:
                        correct += 1
                    elif suspect != -1:
                        misdiagnosis += 1
                    detected_fingerprints.append(scheme.detected_fp)
                    counts.append(scheme.count)
            if str(config['gamma']) not in df.keys():
                df[str(config['gamma'])] = dict()
            df[str(config['gamma'])][str(size)] = detected_fingerprints
            # write to log file
            f = open("log/"
                     "/bit_flipping_attack_{}_{}.txt".format(config['data'], timestamp), "a+")
            f.write(str(datetime.fromtimestamp(int(datetime.timestamp(datetime.now())))))
            f.write("\nseed: " + str(seed))
            f.write("\nData: " + config['data'])
            f.write("\n(fraction flipped, gamma, xi, length of a fingerprint): " + str(
                    (size, config['gamma'], config['xi'], config['fingerprint_bit_length'])))
            f.write("\nCorrect: " + str(correct) + "/" + str(config['n_experiments'] * config['n_fp_experiments']))
            f.write("\nWrong: " + str(config['n_experiments'] * config['n_fp_experiments'] - correct) + "/" + str(
                config['n_experiments'] * config['n_fp_experiments'])
                    + "\n\t- out of which misdiagnosed: " + str(misdiagnosis))
            f.write("\nDetected fingerprints: " + str(detected_fingerprints))
            f.write("\nVotes: " + str(counts))
            f.write("\n\n--------------------------------------------------------------\n\n")

            results.append(correct)

            f.write("intermediate summary\n")
            f.write("\n(gamma, xi, length of a fingerprint): " + str(
                (config['gamma'], config['xi'], config['fingerprint_bit_length'])))
            f.write(
                "\nCorrect: " + str(results) + "\n\t/" + str(str(config['n_experiments'] * config['n_fp_experiments'])))
            f.close()

            # if correct == 0:  # for expected bad robustness (FRACTIONS NEED TO BE IN ASCENDING ORDER (config file))
            if correct == int(config['n_fp_experiments'] * config['n_experiments']):  # for expected good robustness (FRACTIONS NEED TO BE IN DESCENDING ORDER (config file))
                done = True
        else:
            # skipping unnecessary calculations since the false miss has already reached the max/min
            # results.append(0)  # for expected bad robustness
            results.append(int(config['n_fp_experiments']*config['n_experiments']))  # for expected good robustness

    f = open("log"
             "/bit_flipping_attack_{}_{}.txt".format(config['data'], timestamp), "a+")
    f.write("SUMMARY\n")
    f.write(str(datetime.fromtimestamp(int(datetime.timestamp(datetime.now())))))
    f.write("\n(gamma, xi, length of a fingerprint): " + str(
        (config['gamma'], config['xi'], config['fingerprint_bit_length'])))
    f.write("\nCorrect: " + str(results) + "\n\t/" + str(str(config['n_experiments'] * config['n_fp_experiments'])))
    f.write("\n\n--------------------------------------------------------------\n\n")
    f.close()

    print("SUMMARY\n")
    print(str(datetime.fromtimestamp(int(datetime.timestamp(datetime.now())))))
    print("\n(gamma, xi, length of a fingerprint): " + str(
        (config['gamma'], config['xi'], config['fingerprint_bit_length'])))
    print("\nCorrect: " + str(results) + "\n\t/" + str(str(config['n_experiments'] * config['n_fp_experiments'])))

    with open('detection_error_analysis/'
              'flipping_attack/'
              'detected_fingerprints_test.json', 'w+') as outfile:
        json.dump(df, outfile)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        config_file_path = sys.argv[1]
        run(config_file_path)
    else:
        run()
