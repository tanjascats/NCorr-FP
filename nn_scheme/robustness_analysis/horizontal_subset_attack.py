# this is a script that runs the experiments of subset attack on
# the scheme for categorical data based on a neighbourhood search
# approach
import json
import random
import sys
from datetime import datetime
from matplotlib import pyplot as plt
import numpy as np

from attacks.horizontal_subset_attack import HorizontalSubsetAttack
from nn_scheme.experimental.blind_scheme import BlindNNScheme

# results in results/horizontal


def run():
    config_file = "config/horizontal.json"
    with open(config_file) as infile:
        config = json.load(infile)
    #n_experiments = 1 #10  # number of times we attack the same fingerprinted file
    #n_fp_experiments = 1 #25  # number of times we run fp insertion

#    size_of_subset = np.array([0.95, 0.9 , 0.85, 0.8 , 0.75, 0.7 , 0.65, 0.6 , 0.55, 0.5 ,
#       0.45, 0.4 , 0.35, 0.3 , 0.25, 0.2 , 0.15, 0.1 , 0.05]) ## if bad robustness is expected, use this to optimise the runs
    size_of_subset = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.40, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75,
              0.8, 0.85, 0.9, 0.95, 1.0]) ## if good robustness is expected
    results = []
    #gamma = 1; xi = 2; fingerprint_bit_length = 16

    scheme = BlindNNScheme(gamma=config['gamma'],
                           xi=config['xi'],
                           fingerprint_bit_length=config['fingerprint_bit_length'])
    attack = HorizontalSubsetAttack()
    #data = "breast_cancer"

    timestamp = str(datetime.fromtimestamp(int(datetime.timestamp(datetime.now())))).replace(' ', '-').replace(':','-')
    f = open("log/"
             "horizontal_subset_attack_{}_{}.txt".format(config['data'], timestamp), "a+")

    done = False
    for size in size_of_subset:
        if not done:
            # for reproducibility
            seed = 332
            random.seed(seed)

            correct, misdiagnosis = 0, 0
            for i in range(config['n_fp_experiments']):
                # fingerprint the data
                secret_key = random.randint(0, 1000)
                fp_dataset = scheme.insertion(dataset_name=config['data'],
                                              recipient_id=1, secret_key=secret_key,
                                              correlated_attributes=config['correlated_attributes'])

                for j in range(config['n_experiments']):
                    # perform the attack
                    release_data = attack.run(dataset=fp_dataset, fraction=size, random_state=secret_key+j)
                    # try to extract the fingerprint
                    suspect = scheme.detection(release_data,
                                               primary_key='Id',
                                               secret_key=secret_key,
                                               correlated_attributes=config['correlated_attributes'])
                    if suspect == 1:
                        correct += 1
                    elif suspect != -1:
                            misdiagnosis += 1

            print("\n\n--------------------------------------------------------------\n\n")
            print("Data: " + config['data'])
            print("(size of subset, gamma, xi, length of a fingerprint): " + str((size, config['gamma'], config['xi'], config['fingerprint_bit_length'])))
            print("Correct: " + str(correct) + "/" + str(config['n_experiments']*config['n_fp_experiments']))
            print("Wrong: " + str(config['n_experiments']*config['n_fp_experiments'] - correct) + "/" + str(config['n_experiments']*config['n_fp_experiments'])
                  + "\n\t- out of which misdiagnosed: " + str(misdiagnosis))

            # write to log file
            f.write(str(datetime.fromtimestamp(int(datetime.timestamp(datetime.now())))))
            f.write("\nseed: " + str(seed))
            f.write("\nData: " + config['data'])
            f.write("\n(size of subset, gamma, xi, length of a fingerprint): " + str((size, config['gamma'], config['xi'], config['fingerprint_bit_length'])))
            f.write("\nCorrect: " + str(correct) + "/" + str(config['n_experiments']*config['n_fp_experiments']))
            f.write("\nWrong: " + str(config['n_experiments']*config['n_fp_experiments'] - correct) + "/" + str(config['n_experiments']*config['n_fp_experiments'])
                  + "\n\t- out of which misdiagnosed: " + str(misdiagnosis))
            f.write("\n\n--------------------------------------------------------------\n\n")

            results.append(correct)

            # if correct == 0:  # for expected bad robustness
            if correct == int(config['n_fp_experiments']*config['n_experiments']):  # for expected good robustness
                done = True
        else:
            # skipping unnecessary calculations since the false miss has already reached the max/min
            # results.append(0)  # for expected bad robustness
            results.append(int(config['n_fp_experiments']*config['n_experiments']))  # for expected good robustness

    f.write("SUMMARY\n")
    f.write(str(datetime.fromtimestamp(int(datetime.timestamp(datetime.now())))))
    f.write("\n(gamma, xi, length of a fingerprint): " + str((config['gamma'], config['xi'], config['fingerprint_bit_length'])))
    f.write("\nCorrect: " + str(results) + "\n\t/" + str(str(config['n_experiments']*config['n_fp_experiments'])))
    f.write("\n\n--------------------------------------------------------------\n\n")
    f.close()


if __name__ == '__main__':
    run()
