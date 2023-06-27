# this is a script that runs the experiments of a vertical subset attack on
# the scheme for categorical data based on a neighbourhood search
# approach
import random
from datetime import datetime
import numpy as np

from attacks.vertical_subset_attack import VerticalSubsetAttack
from scheme import CategoricalNeighbourhood

n_experiments = 10  # (20) number of times we attack the same fingerprinted file
n_fp_experiments = 30  # (50) number of times we run fp insertion

size_of_subset = np.array([i for i in range(20)])  # number of columns to be DELETED
results = []
gamma = 3; xi = 2; fingerprint_bit_length = 8

scheme = CategoricalNeighbourhood(gamma=gamma, xi=xi, fingerprint_bit_length=fingerprint_bit_length)
attack = VerticalSubsetAttack()
data = 'german_credit'

f = open("robustness_analysis/categorical_neighbourhood/log/vertical_subset_attack_" + data + ".txt", "a+")

for size in size_of_subset:
    # for reproducibility
    seed = 332
    random.seed(seed)

    correct, misdiagnosis = 0, 0
    for i in range(n_fp_experiments):
        # fingerprint the data
        secret_key = random.randint(0, 1000)
        fp_dataset = scheme.insertion(dataset_name=data, buyer_id=1, secret_key=secret_key)

        for j in range(n_experiments):
            # perform the attack
            release_data = attack.run_random(dataset=fp_dataset, number_of_columns=size)
            # try to extract the fingerprint
            suspect = scheme.detection(dataset_name=data, real_buyer_id=1, secret_key=secret_key,
                                dataset=release_data)
            if suspect == 1:
                correct += 1
            elif suspect != -1:
                    misdiagnosis += 1

    #print("\n\n--------------------------------------------------------------\n\n")
    #print("Data: german credit")
    #print("(size of subset, gamma, xi, length of a fingerprint): " + str((size, gamma, xi, fingerprint_bit_length)))
    #print("Correct: " + str(correct) + "/" + str(n_experiments*n_fp_experiments))
    #print("Wrong: " + str(n_experiments*n_fp_experiments - correct) + "/" + str(n_experiments*n_fp_experiments)
    #      + "\n\t- out of which misdiagnosed: " + str(misdiagnosis))

    # write to log file
    f.write(str(datetime.fromtimestamp(int(datetime.timestamp(datetime.now())))))
    f.write("\nseed: " + str(seed))
    f.write("\nData: " + data)
    f.write("\n(size of subset, gamma, xi, length of a fingerprint): " + str((size, gamma, xi,
                                                                            fingerprint_bit_length)))
    f.write("\nCorrect: " + str(correct) + "/" + str(n_experiments*n_fp_experiments))
    f.write("\nWrong: " + str(n_experiments*n_fp_experiments - correct) + "/" + str(n_experiments*n_fp_experiments)
          + "\n\t- out of which misdiagnosed: " + str(misdiagnosis))
    f.write("\n\n--------------------------------------------------------------\n\n")

    results.append(correct)

f.write("SUMMARY\n")
f.write(str(datetime.fromtimestamp(int(datetime.timestamp(datetime.now())))))
f.write("\n(gamma, xi, length of a fingerprint): " + str((gamma, xi, fingerprint_bit_length)))
f.write("\nCorrect: " + str(results) + "\n\t/" + str(n_experiments * n_fp_experiments))
f.write("\n\n--------------------------------------------------------------\n\n")
f.close()

print("SUMMARY\n")
print(str(datetime.fromtimestamp(int(datetime.timestamp(datetime.now())))))
print("\n(gamma, xi, length of a fingerprint): " + str((gamma, xi, fingerprint_bit_length)))
print("\nCorrect: " + str(results) + "\n\t/" + str(n_experiments * n_fp_experiments))

