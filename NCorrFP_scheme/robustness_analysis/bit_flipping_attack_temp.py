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
from NCorrFP_scheme.NCorrFP import NCorrFP


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
            print(size)
            seed = 332
            random.seed(seed)

            correct, misdiagnosis = 0, 0
            detected_fingerprints = []
            counts = []
            for i in range(config['n_fp_experiments']):
                # fingerprint the data
                secret_key = random.randint(0, 1000)
                fp_dataset = scheme.insertion(dataset_name=config['data'],
                                              recipient_id=1, secret_key=secret_key,
                                              correlated_attributes=config['correlated_attributes'])
                for j in range(config['n_experiments']):
                    # perform the attack
                    release_data = attack.run_temp(dataset=fp_dataset, fraction=size)
                    # try to extract the fingerprint
                    suspect = scheme.detection_temp(release_data,
                                               primary_key='Id',
                                               secret_key=secret_key,
                                               correlated_attributes=config['correlated_attributes'])
                    if suspect == 1:
                        correct += 1
                    elif suspect != -1:
                        misdiagnosis += 1
                    detected_fingerprints.append(scheme.detected_fp)
                    counts.append(scheme.count)




if __name__ == '__main__':
    if len(sys.argv) > 1:
        config_file_path = sys.argv[1]
        run(config_file_path)
    else:
        run()
