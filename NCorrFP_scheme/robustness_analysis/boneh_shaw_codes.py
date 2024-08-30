import numpy as np
from scipy.stats import mode
import itertools
import random
from datetime import datetime


def generate_boneh_shaw_code(n_users, code_length, seed):
    """
    Generates Boneh-Shaw fingerprinting codes for a specified number of users and code length.

    Args:
    n_users (int): Number of users to generate codes for.
    code_length (int): Length of the fingerprint code.

    Returns:
    np.ndarray: A matrix of Boneh-Shaw codes of shape (n_users, code_length).
    """
    np.random.seed(seed)  # seed for reproducibility
    # Initialize the fingerprint code matrix
    codes = np.zeros((n_users, code_length), dtype=int)

    # Generate the fingerprint codes
    for i in range(n_users):
        for j in range(code_length):
            # Each position in the code is set to 1 with a certain probability
            codes[i, j] = 1 if np.random.rand() < 0.5 else 0

    return codes


def detect_colluders(codes, marked_code, k=1):
    """
    Detects colluders based on the marked code and threshold.

    Args:
    codes (np.ndarray): Matrix of Boneh-Shaw codes of shape (n_users, code_length).
    marked_code (np.ndarray): The marked code (suspicious code).
    threshold (float): Detection threshold.

    Returns:
    list: List of suspected colluders.
    """

    n_users, code_length = codes.shape
    scores = np.zeros(n_users)

    # Calculate the score for each user
    for user in range(n_users):
        scores[user] = np.sum(codes[user] == marked_code)

    # Calculate dynamic threshold based on mean and standard deviation
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    threshold = mean_score + k * std_score
    print("Dynamic threshold: ", threshold)

    # Identify colluders
    print(scores)
    colluders = [user for user in range(n_users) if scores[user] > threshold]

    return colluders


def example():
    # Example usage
    # Parameters that work together (10, 32, 20), (10, 64, 40)
    n_users = 10
    code_length = 64
    threshold = 40

    # Generate Boneh-Shaw codes
    codes = generate_boneh_shaw_code(n_users, code_length, seed=0)

    # Assume we have a marked code that we suspect is the result of collusion
    # For the sake of the example, we just use one of the generated codes
    # marked_code = codes[0]  # In practice, this would be the suspicious code to analyze

    marked_code = np.where(codes[0] == codes[1], codes[0], 1)  # Colluding two

    # Colluding three
    marked_code = np.zeros(codes[0].shape, dtype=codes[0].dtype)

    for i in range(len(codes[0])):
        values = [codes[0][i], codes[2][i], codes[5][i]]
        majority_value = mode(values)[0]
        marked_code[i] = majority_value

    # Detect colluders
    suspected_colluders = detect_colluders(codes, marked_code)#, threshold)

    print("Generated Boneh-Shaw Codes:\n", codes)
    print("Marked Code:\n", marked_code)
    print("Suspected Colluders:\n", suspected_colluders)


def experiment(seed, outfile):
    random.seed(seed)  # seed for reproducibility

    n_users = [10, 100, 1000]
    code_lengths = [16, 32, 64, 128, 256, 512, 1024, 2048]
    collusion_sizes = [2, 3, 4, 5, 10, 15, 20, 50, 100]  # condition: collusion_size < n_users/2
    confidences = [0.3, 0.5, 1, 1.5]  # confidence

    # iterate all parameter combinations
    for (n_user, code_length, collusion_size, k) in itertools.product(n_users, code_lengths, collusion_sizes, confidences):
        if collusion_size <= n_user:  # condition for a realistic collusion size
            # Generate Boneh-Shaw codes
            codes = generate_boneh_shaw_code(n_user, code_length, seed)

            # Collusion
            collusion = random.sample(range(n_user), collusion_size)  # randomly samples colluding partners
            marked_code = np.zeros(codes[0].shape, dtype=codes[0].dtype)
            for i in range(len(codes[0])):
                values = [codes[c][i] for c in collusion]
                majority_value = mode(values)[0]
                marked_code[i] = majority_value

            # Detect colluders
            print("Parameters: ",  (n_user, code_length, collusion_size, k))
            suspected_colluders = detect_colluders(codes, marked_code, k)  # , threshold)
            print("Suspected Colluders:\n", suspected_colluders)
            print("Real Colluders:\n", collusion)

            # Measurements
            detected_tot = len(suspected_colluders)
            detected_correct = sum(1 for elem in suspected_colluders if elem in collusion)  # true positive
            detected_incorrect = sum(1 for elem in suspected_colluders if elem not in collusion)  # false positive
            undetected = sum(1 for elem in collusion if elem not in suspected_colluders)  # false negative
            print("Measurements: ", (detected_tot, detected_correct, detected_incorrect, undetected), "\n")

            # Record results
            with open(outfile, 'a') as f:
                f.write(','.join(map(str, [seed, n_user, code_length, collusion_size, k,
                                           detected_tot, detected_correct, detected_incorrect, undetected])) + '\n')


#timestamp = str(datetime.fromtimestamp(int(datetime.timestamp(datetime.now())))).replace(' ', '-').replace(':','-')
#outfile = "collusion/bosh_{}.csv".format(timestamp)
#with open(outfile, "a+") as file:
#    file.write("exp_no,n_users,code_length,collusion_size,confidence,"
#               "detected_tot,detected_correct,detected_incorrect,undetected")

#for i in range(10):
#    experiment(seed=i, outfile=outfile)
example()
