import numpy as np
from scipy.stats import mode


def calculate_code_length(n_users, epsilon):
    """
    Calculate the length of the Tardos code based on the number of users and epsilon.

    Args:
    n_users (int): Number of users to generate codes for.
    epsilon (float): Parameter to control the error probability.

    Returns:
    int: Calculated code length.
    """
    return int(np.ceil((100 * np.log(n_users)) / (epsilon ** 2)))


def generate_tardos_code(recipient_id, fp_len, secret_key):
    """

   Args:
       recipient_id:
       fp_len:
       secret_key:

   Returns:

   """
    # print("Code length: ", fp_len)
    # Initialize the probability vector
    # - random vector from a beta distribution (with alpha=beta=0.5 it's a U-shaped distribution [0,1])
    # seed ensures that the probability vector stays the same on every fingerprint creation
    np.random.seed(secret_key)
    p = np.random.beta(0.5, 0.5, size=fp_len)
    # Generate one Tardos code
    np.random.seed(recipient_id)  # we need to do this, otherwise every recipient gets the same code
    code = (np.random.rand(fp_len) < p).astype(int)
    # todo: consider appending the new code to a codebook if necessary -- for now there is no assignment code-recipient

    return code


def generate_tardos_codebook(n_users, fp_len, secret_key):
    """
    Generates Tardos codes for a specified number of users and code length -- the codebook.

    Args:
    n_users (int): Number of users to generate codes for.
    epsilon (float): Parameter to control the error probability.

    Returns:
    np.ndarray: A matrix of Tardos codes of shape (n_users, code_length).
    np.ndarray: The probability vector used to generate the codes.
    """
    code_length = fp_len
    print("Code length: ", code_length)
    # Initialize the probability vector
    # - random vector from a beta distribution (with alpha=beta=0.5 it's a U-shaped distribution [0,1])
    np.random.seed(secret_key)
    p = np.random.beta(0.5, 0.5, size=code_length)
    # Generate the Tardos codes
    codes = np.zeros((n_users, code_length), dtype=int)
    for user in range(n_users):
        codes[user] = (np.random.rand(code_length) < p).astype(int)
    return codes, p


def score_users(code, n_users, secret_key):  # threshold
    code_length = len(code)
    codebook, p = generate_tardos_codebook(n_users, code_length, secret_key)
    scores = np.zeros(n_users)

    # If a user’s code matches the suspicious code (marked code) at positions where the probability of being 1 (p[pos])
    # is high, the user gets a higher score. Similarly, if their code deviates in certain positions, they are penalized.
    for user in range(n_users):
        for pos in range(code_length):
            if code[pos] == 1:
                score_update = np.log(1 / p[pos]) if codebook[user, pos] == 1 else np.log(1 / (1 - p[pos]))
                scores[user] += score_update
            else:
                score_update = np.log(1 / (1 - p[pos])) if codebook[user, pos] == 1 else np.log(1 / p[pos])
                scores[user] += score_update
    print("Scores: ", scores)
    return scores


def detect_colluders(code, secret_key, k=1.0):
    scores = score_users(code, marked_code, secret_key)

    # Calculate dynamic threshold based on mean and standard deviation
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    threshold = mean_score + k * std_score
    print("Dynamic threshold: ", threshold)

    # Identify colluders
    colluders = [user for user in range(n_users) if scores[user] > threshold]
    return colluders


def __generate_tardos_code(n_users, epsilon):
    """
    Generates Tardos codes for a specified number of users and code length based on epsilon.

    Args:
    n_users (int): Number of users to generate codes for.
    epsilon (float): Parameter to control the error probability.

    Returns:
    np.ndarray: A matrix of Tardos codes of shape (n_users, code_length).
    np.ndarray: The probability vector used to generate the codes.
    """
    code_length = calculate_code_length(n_users, epsilon)
    print("Code length: ", code_length)
    # Initialize the probability vector
    p = np.random.beta(0.5, 0.5, size=code_length)

    # Generate the Tardos codes
    codes = np.zeros((n_users, code_length), dtype=int)
    for user in range(n_users):
        codes[user] = (np.random.rand(code_length) < p).astype(int)

    return codes, p


def __score_users_old(codes, marked_code, p):  # threshold
    """
    Detects colluders based on the marked code and threshold.

    Args:
    codes (np.ndarray): Matrix of Tardos codes of shape (n_users, code_length).
    marked_code (np.ndarray): The marked code (suspicious code).
    p (np.ndarray): Probability vector used to generate the codes.
    k (float): Number of standard deviations for calculating the dynamic threshold. The less, the less confident.

    Returns:
        np.ndarray:
    """
    n_users, code_length = codes.shape
    scores = np.zeros(n_users)

    # If a user’s code matches the suspicious code (marked code) at positions where the probability of being 1 (p[pos])
    # is high, the user gets a higher score. Similarly, if their code deviates in certain positions, they are penalized.
    for user in range(n_users):
        for pos in range(code_length):
            if marked_code[pos] == 1:
                score_update = np.log(1 / p[pos]) if codes[user, pos] == 1 else np.log(1 / (1 - p[pos]))
                scores[user] += score_update
            else:
                score_update = np.log(1 / (1 - p[pos])) if codes[user, pos] == 1 else np.log(1 / p[pos])
                scores[user] += score_update
    print("Scores: ", scores)
    return scores


def _detect_colluders_old(codes, marked_code, p, k=1.0):
    scores = __score_users_old(codes, marked_code, p)

    # Calculate dynamic threshold based on mean and standard deviation
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    threshold = mean_score + k * std_score
    print("Dynamic threshold: ", threshold)

    # Identify colluders
    colluders = [user for user in range(n_users) if scores[user] > threshold]
    return colluders


if __name__ == '__main__':
    n_users = 5
    epsilon = 0.1  # error probability, i.e., how likely it is for innocent users to be falsely accused or for guilty users
    # to go undetected. Smaller eps makes the code more secure but increases the length of the code.
    fp_len = 64
    # threshold = 3100000

    # Generate Tardos codes
    # codes, p = generate_tardos_code(n_users, epsilon)
    codes, p = generate_tardos_code(n_users, fp_len, 101)

    # Assume we have a marked code that we suspect is the result of collusion
    # marked_code = codes[0]  # In practice, this would be the suspicious code
    # Colluding two
    marked_code = np.where(codes[2] == codes[4], codes[2], 1)
    print("Marked code: ", marked_code)
    # Colluding three
    # marked_code = np.zeros(codes[0].shape, dtype=codes[0].dtype)
    # for i in range(len(codes[0])):
    #    values = [codes[0][i], codes[2][i], codes[4][i]]
    #    majority_value = mode(values)[0]
    #    marked_code[i] = majority_value

    # Detect colluders
    suspected_colluders = _detect_colluders_old(codes, marked_code, p, k=0.8)

    print("Generated Tardos Codes:\n", codes)
    print("Probability Vector:\n", p)
    print("Suspected Colluders:\n", suspected_colluders)
