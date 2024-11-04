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
    n_users (int): Number of users to generate codes for. Extract from the fingerprinting scheme which should keep the
    record.
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
        np.random.seed(user)  # to get the same fingerprint as we would do by generating only one specific
        codes[user] = (np.random.rand(code_length) < p).astype(int)
    return codes, p


def score_users(suspicious_code, n_users, secret_key):  # threshold
    """
    Calculate scores for each user that estimates how likely have they participated in the collusion that created the
    suspicious code. Larger scores indicate more confidence that the user is a participant of the collusion.
    Args:
        suspicious_code: suspicious fingerprint
        n_users: total number of generated fingerprints
        secret_key: owner's secret

    Returns:

    """
    code_length = len(suspicious_code)
    codebook, p = generate_tardos_codebook(n_users, code_length, secret_key)
    scores = np.zeros(n_users)

    # If a user’s code matches the suspicious code (marked code) at positions where the probability of being 1 (p[pos])
    # is high, the user gets a higher score. Similarly, if their code deviates in certain positions, they are penalized.
    for user in range(n_users):
        for pos in range(code_length):
            if suspicious_code[pos] == 1:
                score_update = np.log(1 / p[pos]) if codebook[user, pos] == 1 else np.log(1 / (1 - p[pos]))
                scores[user] += score_update
            else:  # if the bit is undecided (2), it assumes 0. Maybe there is room for improvement
                score_update = np.log(1 / (1 - p[pos])) if codebook[user, pos] == 1 else np.log(1 / p[pos])
                scores[user] += score_update
    print("Scores: ", scores)
    return scores


def check_exact_matching(code, secret_key, n_users):
    """
    Checks exact matching of a provided code to any code from the codebook and returns the matching recipient's ID.
    Args:
        code: code to check against the codebook
        secret_key: owner's secret
        n_users: total number of created fingerprints (dataset recipients)

    Returns: (int) recipient ID or -1 if no matching is found

    """
    code_length = len(code)

    # generate tardos codebook to compare against
    codebook, p = generate_tardos_codebook(n_users, code_length, secret_key)
    print(codebook)

    for user in range(n_users):
        if np.array_equal(code, codebook[user]):
            return user

    return -1


def decode_fingerprint(fingerprint, secret_key, total_n_recipients):
    """
    Decodes a detected tardos fingerprint.
    Args:
        fingerprint: bit string
        secret_key: owner's secret key

    Returns:

    """
    # First, check exact matching.
    suspect = check_exact_matching(fingerprint, secret_key, total_n_recipients)
    if suspect == -1:  # if there is no direct matching in the codebook, apply probabilistic collusion matching
        detect_colluders(fingerprint, secret_key, total_n_recipients)


def detect_colluders(code, secret_key, total_n_recipients, k=1.0):
    """
    Detect colluders from a compromised fingerprint based on probabilistic Tardos codes.
    Args:
        code: (list) compromised fingerprint
        secret_key: owner's secret
        total_n_recipients: (int) total number of created fingerprints
        k: collusion confidence -- the larger, the more likely that the detected colluders are the true colluders; if
        too small, can lead to false positives; if too big, it can lead to false negatives

    Returns: a list of detected colluding recipients

    """
    scores = score_users(code, total_n_recipients, secret_key)

    # Calculate dynamic threshold based on mean and standard deviation
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    threshold = mean_score + k * std_score
    print("Dynamic threshold: ", threshold)

    # Identify colluders
    colluders = [user for user in range(total_n_recipients) if scores[user] > threshold]
    return colluders


def demo():
    n_users = 5
    epsilon = 0.1  # error probability, i.e. how likely will the innocent users to be falsely accused or the guilty
    # users go undetected. Smaller eps makes the code more secure but increases the length of the code.
    fp_len = 32
    # threshold = 3100000

    # Generate Tardos codes
    # codes, p = generate_tardos_code(n_users, epsilon)
    codes, p = generate_tardos_codebook(n_users, fp_len, 101)  # generates entire codebook

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

    # Detect a clean fingerprint
    suspected_recipient = check_exact_matching(secret_key=101, code=codes[1], n_users=n_users)
    # Detect colluders
    suspected_colluders = detect_colluders(code=marked_code, total_n_recipients=n_users, secret_key=101, k=0.8)

    print("Generated Tardos Codes:\n", codes)
    print("Probability Vector:\n", p)
    print("Detection of a clean fingerprint:\n", suspected_recipient)
    print("Suspected Colluders:\n", suspected_colluders)


if __name__ == '__main__':
    demo()
