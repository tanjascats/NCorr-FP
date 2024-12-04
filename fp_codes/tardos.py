import numpy as np


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


def generate(recipient_id, secret_key, fp_len=None, epsilon=0.1):
    """

   Args:
       recipient_id (int): id of the recipient
       fp_len (int): length of the fingerprint in bits (should be the same for all recipients)
       secret_key: owner's secret
       epsilon:

   Returns:

   """
    if fp_len is not None:
        code_length = fp_len
    else:
        exit("You must provide the Tardos code length.")
        # todo: code_length = calculate_code_length(n_users, epsilon)
    # print("Code length: ", fp_len)
    # Initialize the probability vector
    # - random vector from a beta distribution (with alpha=beta=0.5 it's a U-shaped distribution [0,1])
    # seed ensures that the probability vector stays the same on every fingerprint creation
    np.random.seed(secret_key)
    p = np.random.beta(0.5, 0.5, size=code_length)
    # Generate one Tardos code
    np.random.seed(recipient_id)  # we need to do this, otherwise every recipient gets the same code
    code = (np.random.rand(fp_len) < p).astype(int)
    # todo: consider appending the new code to a codebook if necessary -- for now there is no assignment code-recipient
    return code


def generate_codebook(n_users, secret_key, fp_len=None, epsilon=0.1):
    """
    Generates Tardos codes for a specified number of users and code length -- the codebook.

    Args:
    n_users (int): Number of users to generate codes for. Extract from the fingerprinting scheme which should keep the
    record.
    epsilon (float): Parameter to control the error probability. Can be used instead of fp_len

    Returns:
    np.ndarray: A matrix of Tardos codes of shape (n_users, code_length).
    np.ndarray: The probability vector used to generate the codes.
    """
    if fp_len is not None:
        code_length = fp_len
    else:
        exit("You must provide the Tardos code length.")
        # todo: code_length = calculate_code_length(n_users, epsilon)
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


def score_users(suspicious_code, secret_key, n_users):  # threshold
    """
    Calculate scores for each user that estimates how likely have they participated in the collusion that created the
    suspicious code. Larger scores indicate more confidence that the user is a participant of the collusion.
    Args:
        suspicious_code: suspicious fingerprint
        n_users: total number of generated fingerprints
        secret_key: owner's secret

    Returns:

    """
    scores = np.zeros(n_users)

    code_length = len(suspicious_code)
    # retreive probabilities
    np.random.seed(secret_key)
    p = np.random.beta(0.5, 0.5, size=code_length)

    # Generate the Tardos codes
    codebook = np.zeros((n_users, code_length), dtype=int)
    for user in range(n_users):
        np.random.seed(user)
        codebook[user] = (np.random.rand(code_length) < p).astype(int)

    # If a user’s code matches the suspicious code (marked code) at positions where the probability of being 1 (p[pos])
    # is high, the user gets a higher score. Similarly, if their code deviates in certain positions, they are penalized.
    for user in range(n_users):
        for pos in range(code_length):
            if suspicious_code[pos] == 1:
                scores[user] += np.log(1 / p[pos]) if codebook[user, pos] == 1 else np.log(1 / (1 - p[pos]))
            else:  # if the bit is undecided (2), it assumes 0. Maybe there is room for improvement
                scores[user] += np.log(1 / (1 - p[pos])) if codebook[user, pos] == 1 else np.log(1 / p[pos])
#    print("Scores: ", scores)
    scores = {i: scores[i] for i in range(len(scores))}
    scores = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))

    return scores


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
    scores_dict = score_users(code, secret_key, total_n_recipients)
    scores = list(scores_dict.values())

    # Calculate dynamic threshold based on mean and standard deviation
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    threshold = mean_score + k * std_score
    print("Dynamic threshold: ", threshold)

    # Identify colluders
    colluders = [user for user in range(total_n_recipients) if scores[user] > threshold]
    # collusion_scores = {i: scores[i] for i in range(len(scores))}
    return colluders


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
    codebook, p = generate_codebook(n_users=n_users, secret_key=secret_key, fp_len=code_length)
    print(codebook)

    for user in range(n_users):
        if np.array_equal(code, codebook[user]):
            return user

    return -1


def exact_matching_scores(code, secret_key, n_users):
    """
        Calculates exact matching scores of a provided code for every code in the codebook.
        Args:
            code: code to check against the codebook
            secret_key: owner's secret
            n_users: total number of created fingerprints (dataset recipients)

        Returns: (dict) recipient_id: matching_score

        """
    code_length = len(code)

    # generate tardos codebook to compare against
    codebook, p = generate_codebook(n_users=n_users, secret_key=secret_key, fp_len=code_length)

    confidence = dict()
    for user in range(n_users):
        confidence[user] = np.sum(code == codebook[user]) / len(code)

    return confidence


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


def demo():
    n_users = 20
    epsilon = 0.1  # error probability, i.e. how likely will the innocent users to be falsely accused or the guilty
    # users go undetected. Smaller eps makes the code more secure but increases the length of the code.
    fp_len = 1024
    secret_key = 101
    # threshold = 3100000

    # Generate Tardos codes
    # codes, p = generate_tardos_code(n_users, epsilon)
    codes, p = generate_codebook(n_users=n_users, secret_key=secret_key, fp_len=fp_len)  # generates entire codebook
    print(p)

    # Assume we have a marked code that we suspect is the result of collusion
    # marked_code = codes[0]  # In practice, this would be the suspicious code
    # Colluding two
    marked_code = np.where(codes[3] == codes[4], codes[3], 1)
    print("Marked code: ", marked_code)
    # Colluding three
    # marked_code = np.zeros(codes[0].shape, dtype=codes[0].dtype)
    # for i in range(len(codes[0])):
    #    values = [codes[0][i], codes[2][i], codes[4][i]]
    #    majority_value = mode(values)[0]
    #    marked_code[i] = majority_value

    # Detect a clean fingerprint
    # suspected_recipient = check_exact_matching(secret_key=secret_key, code=codes[2], n_users=n_users)
    # Detect colluders
    scores = sorted(score_users(marked_code, secret_key=secret_key, n_users=n_users).items(), key=lambda x: x[1], reverse=True)
    suspected_colluders = detect_colluders(code=marked_code, total_n_recipients=n_users, secret_key=secret_key, k=1)

    print("Generated Tardos Codes:\n", codes)
    print("Probability Vector:\n", p)
 #   print("Detection of a clean fingerprint:\n", suspected_recipient)
    print("Suspected Colluders:\n", suspected_colluders)
    print("Scores ranked:\n", scores)


if __name__ == '__main__':
    demo()
