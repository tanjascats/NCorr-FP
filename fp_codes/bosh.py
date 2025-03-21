import numpy as np


def min_diff(codes):
    """
    Calculates the minimum bit difference between any two fingerprints.
    Args:
        codes (np.ndarray): Matrix of fingerprint codes for all users.

    Returns:

    """
    n_rows = codes.shape[0]
    min_distance = np.inf

    # Compare each pair of bit-strings
    for i in range(n_rows):
        for j in range(i + 1, n_rows):
            distance = np.sum(codes[i] != codes[j])
            if distance < min_distance:
                min_distance = distance

    return min_distance


def generate_random_codes(n_users, code_length):
    """
    Generate random fingerprint codes for n_users.

    Args:
    n_users (int): Number of users.
    code_length (int): Length of each fingerprint (number of bits).

    Returns:
    np.ndarray: Matrix of fingerprint codes with shape (n_users, code_length).
    """
    # Generate random binary codes for each user (BoSh codes)
    codes = np.random.randint(0, 2, size=(n_users, code_length))
    return codes


def generate_bosh_codes(n_users, code_length, secret_key):
    """
        Generate Boneh-Shaw (BoSh) fingerprint codes for n_users.

        Args:
        n_users (int): Number of users. c in collusion notation, i.e. max number of detectable colluders.
        code_length (int): Length of each fingerprint (number of bits).

        Returns:
        np.ndarray: Matrix of fingerprint codes with shape (n_users, code_length).
        """
    r = code_length//n_users  # r -> block size
    if r < r:
        exit('Increase the length of the fingerprint code, or define less recipients.\n\tFor n_users={}, the min '
             'fingerprint length is {}.\n\tFor code_length={}, the max n_users is {}.'.format(n_users, n_users,
                                                                                              code_length, code_length))
    # Initialize the codebook (code_length = c * r)
    codebook = np.zeros((n_users, code_length), dtype=int)

    # Create the codebook where each user's codeword follows the rule
    for i in range(0, n_users):
        # Fill the first (i-1)*r positions with zeroes, the rest with ones
        codebook[i, :i * r] = 0
        codebook[i, i * r:] = 1

    # Apply the same random permutation to each codeword
    for i in range(n_users):
        np.random.seed(secret_key)
        np.random.shuffle(codebook[i])

    return codebook


def generate_pirate_fingerprint(codes, colluders):
    """
    Generate a pirate fingerprint from a set of colluders using majority voting.

    Args:
    codes (np.ndarray): Matrix of fingerprint codes for all users.
    colluders (list of int): List of user indices who are colluding.

    Returns:
    np.ndarray: The pirate fingerprint generated by colluding users.
    """
    pirate_fingerprint = np.zeros(codes.shape[1], dtype=int)

    # Use majority voting for each bit across colluders' fingerprints
    for j in range(codes.shape[1]):
        bits = codes[colluders, j]  # Get the bits from colluders at position j
        pirate_fingerprint[j] = 1 if np.sum(bits) > len(colluders) / 2 else 0

    return pirate_fingerprint


def detect_colluders(codes, pirate_fingerprint, t=1):
    """
    Detect colluders by calculating suspicion scores based on the pirate fingerprint.

    Args:
    codes (np.ndarray): Matrix of fingerprint codes for all users.
    pirate_fingerprint (np.ndarray): The pirate fingerprint obtained from collusion.
    t (float): The factor for standard deviation affecting the threshold for accusation. Larger values lead to more confidence.

    Returns:
    list: List of suspected colluders (user indices). List of suspicion scores.
    """
    # todo: this works but could be imporoved by introducing probabilities
    n_users, code_length = codes.shape
    suspicion_scores = np.zeros(n_users)

    # Calculate suspicion score for each user
    for i in range(n_users):
        for j in range(code_length):
            if codes[i][j] == pirate_fingerprint[j]:
                suspicion_scores[i] += np.log(2)  # Positive contribution for matching bit
            else:
                suspicion_scores[i] += np.log(1)  # No contribution for non-matching bit (log(1) = 0)

    threshold = suspicion_scores.mean() + t * suspicion_scores.std()

    # Accuse users whose suspicion score exceeds the threshold
    suspected_colluders = [i for i, score in enumerate(suspicion_scores) if score >= threshold]

    return suspected_colluders, suspicion_scores


def detect_recipient(codes, pirate_fingerprint):
    """
    Detects the potential recipient based on suspicious fingerprint by calculating similarity scores.
    Args:
        codes (np.ndarray): Matrix of fingerprint codes for all users.
        pirate_fingerprint (np.ndarray): The suspect fingerprint extracted from data.

    Returns: A top suspicious user (index). List of similarity scores.

    """
    n_users, code_length = codes.shape
    similarity_scores = np.zeros(n_users)

    # Calculate similarity score for each user
    for i in range(n_users):
        similarity_scores[i] = np.sum(pirate_fingerprint == codes[i]) / len(pirate_fingerprint)

    # Accuse user whose similarity is the highest
    suspected_user = np.argmax(similarity_scores)

    return suspected_user, similarity_scores


def example():
    n_users = 20
    code_length = 64
    colluders = [1, 2]  # These users collude to create a pirate fingerprint
    secret_key = 101

    # Generate BoSh codes
    codes = generate_bosh_codes(n_users, code_length, secret_key)
    print("Generated BoSh Codes:\n", codes)
    print("Min distance between any two codes: ", min_diff(codes))

    # Generate a pirate fingerprint by colluders
    pirate_fingerprint = generate_pirate_fingerprint(codes, colluders)
    print("Pirate Fingerprint (created by colluding majority voting):\n", pirate_fingerprint)

    # Try to detect one user
    suspect, similarity_scores = detect_recipient(codes, pirate_fingerprint)
    print("Suspected recipient: ", suspect)
    print("Confidence scores: ", similarity_scores)
    # Detect colluders based on the pirate fingerprint
    suspected_colluders, suspicion_scores = detect_colluders(codes, pirate_fingerprint, t=1)
    print("Suspected Colluders:", suspected_colluders)
    print("Suspicion Scores:", suspicion_scores)

    # Generate random codes
    codes = generate_random_codes(n_users, code_length)
    print("Generated Random Codes:\n", codes)
    print("Min distance between any two codes: ", min_diff(codes))

    # Generate a pirate fingerprint by colluders
    pirate_fingerprint = generate_pirate_fingerprint(codes, colluders)
    print("Pirate Fingerprint:\n", pirate_fingerprint)

    # Try to detect one user
    suspect, similarity_scores = detect_recipient(codes, pirate_fingerprint)
    print("Suspected recipient: ", suspect)
    print("Confidence scores: ", similarity_scores)
    # Detect colluders based on the pirate fingerprint
    suspected_colluders, suspicion_scores = detect_colluders(codes, pirate_fingerprint, t=1)
    print("Suspected Colluders:", suspected_colluders)
    print("Suspicion Scores:", suspicion_scores)


if __name__ == '__main__':
    example()
