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


def generate_tardos_code(n_users, epsilon):
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


def detect_colluders(codes, marked_code, p, k=1):  # threshold
    """
    Detects colluders based on the marked code and threshold.

    Args:
    codes (np.ndarray): Matrix of Tardos codes of shape (n_users, code_length).
    marked_code (np.ndarray): The marked code (suspicious code).
    p (np.ndarray): Probability vector used to generate the codes.
    threshold (float): Detection threshold.

    Returns:
    list: List of suspected colluders.
    """

    n_users, code_length = codes.shape
    scores = np.zeros(n_users)

    for user in range(n_users):
        for pos in range(code_length):
            if marked_code[pos] == 1:
                scores[user] += np.log(1 / p[pos]) if codes[user, pos] == 1 else np.log(1 / (1 - p[pos]))
            else:
                scores[user] += np.log(1 / (1 - p[pos])) if codes[user, pos] == 1 else np.log(1 / p[pos])
    print("Scores: ", scores)

    # Calculate dynamic threshold based on mean and standard deviation
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    threshold = mean_score + k * std_score
    print("Dynamic threshold: ", threshold)

    # Identify colluders
    colluders = [user for user in range(n_users) if scores[user] > threshold]

    return colluders


# Example usage
n_users = 10
epsilon = 0.1
# threshold = 3100000

# Generate Tardos codes
codes, p = generate_tardos_code(n_users, epsilon)

# Assume we have a marked code that we suspect is the result of collusion
# marked_code = codes[0]  # In practice, this would be the suspicious code
# marked_code = np.where(codes[0] == codes[9], codes[0], 1)  # Colluding two
# Colluding three
marked_code = np.zeros(codes[0].shape, dtype=codes[0].dtype)
for i in range(len(codes[0])):
    values = [codes[0][i], codes[2][i], codes[5][i]]
    majority_value = mode(values)[0]
    marked_code[i] = majority_value

# Detect colluders
suspected_colluders = detect_colluders(codes, marked_code, p) #, threshold)

print("Generated Tardos Codes:\n", codes)
print("Probability Vector:\n", p)
print("Suspected Colluders:\n", suspected_colluders)
