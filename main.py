import NCorrFP_scheme.NCorrFP
import attacks.bit_flipping_attack
from NCorrFP_scheme.NCorrFP import NCorrFP
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, norm
import gower
import pandas as pd


def test_knn():
#    scheme = CategoricalNeighbourhood(gamma=1)
    scheme = NCorrFP(gamma=5, fingerprint_bit_length=16)
    data = "datasets/breast_cancer_full.csv"
    fingerprinted_data = scheme.insertion('breast-cancer', primary_key_name='Id', secret_key=101, recipient_id=4,
                                          outfile='NCorrFP_scheme/outfiles/fp_data_blind_corr_inv_node_101_g5.csv',
                                          correlated_attributes=['inv-nodes', 'node-caps'])

    suspect = scheme.detection(fingerprinted_data, secret_key=101, primary_key='Id',
                               correlated_attributes=['inv-nodes', 'node-caps'],
                               original_columns=["age","menopause","tumor-size","inv-nodes","node-caps","deg-malig","breast","breast-quad",
    "irradiat","recurrence"])


def knn_adult_census():
    scheme = NCorrFP(gamma=10, fingerprint_bit_length=32)

    fingerprinted_data = scheme.insertion('adult', primary_key_name='Id', secret_key=100, recipient_id=4,
                                          outfile='NCorrFP_scheme/outfiles/adult_fp_acc_wc_en_100.csv',
                                          correlated_attributes=['relationship', 'marital-status', 'occupation', 'workclass', 'education-num'])
    suspect = scheme.detection(fingerprinted_data, secret_key=100, primary_key='Id',
                               correlated_attributes=['relationship', 'marital-status', 'occupation', 'workclass',
                                                      'education-num'])


def test_vertical_attack_bc():
    scheme = NCorrFP(gamma=1, fingerprint_bit_length=8)
    fingerprinted_data = scheme.insertion('breast-cancer', primary_key_name='Id', secret_key=601, recipient_id=4,
                                          outfile='NCorrFP_scheme/outfiles/fp_data_blind_corr_all_attributes.csv',
                                          correlated_attributes=['age', 'menopause', 'inv-nodes', 'node-caps'])
    fingerprinted_data = fingerprinted_data.drop(['age'], axis=1)
    suspect = scheme.detection(fingerprinted_data, secret_key=601, primary_key='Id',
                               correlated_attributes=['age', 'menopause', 'inv-nodes', 'node-caps'],
                               original_columns=["age", "menopause", "tumor-size", "inv-nodes", "node-caps",
                                                 "deg-malig", "breast", "breast-quad",
                                                 "irradiat", "recurrence"])


def test_vertical_adult():
    scheme = NCorrFP(gamma=20, fingerprint_bit_length=32)

    fingerprinted_data = scheme.insertion('adult', primary_key_name='Id', secret_key=100, recipient_id=4,
                                          outfile='NCorrFP_scheme/outfiles/adult_fp_acc_wc_en_100.csv',
                                          correlated_attributes=['relationship', 'marital-status', 'occupation',
                                                                 'workclass', 'education-num'])
    fingerprinted_data = fingerprinted_data[["Id", "age", "workclass","fnlwgt","education","education-num",
                                                 "marital-status","occupation"]]
    suspect = scheme.detection(fingerprinted_data, secret_key=100, primary_key='Id',
                               correlated_attributes=['relationship', 'marital-status', 'occupation', 'workclass',
                                                      'education-num'],
                               original_columns=["age","workclass","fnlwgt","education","education-num",
                                                 "marital-status","occupation",
                                                 "relationship","race","sex","capital-gain","capital-loss",
                                                 "hours-per-week","native-country","income"])


def test_flipping_bc():
    scheme = NCorrFP(gamma=1, fingerprint_bit_length=16)

    fingerprinted_data = scheme.insertion('breast-cancer', primary_key_name='Id', secret_key=100, recipient_id=4,
                                          outfile='NCorrFP_scheme/outfiles/adult_fp_acc_wc_en_100.csv',
                                          correlated_attributes=['age', 'menopause', 'inv-nodes', 'node-caps'])
    attack = attacks.bit_flipping_attack.BitFlippingAttack()
    attacked_data = attack.run(fingerprinted_data, 0.01)
    print(attacked_data.size)
    scheme.detection(attacked_data, secret_key=100, primary_key='Id',
                     correlated_attributes=['age', 'menopause', 'inv-nodes', 'node-caps'])


def test_flipping_adult():
    scheme = NCorrFP(gamma=1, fingerprint_bit_length=16)

    fingerprinted_data = scheme.insertion('adult', primary_key_name='Id', secret_key=100, recipient_id=4,
                                          outfile='NCorrFP_scheme/outfiles/adult_fp_acc_wc_en_100.csv',
                                          correlated_attributes=['relationship', 'marital-status', 'occupation',
                                                                 'workclass', 'education-num'])
    attack = attacks.bit_flipping_attack.BitFlippingAttack()
    attacked_data = attack.run(fingerprinted_data, 0.01)
    scheme.detection(fingerprinted_data, secret_key=100, primary_key='Id',
                     correlated_attributes=['relationship', 'marital-status', 'occupation', 'workclass',
                                            'education-num'])


def test_demo():
    scheme = NCorrFP(gamma=1, fingerprint_bit_length=16)
    data = "datasets/breast_cancer_full.csv"
    fingerprinted_data, iter_log = scheme.demo_insertion('breast-cancer', primary_key_name='Id', secret_key=501,
                                                         recipient_id=4,
                                          outfile='NCorrFP_scheme/outfiles/fp_data_blind_corr_all_attributes.csv',
                                          correlated_attributes=['age', 'menopause', 'inv-nodes', 'node-caps'])
    #
    suspect, d_iter_log = scheme.demo_detection(fingerprinted_data, secret_key=501, primary_key='Id',
                               correlated_attributes=['age', 'menopause', 'inv-nodes', 'node-caps'],
                               original_columns=["age", "menopause", "tumor-size", "inv-nodes", "node-caps",
                                                 "deg-malig", "breast", "breast-quad",
                                                 "irradiat", "recurrence"])
    return d_iter_log


def knn_multi_corr():
    # code to replicate fingerprint embedding with NCorrFP scheme
    from NCorrFP_scheme.NCorrFP import NCorrFP
    scheme = NCorrFP(gamma=1, fingerprint_bit_length=16)
    fingerprinted_data = scheme.insertion('breast-cancer', primary_key_name='Id', secret_key=100, recipient_id=4,
                                          outfile='NCorrFP_scheme/outfiles/fp_data_multicorr_100.csv',
                                          correlated_attributes=[['age', 'menopause'], ['inv-nodes', 'node-caps']])

    suspect = scheme.detection(fingerprinted_data, secret_key=100, primary_key='Id',
                               correlated_attributes=[['age', 'menopause'], ['inv-nodes', 'node-caps']],
                               original_columns=["age", "menopause", "tumor-size", "inv-nodes", "node-caps",
                                                 "deg-malig", "breast", "breast-quad",
                                                 "irradiat", "recurrence"])


def estimate_distribution(data):
    # Create a kernel density estimate (KDE)
    kde = gaussian_kde(data)

    # Create a range of values from min to max of data
    x = np.linspace(min(data), max(data), 1000)

    # Evaluate KDE on the range of values
    kde_values = kde(x)

    # Plot the KDE result
    plt.plot(x, kde_values, label='KDE')
    plt.title('Kernel Density Estimate of the Data Distribution')
    plt.xlabel('Values')
    plt.ylabel('Density')

    plt.hist(data, bins=10, density=True)
    plt.show()

    # Return the KDE function and the x values for further use if needed
    return kde, x


def plot_pdf_from_data(data):
    # Fit a normal distribution to the data: Get the mean and standard deviation
    mean, std_dev = np.mean(data), np.std(data)

    # Create a range of x values (e.g., from min(data) to max(data))
    x = np.linspace(min(data), max(data), 1000)

    # Calculate the PDF values for this range using the normal distribution
    pdf_values = norm.pdf(x, mean, std_dev)

    # Plot the histogram of the data and the fitted PDF
    plt.hist(data, bins=10, density=True, alpha=0.6, color='g', label='Data Histogram')
    plt.plot(x, pdf_values, label=f'Fitted Normal PDF\n(mean={mean:.2f}, std={std_dev:.2f})', color='blue')
    plt.title('Fitted Normal Distribution and PDF')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.show()


def sample_with_threshold(data, percentage_to_exclude, num_samples=1):
    # Step 1: Create a KDE based on the data
    kde = gaussian_kde(data)

    # Step 2: Create a range of values over which to evaluate the KDE
    x = np.linspace(min(data), max(data), 1000)

    # Step 3: Evaluate the PDF and CDF
    pdf_values = kde(x)
    cdf_values = np.cumsum(pdf_values) / np.sum(pdf_values)  # Normalize CDF

    # Plot the PDF for visualization
    plt.plot(x, pdf_values, label='KDE')
    plt.title('PDF with dense areas')
    plt.xlabel('Values')
    plt.ylabel('Density')

    # Step 4: Exclude a certain percentage of the most dense areas
    threshold = np.percentile(cdf_values, 100 - percentage_to_exclude)
    valid_indices = np.where(cdf_values > threshold)[0]
    x_valid = x[valid_indices]

    # Step 5: Sample from the less dense areas
    sampled_indices = np.random.choice(valid_indices, size=num_samples)
    new_samples = x[sampled_indices]
    plt.hist(data, bins=10, density=True, alpha=0.6, label='Data')
    plt.hist(new_samples, bins=10, density=True, alpha=0.6, label='Sampled values')
    plt.legend()
    plt.show()

    return new_samples


def sample_from_dense_areas(data, exclude_percent=0.1, num_samples=1):
    # Create a KDE based on the data (PDF estimation)
    kde = gaussian_kde(data)

    # Create a range of values to evaluate the PDF
    x = np.linspace(min(data), max(data), 1000)
    pdf_values = kde(x)

    # Identify the threshold to exclude a percentage of the densest areas
    threshold = np.percentile(pdf_values, exclude_percent*100)

    # Mask the CDF to only include values within the percentile range
    mask = (pdf_values >= threshold)

    # Re-normalize the masked PDF and CDF
    masked_pdf = np.where(mask, pdf_values, 0)
    masked_cdf = np.cumsum(masked_pdf)
    masked_cdf /= masked_cdf[-1]

    # Inverse transform sampling from the adjusted CDF
    random_values = np.random.rand(num_samples)
    sampled_values = np.interp(random_values, masked_cdf, x)
    print(sampled_values)

    # Plot the PDF, masked PDF, and the sampled values
    plt.plot(x, pdf_values, label='Original PDF')
    plt.plot(x, masked_pdf, label='Modified PDF ({}th percentile)'.format(int(100*exclude_percent)))
    plt.scatter(sampled_values, [0] * num_samples, color='red', label='Sampled Values', zorder=5)
    plt.title('Sampling from Dense Areas')
    plt.xlabel('Values')
    plt.ylabel('Density')
    plt.legend()
    plt.show()

    return sampled_values


def test_gower_distance():
    # Example DataFrame with both categorical and continuous columns
    data = pd.DataFrame({
        'Age': [25, 30],
        'Height': [170, 180],
        'Gender': ['Male', 'Female'],
    #    'Color': ['Red', 'Red']
    })

    # Calculate Gower distance matrix
    gower_distance_matrix = gower.gower_matrix(data)
    print(gower_distance_matrix)


def test_detection_continuous():
    scheme = NCorrFP(gamma=2, fingerprint_bit_length=32, k=100)
    original_path = "NCorrFP_scheme/test/test_data/synthetic_1000_3_continuous.csv"
    original = pd.read_csv(original_path)
    correlated_attributes = ['X', 'Y']
    recipient = 2
    fingerprinted_data = scheme.insertion(original_path, primary_key_name='Id', secret_key=101, recipient_id=recipient,
                                          correlated_attributes=correlated_attributes, save_computation=True,
                                          outfile='NCorrFP_scheme/test/out_data/test_id{}.csv'.format(recipient))
    suspect = scheme.detection(fingerprinted_data, secret_key=101, primary_key='Id',
                               correlated_attributes=correlated_attributes,
                               original_columns=["X", 'Y', 'Z'])


if __name__ == '__main__':
    # data = np.random.randint(0, 1000, 100)  # Generating some example data
    #kde_function, x_values = estimate_distribution(data)
    #plot_pdf_from_data(data)
    #data = np.random.normal(loc=0, scale=1, size=1000)  # Example data from a normal distribution

    # Exclude the top 10% most dense areas and sample new values
    # new_values = sample_from_dense_areas(data, exclude_percent=0.1, num_samples=20)
    # print("New sampled values (from less dense areas):", new_values)
    #test_gower_distance()
    #NCorrFP_scheme.NCorrFP.plot_runtime()
    test_detection_continuous()
