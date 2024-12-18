import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from itertools import product
from matplotlib import colors
from scipy.stats import gaussian_kde


def plot_vote_error_rate(results):
    """
    Plots the vote error rate with standard deviation shading for different neighborhood sizes.

    Args:
    - results (pd.DataFrame): DataFrame containing 'embedding_ratio', 'vote error rate', and 'neighbourhood size' columns.
    """
    # Group by 'gamma' and 'neighbourhood size' to calculate mean and standard deviation
    grouped_data = results.groupby(['embedding_ratio', 'k'])['vote_error'].agg(['mean', 'std']).reset_index()

    # Plotting
    plt.figure(figsize=(10, 6))
    markers = ['o', 'v', 'x', 's']
    # Iterate through each neighborhood size for separate lines and shaded areas
    for i, neighborhood_size in enumerate(grouped_data['k'].unique()):
        # Filter data for the current neighborhood size
        data = grouped_data[grouped_data['k'] == neighborhood_size]

        # Plot line for average vote error rate
        plt.plot(data['embedding_ratio'], data['mean'],
                 label=f'{neighborhood_size} / {round(100 * neighborhood_size / 30000, 2)}%',
                 marker=markers[i])

        # Plot shaded area for standard deviation
        plt.fill_between(data['embedding_ratio'], data['mean'] - data['std'], data['mean'] + data['std'], alpha=0.2)

    # Plot the ideal scenario for comparison
    # plt.plot(data['embedding_ratio'], [0.0 for i in range(len(data['embedding_ratio']))], linestyle=(0, (5,10)), linewidth=1.4,
    #         label=f'Ideal scenario')

    # plt.ylim(0, 1)
    # Labels and legend
    plt.xlabel('Fingerprint embedding ratio (1/gamma) [0, 1]')
    plt.ylabel('Vote Error Rate [0, 1]')
    plt.title('Vote Error Rate vs Fingerprint embedding ratio with Neighborhood Size')
    plt.legend(title='Neighborhood Size\n(#records / %data size)')
    plt.grid(True)
    plt.show()


def plot_tp_confidence(results):
    """
    Plots the confidence rate for matching the correct recipient with standard deviation shading for different FP lengths.

    Args:
    - results (pd.DataFrame): DataFrame containing 'embedding_ratio', 'tp', and 'fingerprint_length' columns.
    """
    # Group by 'embedding_ratio' and 'fingerprint length' to calculate mean and standard deviation
    grouped_data = results.groupby(['embedding_ratio', 'fingerprint_length'])['tp'].agg(['mean', 'std']).reset_index()

    # Plotting
    plt.figure(figsize=(10, 6))
    markers = ['o', 'v', 'x', 's']

    # Iterate through each neighborhood size for separate lines and shaded areas
    for i, fingerprint_length in enumerate(grouped_data['fingerprint_length'].unique()):
        # Filter data for the current neighborhood size
        data = grouped_data[grouped_data['fingerprint_length'] == fingerprint_length]

        # Plot line for average vote error rate
        plt.plot(data['embedding_ratio'], data['mean'], label=f'{fingerprint_length}-bit', marker=markers[i])

        # Plot shaded area for standard deviation
        plt.fill_between(data['embedding_ratio'], data['mean'] - data['std'], data['mean'] + data['std'], alpha=0.2)

    # Labels and legend
    plt.xlabel('Fingerprint embedding ratio (1/gamma) [0, 1]')
    plt.ylabel('Correct extraction confidence [0, 1]')
    plt.title('Correct extraction confidence')
    plt.legend(title='Fingerprint Length (in bits)')
    plt.grid(True)
    plt.show()


def plot_tn_confidence(results):
    """
    Plots the confidence rate for matching the wrong recipient with standard deviation shading for different FP lengths.

    Args:
    - results (pd.DataFrame): DataFrame containing 'embedding_ratio', 'tn', and 'fingerprint_length' columns.
    """
    # Group by 'embedding_ratio' and 'fingerprint length' to calculate mean and standard deviation
    grouped_data = results.groupby(['embedding_ratio', 'fingerprint_length'])['tn'].agg(['mean', 'std']).reset_index()

    # Plotting
    plt.figure(figsize=(10, 6))
    markers = ['o', 'v', 'x', 's']

    # Iterate through each neighborhood size for separate lines and shaded areas
    for i, fingerprint_length in enumerate(grouped_data['fingerprint_length'].unique()):
        # Filter data for the current neighborhood size
        data = grouped_data[grouped_data['fingerprint_length'] == fingerprint_length]

        # Plot line for average vote error rate
        plt.plot(data['embedding_ratio'], data['mean'], label=f'{fingerprint_length}-bit', marker=markers[i])

        # Plot shaded area for standard deviation
        plt.fill_between(data['embedding_ratio'], data['mean'] - data['std'], data['mean'] + data['std'], alpha=0.2)

    # Labels and legend
    plt.xlabel('Fingerprint embedding ratio (1/gamma) [0, 1]')
    plt.ylabel('Wrong extraction confidence [0, 1]')
    plt.title('Wrong extraction confidence')
    plt.legend(title='Fingerprint Length (in bits)')
    plt.grid(True)
    plt.show()


def plot_tp_tn_confidence(results):
    """
        Plots the confidence rate for matching the wrong recipient and true recipient with standard deviation shading
        for different FP lengths.

        Args:
        - results (pd.DataFrame): DataFrame containing 'embedding_ratio', 'tn', 'tp', 'code' and 'fingerprint_length' columns.
        """
    # Group by 'embedding_ratio' and 'fingerprint_length' to calculate mean and standard deviation
    grouped_data_tn = results.groupby(['embedding_ratio', 'fingerprint_length', 'code'])['tn'].agg(['mean', 'std']).reset_index()
    grouped_data_tp = results.groupby(['embedding_ratio', 'fingerprint_length'])['tp'].agg(['mean', 'std']).reset_index()

    # Plotting
    plt.figure(figsize=(10, 6))
    markers = ['o', 'v', 'x', 's']

    # Iterate through each neighborhood size for separate lines and shaded areas
    for i, fingerprint_length in enumerate(grouped_data_tn['fingerprint_length'].unique()):
        # Filter data for the current neighborhood size
        data_tn_hash = grouped_data_tn[(grouped_data_tn['fingerprint_length'] == fingerprint_length) & (grouped_data_tn['code'] == 'hash')]
        data_tn_tardos = grouped_data_tn[(grouped_data_tn['fingerprint_length'] == fingerprint_length) & (grouped_data_tn['code'] == 'tardos')]
        data_tp = grouped_data_tp[grouped_data_tp['fingerprint_length'] == fingerprint_length]

        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        line_styles = [':', '--', '-']  # dotted, dashed, solid
        markers = ['o', 'v', 'x', 's']

        # Assign the color for this fingerprint length
        color = colors[i]

        # Plot for tn_hash
        plt.plot(data_tn_hash['embedding_ratio'], data_tn_hash['mean'], label=f'{fingerprint_length}-bit (hash, tn)', color=color, linestyle=line_styles[0], marker=markers[i])
        plt.fill_between(data_tn_hash['embedding_ratio'],
                         data_tn_hash['mean'] - data_tn_hash['std'],
                         data_tn_hash['mean'] + data_tn_hash['std'],
                         color=color, alpha=0.2)

        # Plot for tn_tardos
        plt.plot(data_tn_tardos['embedding_ratio'], data_tn_tardos['mean'], label=f'{fingerprint_length}-bit (tardos, tn)', color=color, linestyle=line_styles[1], marker=markers[i])
        plt.fill_between(data_tn_tardos['embedding_ratio'],
                         data_tn_tardos['mean'] - data_tn_tardos['std'],
                         data_tn_tardos['mean'] + data_tn_tardos['std'],
                         color=color, alpha=0.2)

        # Plot for tp
        plt.plot(data_tp['embedding_ratio'], data_tp['mean'], label=f'{fingerprint_length}-bit (tp)', color=color, linestyle=line_styles[2], marker=markers[i]) # linewidth=2
        plt.fill_between(data_tp['embedding_ratio'],
                         data_tp['mean'] - data_tp['std'],
                         data_tp['mean'] + data_tp['std'],
                         color=color, alpha=0.2)

    # Labels and legend
    plt.xlabel('Fingerprint embedding ratio (1/gamma) [0, 1]')
    plt.ylabel('Extraction confidence [0, 1]')
    plt.title('Extraction confidence')
    plt.legend()#title='Fingerprint Length (in bits)')
    plt.grid(True)
    plt.show()


def plot_data_accuracy(results):
    """
    Plots the data accuracy, i.e. how many values change in the fingerprinted dataset.

    Args:
    - results (pd.DataFrame): DataFrame containing 'embedding_ratio', 'accuracy', and 'neighbourhood size' columns.
    """
    # Group by 'gamma' and 'neighbourhood size' to calculate mean and standard deviation
    grouped_data = results.groupby(['embedding_ratio', 'k'])['accuracy'].agg(['mean', 'std']).reset_index()

    # Plotting
    plt.figure(figsize=(10, 6))
    markers = ['o', 'v', 'x', 's']

    # Iterate through each neighborhood size for separate lines and shaded areas
    for i, neighborhood_size in enumerate(grouped_data['k'].unique()):
        # Filter data for the current neighborhood size
        data = grouped_data[grouped_data['k'] == neighborhood_size]

        # Plot line for average vote error rate
        plt.plot(data['embedding_ratio'], 1.0 - data['mean'],
                 label=f'{neighborhood_size} / {round(100 * neighborhood_size / 30000, 2)}%',
                 marker=markers[i])

        # Plot shaded area for standard deviation
        plt.fill_between(data['embedding_ratio'], 1.0 - data['mean'] - data['std'], 1.0 - data['mean'] + data['std'],
                         alpha=0.2)

    # Plot the ideal scenario for comparison
    # plt.plot(data['embedding_ratio'], [0.0 for i in range(len(data['embedding_ratio']))], linestyle=(0, (5,10)), linewidth=1.4,
    #         label=f'Ideal scenario')

    # plt.ylim(0, 1)
    # Labels and legend
    plt.xlabel('Fingerprint embedding ratio (1/gamma) [0, 1]')
    plt.ylabel('Dataset accuracy [0, 1]')
    plt.title('Dataset accuracy (value changes in a fingerprinted dataset)')
    plt.legend(title='Neighborhood Size\n(#records / %data size)')
    plt.grid(True)
    plt.show()


def plot_delta_mean(results):
    """
    Plots the vote error rate with standard deviation shading for different neighborhood sizes.

    Args:
    - results (pd.DataFrame): DataFrame containing 'embedding_ratio', 'vote error rate', and 'neighbourhood size' columns.
    """
    # Group by 'gamma' and 'neighbourhood size' to calculate mean and standard deviation
    grouped_data = results.groupby(['embedding_ratio', 'attribute'])['rel_delta_mean'].agg(
        ['mean', 'std']).reset_index()

    # Plotting
    plt.figure(figsize=(10, 6))

    # Iterate through each neighborhood size for separate lines and shaded areas
    for i, attribute in enumerate(grouped_data['attribute'].unique()):
        # Filter data for the current neighborhood size
        data = grouped_data[grouped_data['attribute'] == attribute]

        # Plot line for average vote error rate
        plt.plot(data['embedding_ratio'], data['mean'], label=f'{attribute}', marker='o')

        # Plot shaded area for standard deviation
        plt.fill_between(data['embedding_ratio'], data['mean'] - data['std'], data['mean'] + data['std'], alpha=0.2)

    # Plot the ideal scenario for comparison
    # plt.plot(data['embedding_ratio'], [0.0 for i in range(len(data['embedding_ratio']))], linestyle=(0, (5,10)), linewidth=1.4,
    #         label=f'Ideal scenario')

    # plt.ylim(0, 1)
    # Labels and legend
    plt.xlabel('Fingerprint embedding ratio (1/gamma) [0, 1]')
    plt.ylabel('Relative \delta mean value')
    plt.title('Change in mean value of the attributes due to a fingerprint')
    plt.legend(title='Attribute')
    plt.grid(True)
    plt.show()


def plot_delta_std(results):
    """
    Plots the vote error rate with standard deviation shading for different neighborhood sizes.

    Args:
    - results (pd.DataFrame): DataFrame containing 'embedding_ratio', 'vote error rate', and 'neighbourhood size' columns.
    """
    # Group by 'gamma' and 'neighbourhood size' to calculate mean and standard deviation
    grouped_data = results.groupby(['embedding_ratio', 'attribute'])['rel_delta_std'].agg(['mean', 'std']).reset_index()

    # Plotting
    plt.figure(figsize=(10, 6))

    # Iterate through each neighborhood size for separate lines and shaded areas
    for i, attribute in enumerate(grouped_data['attribute'].unique()):
        # Filter data for the current neighborhood size
        data = grouped_data[grouped_data['attribute'] == attribute]

        # Plot line for average vote error rate
        plt.plot(data['embedding_ratio'], data['mean'], label=f'{attribute}', marker='o')

        # Plot shaded area for standard deviation
        plt.fill_between(data['embedding_ratio'], data['mean'] - data['std'], data['mean'] + data['std'], alpha=0.2)

    # Plot the ideal scenario for comparison
    # plt.plot(data['embedding_ratio'], [0.0 for i in range(len(data['embedding_ratio']))], linestyle=(0, (5,10)), linewidth=1.4,
    #         label=f'Ideal scenario')

    # plt.ylim(0, 1)
    # Labels and legend
    plt.xlabel('Fingerprint embedding ratio (1/gamma) [0, 1]')
    plt.ylabel('Relative \delta std value')
    plt.title('Change in standard deviation of the attributes due to a fingerprint')
    plt.legend(title='Attribute')
    plt.grid(True)
    plt.show()


def plot_histogram_grid(datasets, baseline_dataset, columns=5, bins=10, gamma=None):
    """
    Plots a grid of histograms for multiple datasets with a baseline dataset for comparison.

    Args:
    - datasets (list of pd.DataFrame): List of DataFrames (each representing a dataset).
    - baseline_dataset (pd.DataFrame): Baseline DataFrame to compare against.
    - columns (int): Number of columns (datasets) in the grid.
    - bins (int): Number of bins for the histograms.
    """
    # Get the list of attributes (assumes all datasets have the same columns)
    if gamma is None:
        gamma = [32, 16, 8, 4, 2]
    attributes = datasets[0].columns
    rows = len(attributes)  # Number of rows in the grid for each attribute

    # Set up the figure size based on rows and columns
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 4, rows * 3))
#   fig.suptitle("Histograms of Dataset Attributes with Baseline Comparison", y=1.02, fontsize=16)

    # Loop through each attribute and dataset to create histograms
    for i, attribute in enumerate(attributes):
        # Loop through each dataset
        for j, dataset in enumerate(datasets):
            ax = axes[i, j]
            ax.hist(dataset[attribute], bins=bins, alpha=0.7, color='blue', edgecolor='black', label='original')

            # Overlay the baseline dataset's histogram as an outline
            ax.hist(baseline_dataset[attribute], bins=bins, histtype='step', linewidth=1.5, color='red',
                    edgecolor='red', label='fingerprinted')

            # Set labels only for the first column and bottom row
            if i == 0:
                ax.set_title(f"Embedding ratio {round(1 / gamma[j], 2)}")
            if j == 0:
                ax.set_ylabel(attribute)

            # Hide x and y ticks if not on bottom row or first column
            # if i < rows - 1:
            #    ax.set_xticklabels([])
            # if j > 0:
            #    ax.set_yticklabels([])

    # Adjust spacing and show plot
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.legend()
    plt.show()


def plot_delta_corr(results):
    """
    Plots the relative correlation change.

    Args:
    - results (pd.DataFrame): DataFrame containing 'embedding_ratio', 'vote error rate', and 'neighbourhood size' columns.
    """
    # Plotting
    plt.figure(figsize=(10, 6))

    # Iterate through each attribute pair for separate lines and shaded areas
    for pair in results.columns[-6:]:
        # Group by 'gamma' to calculate mean and standard deviation
        grouped_data = results.groupby(['embedding_ratio'])[pair].agg(['mean', 'std']).reset_index()

        # Plot line for average vote error rate
        plt.plot(grouped_data['embedding_ratio'], grouped_data['mean'], label=f'{pair}', marker='o')

        # Plot shaded area for standard deviation
        plt.fill_between(grouped_data['embedding_ratio'], grouped_data['mean'] - grouped_data['std'],
                         grouped_data['mean'] + grouped_data['std'], alpha=0.2)

    # plt.ylim(0, 1)
    # Labels and legend
    plt.xlabel('Fingerprint embedding ratio (1/gamma) [0, 1]')
    plt.ylabel('Relative \delta pearson correlation')
    plt.title('Correlation change between pairs of attributes')
    plt.legend(title='Attribute pair')
    plt.grid(True)
    plt.show()


def plot_fidelity(results):
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(24, 10), sharex=True)
    metrics = ['rel_delta_mean', 'rel_delta_std', 'hellinger_distance', 'kl_divergence', 'emd', 'ks']

    for i, ax in enumerate(axes.flat):
        if i > 6:
            break
        # Group by 'gamma' and 'neighbourhood size' to calculate mean and standard deviation
        grouped_data = results.groupby(['embedding_ratio', 'attribute'])[metrics[i]].agg(
            ['mean', 'std']).reset_index()

        # Iterate through each neighborhood size for separate lines and shaded areas
        for j, attribute in enumerate(grouped_data['attribute'].unique()):
            # Filter data for the current neighborhood size
            data = grouped_data[grouped_data['attribute'] == attribute]

            # Plot line for average vote error rate
            ax.plot(data['embedding_ratio'], data['mean'], label=f'{attribute}', marker='o')

            # Plot shaded area for standard deviation
            ax.fill_between(data['embedding_ratio'], data['mean'] - data['std'], data['mean'] + data['std'], alpha=0.2)
        ax.set_xlabel('Fingerprint embedding ratio (1/gamma) [0, 1]')
        ax.set_ylabel(f'{metrics[i]}')
        ax.grid(True)
        ax.set_title(f'{metrics[i]}')
        if i == 0:
            ax.legend(title='Attribute')

    # Adjust layout to avoid overlapping
    plt.tight_layout()
    plt.show()


# for converting fingerprint to color pattern for plotting the count state
def binary_to_pairs(binary_list):
    # Initialize an empty list to store the result
    result = []

    # Iterate over each element in the binary list
    for value in binary_list:
        if value == 0:
            # If the input value is 0, create the pair (1, 0)
            pair = (1, 0)
        else:
            # If the input value is 1, create the pair (0, 1)
            pair = (0, 1)
        # Append the pair to the result list
        result.append(pair)

    return result


def plot_count_updates(count_state, fingerprint):
    # Extract the first and second values of each pair for the rows
    first_row = [pair[0] for pair in count_state]
    second_row = [pair[1] for pair in count_state]

    # Extract the color flags for each row
    color_flags = binary_to_pairs(fingerprint)  # 1 = green, 0 = red
    first_row_flags = [flag[0] for flag in color_flags]
    second_row_flags = [flag[1] for flag in color_flags]

    # Normalize values for color intensity
    all_values = first_row + second_row
    norm = colors.Normalize(vmin=min(all_values), vmax=max(all_values))

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(len(count_state), 2))

    # Create the table with two rows (first row, second row)
    # Arrange large fingerprints into a few rows of 32 bits
    table_data = [first_row, second_row]
    table = ax.table(cellText=table_data, loc='center', cellLoc='center')#, colWidths=[0.05] * len(count_state))  # [0.05]

    # Hide the axes
    ax.axis('off')

    # Iterate over the table cells to set the colors based on value and color flag
    for i in range(2):  # Two rows
        for j in range(len(count_state)):
            # Get the value for the cell
            value = table_data[i][j]

            # Determine the color based on the color flags
            if i == 0:  # First row
                if first_row_flags[j] == 1:
                    color = plt.cm.Greens(norm(value))  # Green colormap if flag is 1
                else:
                    color = plt.cm.Reds(norm(value))  # Red colormap if flag is 0
            else:  # Second row
                if second_row_flags[j] == 1:
                    color = plt.cm.Greens(norm(value))  # Green colormap if flag is 1
                else:
                    color = plt.cm.Reds(norm(value))  # Red colormap if flag is 0

            # Set the color of the cell
            cell = table[(i, j)]
            cell.set_facecolor(color)
            cell.set_text_props(color='black')

    # Display the plot
    plt.show()


def values_for_plot_distribution(target_values):
    data = target_values
    kde = gaussian_kde(data)
    # Create a range of values to evaluate the PDF
    x = np.linspace(min(data), max(data), 1000)
    pdf_values = kde(x)

    threshold = np.percentile(pdf_values, 0.75 * 100)

    mask = (pdf_values >= threshold)
    # Re-normalize the masked PDF and CDF
    masked_pdf = np.where(mask, pdf_values, 0)
    masked_cdf = np.cumsum(masked_pdf)
    masked_cdf /= masked_cdf[-1]

    return x, pdf_values, masked_pdf


def show_embedding_iteration(iteration, fingerprinted_data, iter_log, dataset):
    print("Marking record no. " + str(iter_log[iteration]['row_index']))
    print("Marking attribute: " + str(iter_log[iteration]['attribute']))
    print("The record to mark: \n" + str(dataset.iloc[[iteration]]))
    print('------------------------------------------------------------------------------------------------------------------')
    print('How to mark the new value:')
    print(f"Based on PRNG from this iteration, we use fingerprint bit at index {iter_log[iteration]['fingerprint_idx']}"
          f", i.e. bit {iter_log[iteration]['fingerprint_bit']} and xor it with the mask bit (also from PRNG), in this "
          f"case {iter_log[iteration]['mask_bit']}.\nThis operation gives us the MARK BIT. Mark bit determines how we "
          f"sample the new value.")
#    if iter_log[iteration]['attribute'] in correlated_attributes:
#        other = list(correlated_attributes); other.remove(iter_log[iteration]['attribute'])
#        print('Neighbourhood: ' +str(iter_log[iteration]['attribute']) + ' is correlated to '+ str(other)+ ' so we are finding the records with most similar values to ' + str(other) + '=' + str(dataset.iloc[iteration][other[0]]))
#    else:
#        print('Neighbourhood: ' + str(iter_log[iteration]['attribute']) + ' is not a correlated attribute, so we are including all attributes to find the closest neighbourhood.')
#    print('Neighbours idx: ' + str(iter_log[iteration]['neighbors']))
#    print('Neighbours dist: ' + str(iter_log[iteration]['dist']))
    print('\nFirst, we look at the values of attribute ' + str(iter_log[iteration]['attribute']) + ' in this neighbourhood, and among these is our potential new value.')
#    print('Target values:' + str(list(dataset.iloc[iter_log[iteration]['neighbors']][iter_log[iteration]['attribute']])))
    print('For this we estimate the distribution of these target values (see the plot below) before sampling one new value.')
    print('There are generally two outcomes for the sampled value:\n\t-mark bit is 1 (50%) - the new value is sampled from the most dense areas of a distribution of the target variable in the neighbourhood\n\t-mark bit is 0(50%) - the new value is sampled from the tails of distribution of the target value in the neighbourhood')
    mark_bit = iter_log[iteration]['mark_bit']
    if mark_bit == 1:
        print('In this case, mark bit is {}, therefore we sample from the dense part of distribution of the variable {} in the neighbourhood. The thresholds are set by an arbitrary percentile (here we use 75th)'.format(mark_bit, iter_log[iteration]['attribute']))
    else:
        print('In this case, mark bit is {}, therefore we sample from tails of distribution of the variable {} in the neighbourhood. The thresholds are set by an arbitrary percentile (here we use 75th)'.format(mark_bit, iter_log[iteration]['attribute']))

    data = list(dataset.iloc[iter_log[iteration]['neighbors']][iter_log[iteration]['attribute']])
    x = np.linspace(min(data), max(data), 100)  # n_points)  # 1000)
    kde = gaussian_kde(data)
    pdf_values = kde(x)
    threshold = np.percentile(pdf_values, 0.75*100)
    if mark_bit:
        mask = (pdf_values >= threshold)
    else:
        mask = (pdf_values < threshold)
    masked_pdf = np.where(mask, pdf_values, 0)
    masked_cdf = np.cumsum(masked_pdf)
    masked_cdf /= masked_cdf[-1]
    np.random.seed(iter_log[iteration]['seed'])
    random_values = np.random.rand(1)
    sampled_values = np.interp(random_values, masked_cdf, x)

    plt.plot(x, pdf_values, label='Original PDF (estimate)')
    plt.plot(x, masked_pdf, label='Modified PDF ({}th percentile)'.format(int(100 * 0.75)))
    plt.scatter(sampled_values, [0] * 1, color='red', label='Sampled Values', zorder=5)
    plt.hist(data, bins=10, density=True, alpha=0.3, label='Neighbourhood data points')
    if mark_bit:
        plt.title('Sampling from high density areas (mark bit = 1)')
    else:
        plt.title('Sampling from low density areas (mark bit = 0)')
    plt.xlabel('Values')
    plt.ylabel('Density')
    plt.legend()
    plt.show()

    print("The sampled continuous value is rounded to the closest existing value from the data (to avoid "
          "perceptibility of marks) and is: " + str(iter_log[iteration]['new_value']))
    print("The fingerprinted record is:")
    print(fingerprinted_data.iloc[[iteration]])


def show_detection_iteration(iteration, det_iter_log, fingerprinted_data, iter_log, fingerprint, dataset):
    print("Detecting from record at idx: " + str(det_iter_log[iteration]['row_index']))
    print("Detecting from attribute: " + str(det_iter_log[iteration]['attribute']))
    print(fingerprinted_data.iloc[[iteration]])
    fingerprinted_value = fingerprinted_data.iloc[det_iter_log[iteration]['row_index']][
        det_iter_log[iteration]['attribute']]
    print('Fingerpritned value: ' + str(fingerprinted_value))
    target_values_det = fingerprinted_data.iloc[det_iter_log[iteration]['neighbors']][
        det_iter_log[iteration]['attribute']].tolist()
    print("----------------------------------------------------------")
#    print("Obtaining neighbourhood....")
#    print('Target values:' + str(target_values_det))
#    print("----------------------------------------------------------")

    print("Observing the distribution of target attribute {} below...".format(det_iter_log[iteration]['attribute']))
    message = ' (i.e. tails of distribution)' if det_iter_log[iteration]['mark_bit'] == 0 else ' (i.e. in densest area)'
    print("Mark bit (where in distribution falls the target value?): " + str(
        det_iter_log[iteration]['mark_bit']) + message)
    print("Mask bit (from PRNG): " + str(det_iter_log[iteration]['mask_bit']))
    print("Fingerprint bit index (from PRNG): " + str(det_iter_log[iteration]['fingerprint_idx']))
    print("Fingerprint bit value (mark bit xor mask bit): " + str(det_iter_log[iteration]['fingerprint_bit']))

    if int(det_iter_log[iteration]['fingerprint_bit']) == int(fingerprint[det_iter_log[iteration]['fingerprint_idx']]):
        print('\nFingerprint bit CORRECT :)')
    else:
        print('\nFingerprint bit FALSE :( (it is just a wrong vote)')

    x_det, pdf_values_det, masked_pdf_det = values_for_plot_distribution(target_values_det)
    target_values_ins = list(dataset.iloc[iter_log[iteration]['neighbors']][iter_log[iteration]['attribute']])
    x_ins, pdf_values_ins, masked_pdf_ins = values_for_plot_distribution(target_values_ins)

    fig, axs = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(10, 4))

    axs[0].plot(x_det, pdf_values_det, label='Original PDF\n (estimate)')
    axs[0].plot(x_det, masked_pdf_det, label='Modified PDF\n ({}th percentile)'.format(int(100 * 0.75)))
    axs[0].hist(target_values_det, bins=10, density=True, alpha=0.3, label='Neighbourhood\n data points')
    axs[0].scatter(fingerprinted_value, 0, color='red', label='Marked value', zorder=5)
    axs[0].set_ylabel('Density')
    axs[0].set_xlabel('Values')
    axs[0].set_title('Fingerprinted data')
    axs[0].legend(prop={'size': 8})

    axs[1].plot(x_ins, pdf_values_ins, label='Original PDF\n (estimate)')
    axs[1].plot(x_ins, masked_pdf_ins, label='Modified PDF\n ({}th percentile)'.format(int(100 * 0.75)))
    axs[1].hist(target_values_ins, bins=10, density=True, alpha=0.3, label='Neighbourhood\n data points')
    axs[1].scatter(iter_log[iteration]['new_value'], 0, color='red', label='Marked value', zorder=5)
    axs[1].set_ylabel('Density')
    axs[1].set_xlabel('Values')
    axs[1].set_title('Original data')
    axs[1].legend(prop={'size': 8})

    plot_count_updates(det_iter_log[iteration]['count_state'], fingerprint)
    print(
        'Table: fingerprint count updates after this iteration (iteration {}). Each column is one fingerprint bit position (e.g. 16-bit --> 16 columns), and each row represents votes for either 0 or 1 being the value of that bit. The final decision is made at the end af the detection algorithm according to majority vote.'.format(
            iteration))