import pandas as pd
import glob


def merge_csv_files(directory_path):
    """
    Merges all CSV files in the specified directory that match specific patterns
    'robustness-{category}-adult*.csv' into separate CSV files for each category.
    Removes duplicates.

    Args:
    directory_path: Path to the directory containing the CSV files.
    """
    categories = ["horizontal", "vertical", "flipping", "clusterflipping", "clusterhorizontal"]
    for category in categories:
        # Pattern to match the files for each category
        file_pattern = f"{directory_path}/robustness-{category}_adult*.csv"

        # List all files that match the pattern
        csv_files = glob.glob(file_pattern)

        if not csv_files:
            print(f"No files found for category: {category}")
            continue

        # Read and concatenate all files into a single DataFrame
        df_list = [pd.read_csv(file) for file in csv_files]
        combined_df = pd.concat(df_list, ignore_index=True)

        # Remove duplicate entries based on all columns
        combined_df.drop_duplicates(inplace=True)

        # Save the combined DataFrame to a new CSV file
        output_filename = f"robustness_{category}_adult.csv"
        combined_df.to_csv(output_filename, index=False)
        print(f"Merged file saved as {output_filename}")


# Example usage
merge_csv_files('.')
