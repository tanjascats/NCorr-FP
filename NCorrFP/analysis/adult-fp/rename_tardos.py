import os


def rename_csv_files():
    # Get the list of all files in the current directory
    files = [f for f in os.listdir('.') if os.path.isfile(f)]

    for file in files:
        # Check if the file is a .csv file
        if file.endswith('.csv') and not (file.endswith('_codetardos.csv') or file.endswith('_codehash.csv')):
            # Get the original name without extension
            original_name, ext = os.path.splitext(file)
            # Create the new name
            new_name = f"{original_name}_codetardos{ext}"
            # Rename the file
            os.rename(file, new_name)
            print(f"Renamed: {file} -> {new_name}")
        elif file.endswith('_codetardos.csv') or file.endswith('_codehash.csv'):
            print(f"Skipped: {file} already has a code suffix.")


if __name__ == "__main__":
    rename_csv_files()
