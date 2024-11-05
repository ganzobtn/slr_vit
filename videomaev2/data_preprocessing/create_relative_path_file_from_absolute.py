import os
import argparse

# Function to process a single file
def process_file(input_file, output_file):
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            # Split the line into path and label
            parts = line.strip().split()
            # Extract the filename and directory
            relative_path = os.path.join(os.path.basename(os.path.dirname(parts[0])), os.path.basename(parts[0]))
            # Write the new line to the output file
            outfile.write(f"{relative_path} {parts[1]}\n")

def main(input_folder, destination_folder):
    # Paths to the input files
    train_file = os.path.join(input_folder, 'train.csv')
    test_file = os.path.join(input_folder, 'test.csv')
    val_file = os.path.join(input_folder, 'val.csv')

    # Paths to the output files
    train_output_file = os.path.join(destination_folder, 'train.csv')
    test_output_file = os.path.join(destination_folder, 'test.csv')
    val_output_file = os.path.join(destination_folder, 'val.csv')

    # Process each file
    process_file(train_file, train_output_file)
    process_file(test_file, test_output_file)
    process_file(val_file, val_output_file)

    print("Files processed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process CSV files to remove absolute paths.")
    parser.add_argument('input_folder', type=str, help="Folder containing the input CSV files.")
    parser.add_argument('destination_folder', type=str, help="Folder to save the processed CSV files.")

    args = parser.parse_args()
    main(args.input_folder, args.destination_folder)
