import pickle
import os

def load_and_print_pickle(file_path):
    """Load and print the contents of a pickle file."""
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            print(f"\nContents of {os.path.basename(file_path)}:")
            print(data)
    else:
        print(f"File not found: {file_path}")

# Paths to the pickle files
result_path = r"results\week_4\QSD1_W4\method_sift\result.pkl"
gt_path = r"data\qsd1_w4\gt_corresps.pkl"

# Print contents of both files
load_and_print_pickle(result_path)
load_and_print_pickle(gt_path)