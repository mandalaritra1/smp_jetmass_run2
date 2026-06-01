import os

# Define the logic for renaming
def determine_extension(filename):
    if any(key in filename for key in ["Uncertainty", "UncertaintySources"]):
        return ".junc.txt"
    elif any(key in filename for key in ["L1", "L2", "L3"]):
        return ".jec.txt"
    else:
        return None  # Skip unknowns

def rename_files_in_directory(directory="."):
    for file in os.listdir(directory):
        if file.endswith(".txt") and not (file.endswith(".jec.txt") or file.endswith(".junc.txt")):
            base = file[:-4]  # remove .txt
            new_ext = determine_extension(file)
            if new_ext:
                new_name = base + new_ext
                os.rename(os.path.join(directory, file), os.path.join(directory, new_name))
                print(f"Renamed: {file} → {new_name}")
            else:
                print(f"Skipped (unknown type): {file}")

if __name__ == "__main__":
    rename_files_in_directory(".")  # Change '.' to your target directory if needed

