import os

# Path to dataset folder
dataset_path = "../Prj3_testing_data"

def clean_label(folder_name):
    """
    Convert folder name into human-readable format
    """
    # Step 1: Replace triple underscore (plant-disease separator)
    name = folder_name.replace("___", " - ")

    # Step 2: Replace remaining underscores with space
    name = name.replace("_", " ")

    return name.strip()

# Get all folder names
folder_names = [
    folder for folder in os.listdir(dataset_path)
    if os.path.isdir(os.path.join(dataset_path, folder))
]

print("Clean Class Labels:\n")

clean_labels = []

for folder in folder_names:
    readable_name = clean_label(folder)
    clean_labels.append(readable_name)
    print(readable_name)

print("\nTotal Classes:", len(clean_labels))
