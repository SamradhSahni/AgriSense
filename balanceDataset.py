import os
import random

# Path to dataset
dataset_path = "Prj3_testing_data"

# Number of images to keep per class
MAX_IMAGES = 100

def limit_images_per_class(dataset_path, max_images=100):
    for folder in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder)

        if os.path.isdir(folder_path):

            # Get only image files
            images = [
                img for img in os.listdir(folder_path)
                if img.lower().endswith(('.jpg', '.jpeg', '.png'))
            ]

            total_images = len(images)

            print(f"\nProcessing: {folder}")
            print(f"Total Images Before: {total_images}")

            if total_images > max_images:
                # Randomly select images to delete
                images_to_delete = random.sample(images, total_images - max_images)

                for img in images_to_delete:
                    os.remove(os.path.join(folder_path, img))

                print(f"Removed {len(images_to_delete)} images")
                print(f"Total Images After: {max_images}")

            else:
                print("No removal needed (less than or equal to 100 images)")

# Run function
limit_images_per_class(dataset_path, MAX_IMAGES)

print("\nDataset balancing complete.")
