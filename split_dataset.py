import os
import shutil
import random

# ==============================
# Configuration
# ==============================

SOURCE_DIR = "Prj3_testing_data"
OUTPUT_DIR = "data"

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42

# ==============================

random.seed(RANDOM_SEED)


def create_directory_structure():
    for split in ["train", "val", "test"]:
        split_path = os.path.join(OUTPUT_DIR, split)
        os.makedirs(split_path, exist_ok=True)


def split_dataset():
    create_directory_structure()

    for class_name in os.listdir(SOURCE_DIR):
        class_path = os.path.join(SOURCE_DIR, class_name)

        if not os.path.isdir(class_path):
            continue

        images = [
            img for img in os.listdir(class_path)
            if img.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        random.shuffle(images)

        total = len(images)

        train_end = int(total * TRAIN_RATIO)
        val_end = train_end + int(total * VAL_RATIO)

        train_images = images[:train_end]
        val_images = images[train_end:val_end]
        test_images = images[val_end:]

        for split_name, split_images in zip(
            ["train", "val", "test"],
            [train_images, val_images, test_images]
        ):
            split_class_dir = os.path.join(OUTPUT_DIR, split_name, class_name)
            os.makedirs(split_class_dir, exist_ok=True)

            for img in split_images:
                src_path = os.path.join(class_path, img)
                dst_path = os.path.join(split_class_dir, img)
                shutil.copy2(src_path, dst_path)

        print(f"\nClass: {class_name}")
        print(f"Train: {len(train_images)}")
        print(f"Val: {len(val_images)}")
        print(f"Test: {len(test_images)}")

    print("\n✅ Dataset splitting completed successfully.")


if __name__ == "__main__":
    split_dataset()
