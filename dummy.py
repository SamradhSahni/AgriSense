import json
from torchvision import datasets

train_dataset = datasets.ImageFolder("data/train")

with open("class_mapping.json", "w") as f:
    json.dump(train_dataset.classes, f)

print("Class mapping saved.")
