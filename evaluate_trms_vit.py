import os
import torch
import timm
import torch.nn as nn
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ======================================================
# CONFIG
# ======================================================
DATA_DIR = "data"
MODEL_PATH = "best_trms_vit.pth"
NUM_CLASSES = 38
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================================================
# TEST TRANSFORMS
# ======================================================
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

test_dataset = datasets.ImageFolder(
    os.path.join(DATA_DIR, "test"),
    transform=test_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# ======================================================
# MODEL ARCHITECTURE
# ======================================================
class CNNTokenExtractor(nn.Module):
    def __init__(self, embed_dim=768):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((14, 14))
        self.proj = nn.Linear(256, embed_dim)

    def forward(self, x):
        f = self.cnn(x)
        f = self.pool(f)
        B, C, H, W = f.shape
        f = f.view(B, C, -1).permute(0, 2, 1)
        tokens = self.proj(f)
        return tokens

class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim=768, num_heads=8):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x2 = self.norm1(x)
        attn_out, _ = self.attn(x2, x2, x2)
        x = x + attn_out
        x = self.norm2(x)
        return x

class TRMS_ViT(nn.Module):
    def __init__(self, num_classes=38):
        super().__init__()
        self.vit = timm.create_model("vit_base_patch16_224", pretrained=False)
        embed_dim = self.vit.embed_dim
        self.vit.head = nn.Identity()
        self.cnn_tokens = CNNTokenExtractor(embed_dim)
        self.cross_attn = CrossAttentionBlock(embed_dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 512),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        vit_tokens = self.vit.forward_features(x)
        cnn_tokens = self.cnn_tokens(x)
        all_tokens = torch.cat([vit_tokens, cnn_tokens], dim=1)
        refined_tokens = self.cross_attn(all_tokens)
        cls_token = refined_tokens[:, 0]
        output = self.classifier(cls_token)
        return output

# ======================================================
# LOAD MODEL
# ======================================================
model = TRMS_ViT(NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ======================================================
# EVALUATION LOOP
# ======================================================
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# ======================================================
# METRICS & PER-CLASS ACCURACY
# ======================================================
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# Core Metrics
accuracy = accuracy_score(all_labels, all_preds)
cm = confusion_matrix(all_labels, all_preds)

# Per-Class Accuracy Calculation
# Logic: Correct Predictions for Class i / Total Actual Samples of Class i
per_class_total = cm.sum(axis=1)
per_class_correct = cm.diagonal()
# Avoid division by zero if a class has no samples
class_accuracies = np.divide(per_class_correct, per_class_total, 
                             out=np.zeros_like(per_class_correct, dtype=float), 
                             where=per_class_total != 0)

print("\n" + "="*30)
print("     OVERALL TEST METRICS")
print("="*30)
print(f"Total Accuracy: {accuracy:.4f}")

print("\n" + "="*30)
print("     ACCURACY PER CLASS")
print("="*30)
for i, class_name in enumerate(test_dataset.classes):
    print(f"{class_name:<35}: {class_accuracies[i]:.4f}")

print("\n" + "="*30)
print("   CLASSIFICATION REPORT")
print("="*30)
print(classification_report(all_labels, all_preds, target_names=test_dataset.classes))

# ======================================================
# CONFUSION MATRIX VISUALIZATION
# ======================================================
plt.figure(figsize=(14, 12))
sns.heatmap(cm, annot=False, cmap="Blues", 
            xticklabels=test_dataset.classes, 
            yticklabels=test_dataset.classes)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix: Per-Class Performance")
plt.tight_layout()
plt.show()