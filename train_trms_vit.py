import os
import torch
import timm
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

# ======================================================
# CONFIGURATION
# ======================================================

DATA_DIR = "data"
NUM_CLASSES = 38
BATCH_SIZE = 16
EPOCHS = 20
LR_BACKBONE = 2e-5
LR_HEAD = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATIENCE = 5

# ======================================================
# DATA AUGMENTATION
# ======================================================

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"),
                                     transform=train_transform)

val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "val"),
                                   transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=0)

val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                        shuffle=False, num_workers=0)

# ======================================================
# CNN TOKEN EXTRACTOR
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

        self.pool = nn.AdaptiveAvgPool2d((14, 14))  # control token count

        self.proj = nn.Linear(256, embed_dim)

    def forward(self, x):
        f = self.cnn(x)                 # (B,256,H,W)
        f = self.pool(f)                # (B,256,14,14)
        B, C, H, W = f.shape
        f = f.view(B, C, -1).permute(0, 2, 1)  # (B,196,256)
        tokens = self.proj(f)           # (B,196,768)
        return tokens


# ======================================================
# CROSS ATTENTION REFINEMENT
# ======================================================

class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim=768, num_heads=8):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim,
                                          num_heads,
                                          batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x2 = self.norm1(x)
        attn_out, _ = self.attn(x2, x2, x2)
        x = x + attn_out
        x = self.norm2(x)
        return x


# ======================================================
# TRMS-ViT MODEL
# ======================================================

class TRMS_ViT(nn.Module):
    def __init__(self, num_classes=38):
        super().__init__()

        self.vit = timm.create_model(
            "vit_base_patch16_224",
            pretrained=True
        )

        embed_dim = self.vit.embed_dim

        # Remove original classifier
        self.vit.head = nn.Identity()

        # CNN Token Generator
        self.cnn_tokens = CNNTokenExtractor(embed_dim)

        # Cross Attention Refinement
        self.cross_attn = CrossAttentionBlock(embed_dim)

        # Classification Head
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 512),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):

        # ViT token features (includes CLS)
        vit_tokens = self.vit.forward_features(x)   # (B,N,768)

        # CNN tokens
        cnn_tokens = self.cnn_tokens(x)             # (B,196,768)

        # Concatenate tokens
        all_tokens = torch.cat([vit_tokens, cnn_tokens], dim=1)

        # Cross-attention refinement
        refined_tokens = self.cross_attn(all_tokens)

        # CLS token
        cls_token = refined_tokens[:, 0]

        output = self.classifier(cls_token)

        return output


model = TRMS_ViT(NUM_CLASSES).to(DEVICE)

# ======================================================
# LOSS & OPTIMIZER
# ======================================================

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

optimizer = optim.AdamW([
    {"params": model.cnn_tokens.parameters(), "lr": LR_HEAD},
    {"params": model.cross_attn.parameters(), "lr": LR_HEAD},
    {"params": model.classifier.parameters(), "lr": LR_HEAD},
    {"params": model.vit.parameters(), "lr": LR_BACKBONE}
], weight_decay=0.01)

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                 T_max=EPOCHS)

# ======================================================
# TRAINING LOOP
# ======================================================

best_val_acc = 0
early_counter = 0

for epoch in range(EPOCHS):

    # -------- TRAIN --------
    model.train()
    train_loss = 0
    train_preds = []
    train_labels = []

    for images, labels in tqdm(train_loader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        train_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)

        train_preds.extend(preds.cpu().numpy())
        train_labels.extend(labels.cpu().numpy())

    train_acc = accuracy_score(train_labels, train_preds)

    # -------- VALIDATION --------
    model.eval()
    val_preds = []
    val_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    val_acc = accuracy_score(val_labels, val_preds)

    scheduler.step()

    print(f"\nEpoch [{epoch+1}/{EPOCHS}]")
    print(f"Train Loss: {train_loss/len(train_loader):.4f}")
    print(f"Train Acc : {train_acc:.4f}")
    print(f"Val Acc   : {val_acc:.4f}")

    # Save best
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_trms_vit.pth")
        early_counter = 0
        print("✅ Best Model Saved")
    else:
        early_counter += 1

    if early_counter >= PATIENCE:
        print("🛑 Early Stopping Triggered")
        break

print("\nTraining Completed")