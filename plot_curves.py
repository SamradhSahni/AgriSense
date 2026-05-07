import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

import timm

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
# CONFIG  (keep identical to your training script)
# ======================================================
DATA_DIR    = "data"
NUM_CLASSES = 38
BATCH_SIZE  = 16
EPOCHS      = 20
LR_BACKBONE = 2e-5
LR_HEAD     = 1e-4
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATIENCE    = 5

SAVE_DIR    = "plots"          # folder where the two PNGs are saved
os.makedirs(SAVE_DIR, exist_ok=True)

# ======================================================
# DATA LOADERS  (identical to training script)
# ======================================================
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"),
                                     transform=train_transform)
val_dataset   = datasets.ImageFolder(os.path.join(DATA_DIR, "val"),
                                     transform=val_transform)
train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                           shuffle=True, num_workers=0)
val_loader    = DataLoader(val_dataset,   batch_size=BATCH_SIZE,
                           shuffle=False, num_workers=0)

# ======================================================
# MODEL, LOSS, OPTIMISER  (identical to training script)
# ======================================================
model     = TRMS_ViT(NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW([
    {"params": model.cnn_tokens.parameters(), "lr": LR_HEAD},
    {"params": model.cross_attn.parameters(), "lr": LR_HEAD},
    {"params": model.classifier.parameters(), "lr": LR_HEAD},
    {"params": model.vit.parameters(),        "lr": LR_BACKBONE}
], weight_decay=0.01)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ======================================================
# HISTORY COLLECTORS
# ======================================================
history = {
    "train_loss": [],
    "val_loss":   [],
    "train_acc":  [],
    "val_acc":    [],
}

best_val_acc  = 0.0
early_counter = 0

# ======================================================
# TRAINING LOOP WITH HISTORY LOGGING
# ======================================================
for epoch in range(EPOCHS):

    # ── TRAIN ────────────────────────────────────────
    model.train()
    running_loss  = 0.0
    train_preds, train_labels = [], []

    for images, labels in tqdm(train_loader,
                               desc=f"Epoch {epoch+1}/{EPOCHS} [train]"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        running_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        train_preds.extend(preds.cpu().numpy())
        train_labels.extend(labels.cpu().numpy())

    train_acc  = accuracy_score(train_labels, train_preds)
    epoch_loss = running_loss / len(train_loader)

    # ── VALIDATION ───────────────────────────────────
    model.eval()
    val_loss_sum = 0.0
    val_preds, val_labels_list = [], []

    with torch.no_grad():
        for images, labels in tqdm(val_loader,
                                   desc=f"Epoch {epoch+1}/{EPOCHS} [val]  "):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss    = criterion(outputs, labels)
            val_loss_sum += loss.item()
            preds = torch.argmax(outputs, dim=1)
            val_preds.extend(preds.cpu().numpy())
            val_labels_list.extend(labels.cpu().numpy())

    val_acc  = accuracy_score(val_labels_list, val_preds)
    val_loss = val_loss_sum / len(val_loader)

    scheduler.step()

    # ── RECORD ───────────────────────────────────────
    history["train_loss"].append(epoch_loss)
    history["val_loss"].append(val_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)

    print(f"\nEpoch [{epoch+1}/{EPOCHS}]  "
          f"Train Loss: {epoch_loss:.4f}  Train Acc: {train_acc:.4f}  "
          f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.4f}")

    # ── SAVE BEST / EARLY STOP ───────────────────────
    if val_acc > best_val_acc:
        best_val_acc  = val_acc
        early_counter = 0
        torch.save(model.state_dict(), "best_trms_vit.pth")
        print("  ✅ Best model saved")
    else:
        early_counter += 1

    if early_counter >= PATIENCE:
        print("  🛑 Early stopping triggered")
        break

# ======================================================
# PLOTTING
# ======================================================
epochs_ran = range(1, len(history["train_acc"]) + 1)

# ── shared style ─────────────────────────────────────
STYLE = {
    "figure.dpi":        300,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.linestyle":    "--",
    "grid.alpha":        0.4,
    "font.family":       "DejaVu Sans",
    "font.size":         11,
}
plt.rcParams.update(STYLE)

TRAIN_COLOR = "#4C72B0"   # blue
VAL_COLOR   = "#DD8452"   # orange

# ── Fig 4a : Accuracy ────────────────────────────────
fig_acc, ax_acc = plt.subplots(figsize=(7, 4.5))

ax_acc.plot(epochs_ran, history["train_acc"],
            color=TRAIN_COLOR, linewidth=2.0,
            marker="o", markersize=4, label="Train accuracy")
ax_acc.plot(epochs_ran, history["val_acc"],
            color=VAL_COLOR,   linewidth=2.0,
            marker="s", markersize=4, linestyle="--", label="Val accuracy")

# annotate best val acc
best_epoch = history["val_acc"].index(max(history["val_acc"])) + 1
best_acc   = max(history["val_acc"])
ax_acc.annotate(
    f"Best val: {best_acc:.4f}\n(epoch {best_epoch})",
    xy=(best_epoch, best_acc),
    xytext=(best_epoch + 0.6, best_acc - 0.04),
    fontsize=9,
    arrowprops=dict(arrowstyle="->", color="gray", lw=1.0),
    color="gray"
)

ax_acc.set_xlabel("Epoch")
ax_acc.set_ylabel("Accuracy")
ax_acc.set_title("Fig. 4a — Training vs. Validation Accuracy (TRMS-ViT)",
                 fontsize=11, pad=10)
ax_acc.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
ax_acc.set_ylim(0, 1.05)
ax_acc.legend(framealpha=0.7, fontsize=10)
fig_acc.tight_layout()
acc_path = os.path.join(SAVE_DIR, "fig4a_accuracy_curve.png")
fig_acc.savefig(acc_path, bbox_inches="tight")
print(f"\n✅ Accuracy curve saved  →  {acc_path}")

# ── Fig 4b : Loss ────────────────────────────────────
fig_loss, ax_loss = plt.subplots(figsize=(7, 4.5))

ax_loss.plot(epochs_ran, history["train_loss"],
             color=TRAIN_COLOR, linewidth=2.0,
             marker="o", markersize=4, label="Train loss")
ax_loss.plot(epochs_ran, history["val_loss"],
             color=VAL_COLOR,   linewidth=2.0,
             marker="s", markersize=4, linestyle="--", label="Val loss")

ax_loss.set_xlabel("Epoch")
ax_loss.set_ylabel("Cross-entropy loss")
ax_loss.set_title("Fig. 4b — Training vs. Validation Loss (TRMS-ViT)",
                  fontsize=11, pad=10)
ax_loss.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
ax_loss.legend(framealpha=0.7, fontsize=10)
fig_loss.tight_layout()
loss_path = os.path.join(SAVE_DIR, "fig4b_loss_curve.png")
fig_loss.savefig(loss_path, bbox_inches="tight")
print(f"✅ Loss curve saved      →  {loss_path}")

plt.close("all")
print("\n🎉 Both curves saved to the 'plots/' folder.")
print(f"   Best Val Accuracy : {best_val_acc:.4f}  (Epoch {best_epoch})")
