import os
import torch
import timm
import torch.nn as nn
import streamlit as st
from torchvision import transforms
from PIL import Image
from dotenv import load_dotenv
from openai import OpenAI

# ==========================
# LOAD ENV VARIABLES
# ==========================
load_dotenv()

llm_client = OpenAI(
    api_key=os.getenv("NVIDIA_API_KEY"),
    base_url="https://integrate.api.nvidia.com/v1"
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================
# MODEL ARCHITECTURE
# ==========================

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


# ==========================
# LOAD TRAINED MODEL
# ==========================

NUM_CLASSES = 38
model = TRMS_ViT(NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load("best_trms_vit.pth", map_location=DEVICE))
model.eval()

# IMPORTANT:
# Replace this with your actual class order used in training
class_names = sorted(os.listdir("data/train"))

# ==========================
# IMAGE TRANSFORM
# ==========================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ==========================
# LLM FUNCTION
# ==========================

def generate_advisory(disease_name):

    prompt = f"""
    You are a professional agricultural plant pathologist.

    The detected disease is: {disease_name}

    Provide structured response with headings:

    1. Cause of Disease
    2. How It Spreads
    3. Chemical Treatment
    4. Organic Treatment
    5. Preventive Measures
    6. Farmer Practical Advice

    Keep language simple, clear, and farmer-friendly.
    Avoid unnecessary scientific jargon.
    """

    response = llm_client.chat.completions.create(
        model="meta/llama3-70b-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        max_tokens=700
    )

    return response.choices[0].message.content


# ==========================
# STREAMLIT UI
# ==========================

st.set_page_config(page_title="Crop Disease AI Advisor", layout="wide")

st.title("🌱 AI Crop Disease Detection & Advisory System")

uploaded_file = st.file_uploader("Upload Crop Leaf Image", type=["jpg", "png", "jpeg"])

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred = torch.max(probs, dim=1)

    disease_name = class_names[pred.item()]
    confidence_score = confidence.item() * 100

    st.success(f"🦠 Detected Disease: {disease_name}")
    st.info(f"Confidence: {confidence_score:.2f}%")

    if st.button("Generate AI Advisory"):

        with st.spinner("Generating expert advisory..."):

            advisory = generate_advisory(disease_name)

        st.markdown("## 🌿 Detailed Advisory")
        st.markdown(advisory)
