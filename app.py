import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# -------------------------
# CONFIG
# -------------------------
st.set_page_config(page_title="Waste Classifier", layout="centered")

CLASS_NAMES = ['cardboard','glass','metal','paper','plastic','trash']

DEVICE = torch.device("cpu")

# -------------------------
# LOAD MODEL
# -------------------------
@st.cache_resource
def load_model():
    model = models.efficientnet_b0(pretrained=True)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, len(CLASS_NAMES))

    model.load_state_dict(torch.load("Best_Waste_Model.pth", map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    return model

model = load_model()

# -------------------------
# TRANSFORM
# -------------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# -------------------------
# COLOR + SUGGESTIONS
# -------------------------
color_map = {
    "cardboard": "🟤",
    "glass": "🟢",
    "metal": "⚙️",
    "paper": "📄",
    "plastic": "🔵",
    "trash": "🗑️"
}

suggestions = {
    "cardboard": "Recycle in paper/cardboard bin ♻️",
    "glass": "Dispose in glass recycling ♻️",
    "metal": "Send to metal recycling ♻️",
    "paper": "Recycle as paper waste ♻️",
    "plastic": "Recycle if clean ♻️",
    "trash": "Dispose in general waste 🗑️"
}

# -------------------------
# UI
# -------------------------
st.title("♻️ Waste Classification System")
st.markdown("Upload an image to classify waste and get disposal guidance")

uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg","png","jpeg"])

# -------------------------
# PREDICTION
# -------------------------
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width="stretch")

    img = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)[0]
        pred = torch.argmax(probs).item()

    confidence = probs[pred].item()
    predicted_class = CLASS_NAMES[pred]

    # -------------------------
    # MAIN RESULT CARD
    # -------------------------
    st.markdown(f"""
    ## ♻️ Prediction Result
    ### {color_map[predicted_class]} {predicted_class.upper()}
    **Confidence:** {confidence*100:.2f}%
    """)

    # 🎉 small animation
    if confidence > 0.8:
        st.balloons()

    # -------------------------
    # PROBABILITIES
    # -------------------------
    st.subheader("📊 Detailed Analysis")

    for i, cls in enumerate(CLASS_NAMES):
        emoji = color_map[cls]
        prob = probs[i].item() * 100

        st.markdown(f"{emoji} **{cls.upper()}** — {prob:.2f}%")
        st.progress(float(probs[i]))

    # -------------------------
    # SUGGESTION
    # -------------------------
    st.info(f"💡 Suggestion: {suggestions[predicted_class]}")