
# Standard libraries
import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
import numpy as np
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import datetime
import pandas as pd
import gdown
import os

# Auto-download model from Google Drive if not present
if not os.path.exists('crack_model.pth'):
    with st.spinner('Loading model for the first time...'):
        gdown.download(
            'https://drive.google.com/uc?id=1s7VORk_MyFxSPFJ7cGGdSUoyitJaXtN6',
            'crack_model.pth', quiet=False
        )

# Page configuration
st.set_page_config(
    page_title="Intactly",
    page_icon="🏗️",
    layout="wide"
)

# Header
col_logo, col_title = st.columns([1, 6])
with col_logo:
    st.image("app400400forapp.png", width=80)
with col_title:
    st.title("Intactly")
    st.caption("Structural intelligence powered by AI. Upload a concrete or stone surface image to detect cracks, visualise damage, and receive a maintenance assessment.")
    st.caption("Built by Ishan")

st.divider()

# Load model
@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load('crack_model.pth', map_location=device))
    model.eval()
    model.to(device)
    return model, device

model, device = load_model()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Severity scoring
def calculate_severity(grayscale_cam, predicted):
    if predicted == 0:
        return 0, 0.0, ""
    mean_activation = grayscale_cam.mean()
    max_activation = grayscale_cam.max()
    raw_score = (mean_activation * 0.6 + max_activation * 0.4)
    severity = min(10, max(1, round(raw_score * 14)))
    crack_percentage = grayscale_cam.mean() * 100
    if severity <= 3:
        rec = "Minor surface crack detected. Routine monitoring is advised — schedule a visual inspection every 30 days and reassess if any progression is observed."
    elif severity <= 6:
        rec = "Moderate crack detected with potential for further deterioration. A professional assessment and repair is recommended within the next 3 months."
    else:
        rec = "Significant structural crack detected. An immediate professional inspection is strongly advised — further use of this structure may pose a safety risk."
    return severity, crack_percentage, rec

# Session history
if 'history' not in st.session_state:
    st.session_state.history = []

# Usage tip
st.info("📌 For best results, upload close-up images of concrete or stone surfaces.")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    img = Image.open(uploaded_file).convert('RGB')
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    img_tensor = transform(img_resized).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence = probs[0][1].item() * 100
        predicted = outputs.argmax(1).item()

    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=img_tensor, targets=[ClassifierOutputTarget(1)])[0]
    visualization = show_cam_on_image(img_array.astype(np.float32), grayscale_cam, use_rgb=True)

    severity, crack_pct, recommendation = calculate_severity(grayscale_cam, predicted)
    label = "Crack Detected" if predicted == 1 else "No Crack Detected"

    st.divider()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Result", label)
    col2.metric("Crack Confidence", f"{confidence:.1f}%")
    col3.metric("Severity", f"{severity}/10" if predicted == 1 else "N/A")
    col4.metric("Crack Coverage", f"{crack_pct:.1f}%" if predicted == 1 else "N/A")

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.image(img_resized, caption="Original Image", use_container_width=True)
    with col2:
        st.image(visualization, caption="AI Focus Map", use_container_width=True)

    st.divider()
    st.subheader("Maintenance Assessment")
    if predicted == 0:
        st.success("No structural damage detected. This surface appears to be in good condition. Continue routine visual inspections as standard practice.")
    elif severity <= 3:
        st.success(recommendation)
    elif severity <= 6:
        st.warning(recommendation)
    else:
        st.error(recommendation)

    st.session_state.history.append({
        "Time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "Result": label,
        "Confidence": f"{confidence:.1f}%",
        "Severity": f"{severity}/10" if predicted == 1 else "N/A",
        "Coverage": f"{crack_pct:.1f}%" if predicted == 1 else "N/A"
    })

if st.session_state.history:
    st.divider()
    st.subheader("Inspection History")
    st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True, hide_index=True)
