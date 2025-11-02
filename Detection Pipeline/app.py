# app.py
# Streamlit UI for "CT Scan Classification"
# Loads model from local directory using dropdown

import os
import numpy as np
from PIL import Image
import streamlit as st
import torch
import torch.nn.functional as F
import json
try:
    import torchvision.transforms as T
except Exception:
    T = None

# Minimal stub to allow torch.load to unpickle models that reference
# VERNetLogitAvg. When the original class definition is available
# (for example from the project's model code), replace this stub
# with the real implementation or import it from the correct module.
class VERNetLogitAvg(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        # This stub does not implement the real model layers. It's
        # only used to satisfy pickle during torch.load. The loaded
        # state_dict will still be applied to this object; if the
        # real class has different modules, loading may fail later.

    def forward(self, x):
        # Pass-through placeholder. Real model will override.
        return x

st.set_page_config(page_title="CT Scan Classification", page_icon="🫁", layout="centered")

# -----------------------------
# Config
# -----------------------------
MODEL_DIR = "models"  # Folder containing your .pt models
os.makedirs(MODEL_DIR, exist_ok=True)  # ensure folder exists

# -----------------------------
# Helpers
# -----------------------------
import torch
import torch.nn as nn
from torchvision.models import (
    vgg19_bn, VGG19_BN_Weights,
    efficientnet_b0, EfficientNet_B0_Weights,
    resnet101, ResNet101_Weights
)

def make_vgg19_bn(num_classes):
    m = vgg19_bn(weights=VGG19_BN_Weights.IMAGENET1K_V1)
    in_feats = m.classifier[-1].in_features
    m.classifier[-1] = nn.Linear(in_feats, num_classes)
    return m

def make_efficientnet_b0(num_classes):
    m = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_feats = m.classifier[1].in_features
    m.classifier[1] = nn.Linear(in_feats, num_classes)
    return m

def make_resnet101(num_classes):
    m = resnet101(weights=ResNet101_Weights.IMAGENET1K_V2)
    in_feats = m.fc.in_features
    m.fc = nn.Linear(in_feats, num_classes)
    return m

class VERNetLogitAvg(nn.Module):
    """
    Simple late-fusion ensemble: average logits from three backbones.
    """
    def __init__(self, num_classes):
        super().__init__()
        self.vgg = make_vgg19_bn(num_classes)
        self.eff = make_efficientnet_b0(num_classes)
        self.res = make_resnet101(num_classes)

    def forward(self, x):
        # Each backbone outputs [B, num_classes] logits
        y1 = self.vgg(x)
        y2 = self.eff(x)
        y3 = self.res(x)
        return (y1 + y2 + y3) / 3.0

# Instantiate ensemble

def load_model(model_path):
    """Load model from local path (with GPU support)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VERNetLogitAvg(7).to(device)
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.to(device).eval()
    return model, device

def preprocess_image(img: Image.Image, size=224, channels=1):
    """Resize and normalize image into tensor."""
    if channels == 1:
        img = img.convert("L")
    else:
        img = img.convert("RGB")

    img = img.resize((size, size))
    arr = np.array(img).astype(np.float32) / 255.0

    if channels == 1:
        tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    else:
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        # Apply ImageNet normalization by default for RGB to match eval transforms
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
        tensor = (tensor - mean) / std
    return tensor


def preprocess_with_torchvision(img: Image.Image, size=224, channels=3, norm='imagenet'):
    """Use torchvision transforms (if available) to preprocess image similar to training."""
    if T is None:
        raise RuntimeError("torchvision.transforms not available in this environment")

    if channels == 1:
        img = img.convert('L')
    else:
        img = img.convert('RGB')

    transforms = [T.Resize((size, size)), T.ToTensor()]
    if norm == 'imagenet' and channels == 3:
        transforms.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    elif norm == 'imagenet' and channels == 1:
        transforms.append(T.Normalize(mean=[0.485], std=[0.229]))

    tfm = T.Compose(transforms)
    tensor = tfm(img).unsqueeze(0)
    return tensor


def infer_and_debug(model, tensor, device):
    """Run model and return predicted index, confidence and full probs array (cpu numpy).

    This mirrors predict() but returns the full probability distribution for debugging.
    """
    tensor = tensor.to(device)
    with torch.no_grad():
        logits = model(tensor)
        if logits.ndim == 1:
            logits = logits.unsqueeze(0)
        if logits.dim() > 2:
            logits = logits.view(logits.size(0), logits.size(1), -1).mean(dim=2)
        probs = F.softmax(logits, dim=1)
        pred_idxs = probs.argmax(dim=1)
        pred_idx = int(pred_idxs[0].item())
        conf = float(probs[0, pred_idx].item())

        probs_np = probs[0].cpu().numpy().tolist()
        logits_shape = tuple(logits.shape)
    return pred_idx, conf, probs_np, logits_shape


def detect_model_input_channels(model):
    """Try to find the expected input channel count by inspecting Conv2d modules.

    Returns an int (channels) or None if not found.
    """
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            # in_channels attribute exists on Conv2d
            try:
                return int(m.in_channels)
            except Exception:
                continue
    return None

def predict(model, tensor, device):
    """Run inference and return predicted index + confidence."""
    tensor = tensor.to(device)
    with torch.no_grad():
        logits = model(tensor)
        if logits.ndim == 1:
            logits = logits.unsqueeze(0)
        # If model returns spatial maps (e.g., shape [B, C, H, W]),
        # collapse spatial dimensions using global average so we get
        # per-class logits of shape [B, C]. This also prevents
        # attempting to call .item() on multi-element tensors.
        if logits.dim() > 2:
            # collapse all dims after the channel dim
            logits = logits.view(logits.size(0), logits.size(1), -1).mean(dim=2)

        probs = F.softmax(logits, dim=1)
        pred_idxs = probs.argmax(dim=1)
        # pick first batch element's prediction and confidence
        pred_idx = int(pred_idxs[0].item())
        conf = float(probs[0, pred_idx].item())
    return pred_idx, conf


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🫁 CT Scan Classification")
st.caption("Select a local model and classify a CT scan as **Normal** or **Cancerous**.")

# Sidebar – model selection
with st.sidebar:
    st.header("Model Selection")
    model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pt")]
    if not model_files:
        st.warning(f"No .pt models found in `{MODEL_DIR}/`")
    else:
        selected_model = st.selectbox("Choose model", model_files)
        input_size = st.slider("Input size", 128, 512, 224, step=32)
        channels = st.radio("Image channels", [1, 3], index=1)
        use_torchvision = st.checkbox("Use torchvision transforms (if available)", value=(T is not None))
        st.markdown("---")
        st.subheader("Label mapping (optional)")
        st.write("Provide a JSON object mapping class ids (0..N-1) to labels, e.g. {\"0\": \"normal\", \"1\": \"cancerous\"}")
        label_map_text = st.text_area("Label map JSON", value='')
        try:
            label_map = json.loads(label_map_text) if label_map_text.strip() else None
        except Exception:
            label_map = None
        debug_mode = st.checkbox("Show debug outputs (probs, logits shape)", value=False)

# Upload CT image
img_file = st.file_uploader("Upload CT slice (.png, .jpg, .jpeg)", type=["png", "jpg", "jpeg"])
if img_file:
    st.image(img_file, caption="CT Scan", use_container_width=True)

# Run classification
if st.button("🔍 Classify"):
    if not model_files:
        st.error(f"No model found in `{MODEL_DIR}/`")
        st.stop()
    if not img_file:
        st.error("Please upload a CT image.")
        st.stop()

    try:
        model_path = os.path.join(MODEL_DIR, selected_model)
        model, device = load_model(model_path)
        st.success(f"✅ Loaded model: `{selected_model}`  |  Device: {device}")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

    try:
        img = Image.open(img_file)

        # Choose preprocessing pipeline
        try:
            if use_torchvision and T is not None:
                x = preprocess_with_torchvision(img, size=input_size, channels=channels if channels==3 else 1)
            else:
                x = preprocess_image(img, size=input_size, channels=channels)

            # Detect model input channels and adapt tensor if needed
            model_in_ch = detect_model_input_channels(model)
            tensor_ch = x.shape[1] if x.dim() >= 2 else None

            # Convert/repeat channels to match model expectation
            if model_in_ch is not None and tensor_ch is not None and model_in_ch != tensor_ch:
                if model_in_ch == 3 and tensor_ch == 1:
                    # repeat single channel to RGB
                    x = x.repeat(1, 3, 1, 1)
                elif model_in_ch == 1 and tensor_ch == 3:
                    # average RGB to single channel
                    x = x.mean(dim=1, keepdim=True)

            # Run inference
            if debug_mode:
                pred_idx, conf, probs_np, logits_shape = infer_and_debug(model, x, device)
            else:
                pred_idx, conf = predict(model, x, device)

        except Exception as e:
            st.error(f"Preprocess / inference error: {e}")
            st.stop()

        # If label_map provided, map index to label
        if label_map:
            # label_map keys may be strings; coerce
            lm = {int(k): v for k, v in label_map.items()}
            pred_label = lm.get(pred_idx, str(pred_idx))
        else:
            # default mapping: 0 -> normal
            pred_label = "normal" if pred_idx == 0 else "cancerous"

        # Rule: class 0 → Normal, anything else → Cancerous
        if pred_label == "normal":
            st.success(f"🟢 Prediction: **{pred_label}**  (Confidence: {conf:.2%})")
        else:
            st.error(f"🔴 Prediction: **{pred_label}**  (Confidence: {conf:.2%})")

        if debug_mode:
            st.markdown("**Debug info**")
            debug_info = {"logits_shape": logits_shape, "probs": probs_np}
            try:
                debug_info.update({"model_in_ch": model_in_ch, "tensor_ch": tensor_ch})
            except Exception:
                pass
            st.write(debug_info)

    except Exception as e:
        st.error(f"Inference error: {e}")
