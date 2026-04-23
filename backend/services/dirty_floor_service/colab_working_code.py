import torch
import pandas as pd
import numpy as np
from transformers import ViTForImageClassification, ViTConfig
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

model_path = "/content/drive/MyDrive/McD/Kitchen_Cleaning_Frame_Classification/ViT_Frame_Classifier_11_02_2026.pth"

# 1️⃣ Recreate architecture exactly like training
config = ViTConfig.from_pretrained("google/vit-base-patch16-224")
config.num_labels = 2

model = ViTForImageClassification(config)

# 2️⃣ Load saved state dict
state_dict = torch.load(model_path, map_location=device)

# 3️⃣ FIX torch.compile() prefix issue
new_state_dict = {}
for k, v in state_dict.items():
    if k.startswith("_orig_mod."):
        new_key = k.replace("_orig_mod.", "")
    else:
        new_key = k
    new_state_dict[new_key] = v

# 4️⃣ Load corrected weights
model.load_state_dict(new_state_dict)

# 5️⃣ Move to device
model.to(device)
model.eval()

print("✅ Model loaded successfully without ANY errors!")


# -------------------------------------------------
# YOUR Training Preprocess (IMPORTANT)
# -------------------------------------------------
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])
# -------------------------------------------------
# Mask Polygons
# -------------------------------------------------
MASKS = [
    np.array([[476, 508], [441, 2], [653, -12], [688, 493]]),
    np.array([[547, 639], [506, 402], [766, 356], [807, 593]]),
    np.array([[-149, 710], [16, -71], [389, 7], [223, 789]]),
    np.array([[1104, 577], [759, 151], [1051, -85], [1396, 340]]),
    np.array([[729, 145], [729, -37], [914, -37], [914, 145]])
]

# -------------------------------------------------
# Apply Mask
# -------------------------------------------------
def apply_polygon_masks_pil(image, masks):
    masked_image = image.copy()
    draw = ImageDraw.Draw(masked_image)
    width, height = image.size

    for poly in masks:
        clipped_poly = []
        for x, y in poly:
            x = max(0, min(x, width - 1))
            y = max(0, min(y, height - 1))
            clipped_poly.append((x, y))
        draw.polygon(clipped_poly, fill=(0, 0, 0))

    return masked_image

# -------------------------------------------------
# Prediction Function
# -------------------------------------------------
def predict_and_show_image(path):
    # Load image
    img = Image.open(path).convert("RGB")

    # Apply mask
    masked_img = apply_polygon_masks_pil(img, MASKS)

    # Show masked image
    plt.figure(figsize=(8, 8))
    plt.imshow(masked_img)
    plt.axis('off')

    # Apply your training preprocess
    img_tensor = preprocess(masked_img).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        outputs = model(pixel_values=img_tensor)
        logits = outputs.logits
        pred = torch.argmax(logits, dim=1)

    classes = ['Clean', 'Dirty']
    prediction = classes[pred.item()]

    plt.title(f"Prediction: {prediction}", fontsize=16)
    plt.show()

    return prediction

# -------------------------------------------------
# Test Image
# -------------------------------------------------
path = "/content/drive/MyDrive/McD/Kitchen_Cleaning_Frame_Classification/Clean/frame_1_1000.jpg"

prediction = predict_and_show_image(path)
print(f"The image is predicted as: {prediction}")
