import os
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EfficientNet.from_name("efficientnet-b0")
model._fc = torch.nn.Linear(model._fc.in_features, 2)
model.load_state_dict(torch.load("pneumonia_model.pth", map_location=device))
model.eval().to(device)

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ✅ Update image path
image_path = "test_images/sample2.jpeg"
image = Image.open(image_path).convert("RGB")
resized_image = image.resize((224, 224))  # resize for correct overlay
input_tensor = transform(resized_image).unsqueeze(0).to(device)

# Grad-CAM++ setup
target_layers = [model._blocks[-1]]
cam = GradCAMPlusPlus(model=model, target_layers=target_layers)

# Run Grad-CAM++
grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(1)])[0]
rgb_img = np.array(resized_image).astype(np.float32) / 255.0  # must match size
visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

# Save the visualization
cv2.imwrite("gradcam_output.jpg", visualization)
print("✅ Grad-CAM++ output saved as gradcam_output.jpg")
