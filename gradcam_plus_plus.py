import os
import torch
import numpy as np
import cv2
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
import matplotlib.pyplot as plt

from model_utils import CustomImageDataset
from augmentations import get_val_transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = EfficientNet.from_pretrained('efficientnet-b0')
model._fc = torch.nn.Linear(model._fc.in_features, 2)
model.load_state_dict(torch.load("pneumonia_model.pth", map_location=device))
model.to(device)
model.eval()

# Hook for Grad-CAM++
gradients = []
activations = []

def forward_hook(module, input, output):
    activations.append(output)

def backward_hook(module, grad_in, grad_out):
    gradients.append(grad_out[0])

# Register hooks to final conv layer
target_layer = model._blocks[-1]._depthwise_conv
target_layer.register_forward_hook(forward_hook)
target_layer.register_backward_hook(backward_hook)

# Load one image from validation set
val_dir = os.path.join("chest_xray", "val")
dataset = CustomImageDataset(val_dir, get_val_transforms())
image, label = dataset[0]
input_tensor = image.unsqueeze(0).to(device)

# Forward and backward pass
output = model(input_tensor)
class_idx = torch.argmax(output)
model.zero_grad()
output[0, class_idx].backward()

# Grad-CAM++ Calculation
grad = gradients[0][0].cpu().numpy()
act = activations[0][0].cpu().numpy()

weights = np.sum(grad, axis=(1, 2)) / (np.sum(grad ** 2, axis=(1, 2)) + 1e-8)
cam = np.sum(weights[:, np.newaxis, np.newaxis] * act, axis=0)
cam = np.maximum(cam, 0)
cam = cv2.resize(cam, (224, 224))
cam = cam - np.min(cam)
cam = cam / np.max(cam)

# Overlay
input_image = image.permute(1, 2, 0).cpu().numpy()
input_image = (input_image * 0.5 + 0.5)  # unnormalize
heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
overlay = cv2.addWeighted(np.uint8(255 * input_image), 0.6, heatmap, 0.4, 0)

# Show result
plt.imshow(overlay[..., ::-1])
plt.title(f"Predicted Class: {class_idx.item()}")
plt.axis("off")
plt.show()
