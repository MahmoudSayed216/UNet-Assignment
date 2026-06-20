import argparse
import os
import json

import torch
import numpy as np
import yaml
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model import UNet


def load_configs(configs_path):
    with open(configs_path, 'r') as f:
        return yaml.safe_load(f)


def load_class_colors(dataset_path):
    cache_path = os.path.join(dataset_path, "color_map.json")
    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"Could not find {cache_path}. Run training at least once so the "
            f"color map gets generated, or check your configs' dataset_path."
        )
    with open(cache_path, 'r') as f:
        colors = json.load(f)
    return [tuple(c) for c in colors]


def preprocess_image(image_path, size=576):
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert("RGB")
    return image, transform(image).unsqueeze(0)   # add batch dim


def reconstruct_colored_mask(probs, class_colors, threshold=0.5):
    """
    probs: (n_classes, H, W) sigmoid probabilities
    class_colors: list of (R, G, B) tuples, one per channel
    Returns: (H, W, 3) uint8 image
    """
    n_classes, h, w = probs.shape
    binary_masks = (probs > threshold).float().numpy()   # (C, H, W)

    colored = np.zeros((h, w, 3), dtype=np.float32)
    for c in range(n_classes):
        color = np.array(class_colors[c], dtype=np.float32)
        colored += binary_masks[c][..., None] * color    # (H,W,1) * (3,) -> (H,W,3)

    # Clip in case overlapping channel predictions push a pixel's summed
    # color above 255 (happens if BCE predicts >1 class active at a pixel)
    colored = np.clip(colored, 0, 255).astype(np.uint8)
    return colored


def main():
    parser = argparse.ArgumentParser(description="Run UNet inference and display the predicted mask.")
    parser.add_argument('--image', required=True, help="Path to the input image")
    parser.add_argument('--model', required=True, help="Path to the trained model .pth file")
    parser.add_argument('--configs', default='./configs.yaml', help="Path to configs.yaml")
    parser.add_argument('--threshold', type=float, default=0.5, help="Threshold for binarizing each channel")
    args = parser.parse_args()

    configs = load_configs(args.configs)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    class_colors = load_class_colors(configs['dataset_path'])
    n_classes = len(class_colors)

    model = UNet(level=int(configs['level']), n_classes=n_classes)
    state_dict = torch.load(args.model, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    original_image, input_tensor = preprocess_image(args.image)
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.sigmoid(logits).squeeze(0).cpu()   # (n_classes, H, W)

    colored_mask = reconstruct_colored_mask(probs, class_colors, threshold=args.threshold)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(original_image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    axes[1].imshow(colored_mask)
    axes[1].set_title("Predicted Mask")
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()