import argparse
import os
import re
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
    return image, transform(image).unsqueeze(0)


def reconstruct_colored_mask(probs, class_colors, threshold=0.5):
    n_classes, h, w = probs.shape
    binary_masks = (probs > threshold).float().numpy()

    colored = np.zeros((h, w, 3), dtype=np.float32)
    for c in range(n_classes):
        color = np.array(class_colors[c], dtype=np.float32)
        colored += binary_masks[c][..., None] * color

    return np.clip(colored, 0, 255).astype(np.uint8)


def parse_level_from_filename(model_path):
    """Extracts the level number from filenames like 'best_model_level-1.pth'."""
    match = re.search(r'level(-?\d+)', os.path.basename(model_path))
    if match is None:
        raise ValueError(
            f"Could not infer 'level' from filename '{model_path}'. "
            f"Pass it explicitly with --levels."
        )
    return int(match.group(1))


def load_model(model_path, level, n_classes, device):
    model = UNet(level=level, n_classes=n_classes)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Run inference with multiple UNet models and compare predicted masks.")
    parser.add_argument('--image', required=True, help="Path to the input image")
    parser.add_argument('--models', required=True, nargs='+', help="Paths to one or more trained model .pth files")
    parser.add_argument('--levels', type=int, nargs='+', default=None,
                         help="Level for each model in --models, in the same order. "
                              "If omitted, levels are parsed from filenames like 'best_model_level-1.pth'.")
    parser.add_argument('--configs', default='./configs.yaml', help="Path to configs.yaml")
    parser.add_argument('--mask_suffix', default='___fuse.png', help="Suffix used for ground-truth mask files")
    parser.add_argument('--threshold', type=float, default=0.5, help="Threshold for binarizing each channel")
    args = parser.parse_args()

    configs = load_configs(args.configs)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    class_colors = load_class_colors(configs['dataset_path'])
    n_classes = len(class_colors)

    if args.levels is not None:
        if len(args.levels) != len(args.models):
            raise ValueError("--levels must have the same number of entries as --models")
        levels = args.levels
    else:
        levels = [parse_level_from_filename(p) for p in args.models]

    # Load original image + its corresponding ground-truth mask
    original_image, input_tensor = preprocess_image(args.image)
    input_tensor = input_tensor.to(device)

    gt_mask_path = args.image + args.mask_suffix
    gt_mask = Image.open(gt_mask_path).convert("RGB").resize((576, 576), Image.NEAREST)

    # Run every model on the same image
    predicted_masks = []
    for model_path, level in zip(args.models, levels):
        model = load_model(model_path, level, n_classes, device)
        with torch.no_grad():
            logits = model(input_tensor)
            probs = torch.sigmoid(logits).squeeze(0).cpu()
        colored_mask = reconstruct_colored_mask(probs, class_colors, threshold=args.threshold)
        predicted_masks.append((os.path.basename(model_path), colored_mask))

    # Display: original image, ground-truth mask, then one panel per model
    n_panels = 2 + len(predicted_masks)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))

    axes[0].imshow(original_image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    axes[1].imshow(gt_mask)
    axes[1].set_title("Ground Truth Mask")
    axes[1].axis('off')

    for ax, (name, mask) in zip(axes[2:], predicted_masks):
        ax.imshow(mask)
        ax.set_title(name)
        ax.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()