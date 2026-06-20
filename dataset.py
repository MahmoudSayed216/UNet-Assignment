import os
import re
import json
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def natural_sort_key(filename):
    return [int(chunk) if chunk.isdigit() else chunk.lower()
            for chunk in re.split(r'(\d+)', filename)]


def discover_colors(dataset_path, image_files, mask_suffix, cache_path):
    
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            return [tuple(c) for c in json.load(f)]

    unique_colors = set()
    for img_name in image_files:
        mask_path = os.path.join(dataset_path, img_name + mask_suffix)
        mask = np.array(Image.open(mask_path).convert("RGB"))
        pixels = mask.reshape(-1, 3)
        unique_colors.update(tuple(int(c) for c in color) for color in np.unique(pixels, axis=0))

    sorted_colors = sorted(unique_colors)   # deterministic ordering

    with open(cache_path, "w") as f:
        json.dump(sorted_colors, f)

    print(f"Discovered {len(sorted_colors)} unique colors -> cached at {cache_path}")
    return sorted_colors


class SegmentationDS(Dataset):
    def __init__(self, configs, split):
        super().__init__()
        self.n_classes = int(configs['n_classes'])
        self.path = configs['dataset_path']
        self.mask_suffix = "___fuse.png"
        self.save_suffix = "___save.png"

        files = sorted(os.listdir(path=self.path), key=natural_sort_key)
        files = [f for f in files
                 if not f.endswith(self.mask_suffix) and not f.endswith(self.save_suffix)]

        self.n_files = int(configs['train_ratio'] * len(files))
        
        self.split_files = None
        if split == 'train':
            self.split_files = files[:self.n_files]
        else:
            self.split_files = files[self.n_files:]
        print("DING")
        # Discover colors using the FULL file list (not just this split),
        # so train and test always agree on channel <-> color mapping.
        cache_path = configs["color_map_path"]
        self.class_colors = discover_colors(self.path, files, self.mask_suffix, cache_path)

        if len(self.class_colors) != self.n_classes:
            print(f"WARNING: found {len(self.class_colors)} unique colors, "
                  f"but n_classes={self.n_classes}. Inspect {cache_path}.")

        self.image_transform = transforms.Compose([
            transforms.Resize((576, 576)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
        ])
        for file in self.split_files:
            print(file)
    def __len__(self):
        return len(self.split_files)

    def __getitem__(self, index):
        img_name = self.split_files[index]
        mask_name = img_name + self.mask_suffix

        image = Image.open(os.path.join(self.path, img_name)).convert("RGB")
        mask = Image.open(os.path.join(self.path, mask_name)).convert("RGB")
        mask = mask.resize((576, 576), Image.NEAREST)   # NEAREST -> no new blended colors

        image_tensor = self.image_transform(image)

        mask_np = np.array(mask)  # (H, W, 3)
        channel_mask = np.zeros((self.n_classes, *mask_np.shape[:2]), dtype=np.float32)

        for class_idx, color in enumerate(self.class_colors):
            channel_mask[class_idx][np.all(mask_np == color, axis=-1)] = 1.0

        return image_tensor, torch.from_numpy(channel_mask)