from dataset import SegmentationDS
from torch.utils.data import DataLoader
from model import UNet
import argparse
import tqdm
import yaml
import torch
import torch.nn as nn
from loss import BCEDiceLoss



def compute_iou(logits, targets, threshold=0.5, eps=1e-7):
    preds = (torch.sigmoid(logits) > threshold).float()
    intersection = (preds * targets).sum(dim=(0, 2, 3))
    union = preds.sum(dim=(0, 2, 3)) + targets.sum(dim=(0, 2, 3)) - intersection
    return (intersection + eps) / (union + eps)


def compute_dice(logits, targets, threshold=0.5, eps=1e-7):
    preds = (torch.sigmoid(logits) > threshold).float()
    intersection = (preds * targets).sum(dim=(0, 2, 3))
    denom = preds.sum(dim=(0, 2, 3)) + targets.sum(dim=(0, 2, 3))
    return (2 * intersection + eps) / (denom + eps)


def evaluate(model, loader, device, n_classes):
    model.eval()
    total_iou = torch.zeros(n_classes, device=device)
    total_dice = torch.zeros(n_classes, device=device)
    n_batches = 0

    with torch.no_grad():
        for image, mask in loader:
            image, mask = image.to(device), mask.to(device)
            output = model(image)

            total_iou += compute_iou(output, mask)
            total_dice += compute_dice(output, mask)
            n_batches += 1

    return total_iou / n_batches, total_dice / n_batches


def train(configs, level):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_classes = configs['n_classes']

    train_dataset = SegmentationDS(configs, split='train')
    test_dataset = SegmentationDS(configs, split='test')

    train_loader = DataLoader(train_dataset, batch_size=int(configs['batch_size']), shuffle=True,
                               num_workers=6, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=int(configs['batch_size']), shuffle=True,
                               num_workers=6, pin_memory=True)

    model = UNet(level=level, n_classes=n_classes)

    n_gpus = torch.cuda.device_count()
    if n_gpus > 1:
        print(f"Using {n_gpus} GPUs via nn.DataParallel")
        model = nn.DataParallel(model)

    model = model.to(device)

    optim = torch.optim.Adam(params=model.parameters(), lr=float(configs['lr']))
    criterion = BCEDiceLoss()

    n_epochs = int(configs['epochs'])
    best_dice = 0.0
    best_iou = 0.0
    best_dice_per_class = None
    best_iou_per_class = None

    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0

        for i, (image, mask) in enumerate(train_loader):
            image, mask = image.to(device), mask.to(device)

            optim.zero_grad()
            output_mask = model(image)
            loss = criterion(output_mask, mask)
            loss.backward()
            optim.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)

        iou_per_class, dice_per_class = evaluate(model, test_loader, device, n_classes)
        mean_iou = iou_per_class.mean().item()
        mean_dice = dice_per_class.mean().item()

        print(f"Epoch [{epoch+1}/{n_epochs}] "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Test mIoU: {mean_iou:.4f} | Test mDice: {mean_dice:.4f}")

        if mean_iou > best_iou:
            best_iou = mean_iou
            best_iou_per_class = iou_per_class

        if mean_dice > best_dice:
            best_dice = mean_dice
            best_dice_per_class = dice_per_class
            state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(state_dict, f'best_model_level{level}.pth')

    print("\nTraining complete.")
    print(f"Best Test mDice: {best_dice:.4f}")
    print(f"Best Test mIoU:  {best_iou:.4f}")
    print(f"Per-class Dice at best epoch: {[round(v, 4) for v in best_dice_per_class.cpu().tolist()]}")
    print(f"Per-class IoU at best epoch:  {[round(v, 4) for v in best_iou_per_class.cpu().tolist()]}")

def load_configs(configs_path):
    file = open(configs_path, mode='r')
    configs = yaml.safe_load(file)
    file.close()
    return configs


def main():
    parser = argparse.ArgumentParser(description="Train the UNet segmentation model.")
    parser.add_argument('--configs', type=str, default='/kaggle/working/UNet-Assignment/configs.yaml',
                         help="Path to the YAML configs file")
    parser.add_argument('--level', type=int, required=True,
                         help="Skip-connection level: -1 (all), 1, or 2")
    args = parser.parse_args()

    configs = load_configs(args.configs)
    print("configs: ", configs)
    print("level: ", args.level)

    train(configs, args.level)


if __name__ == "__main__":
    main()