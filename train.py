import argparse
import os
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import json
from ralftrainer import RALFTrainer
from ffdataset import ForgeryFaceDataset
from utils import fixed_seed
from catalyst.data import BatchBalanceClassSampler


def train_arg():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--dataset", type=str, choices=["FF++", "CelebDF", "DFDC"], default="FF++",
                        help="Dataset to train on.")
    parser.add_argument("--manipulation", type=str, choices=["all", "Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"], default="all",
                        help="Type of manipulation to include in the dataset.")
    parser.add_argument("--quality", type=str, choices=['raw', 'c23', 'c40'], default='c23',
                        help="Quality level of the images.")
    parser.add_argument("--image_size", type=int, default=256,
                        help="Size of the input images.")
    parser.add_argument("--norm", action="store_true",
                        help="Apply normalization to the input images.")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for training.")
    parser.add_argument("--num_epoch", type=int, default=100,
                        help="Number of training epochs.")
    parser.add_argument("--val_epoch", type=int, default=5,
                        help="Frequency of validation during training.")
    parser.add_argument("--lr", type=float, default=0.0001,
                        help="Learning rate for the optimizer.")
    parser.add_argument("--log_dir", type=Path, required=True,
                        help="Directory to save logs and checkpoints.")
    parser.add_argument("--save_best", action="store_true",
                        help="Save the best weight based on validation performance.")
    parser.add_argument("--ckpt_epoch", type=int, default=5,
                        help="Frequency of saving checkpoints during training.")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of worker processes for data loading.")
    parser.add_argument("--seed", type=int,
                        help="Random seed for reproducibility.")

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = train_arg()
    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir = str(args.log_dir)
    with open(os.path.join(args.log_dir, "args.json"), 'w') as f:
        json.dump(vars(args) , f, indent=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device} for training.")

    if args.seed is not None:
        fixed_seed(args.seed)
    
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) if args.norm else nn.Identity(),
    ])
    
    train_set = ForgeryFaceDataset(args.dataset, mode="train", manipulation=args.manipulation, quality=args.quality, transform=train_transform)

    # To perform contrastive loss
    train_batch_sampler = BatchBalanceClassSampler(train_set.label, num_classes=2, num_samples=args.batch_size // 2)
    train_loader = DataLoader(train_set, batch_sampler=train_batch_sampler, num_workers=args.num_workers, pin_memory=True)

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((args.image_size, args.image_size)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) if args.norm else nn.Identity(),
    ])
    val_set = ForgeryFaceDataset(args.dataset, mode="val", manipulation=args.manipulation, quality=args.quality, transform=val_transform)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    trainer = RALFTrainer(
        log_dir=args.log_dir,
        lr=args.lr,
        device=device
    )
    model = trainer.fit(args.num_epoch, train_loader, val_loader, args.val_epoch, args.ckpt_epoch, args.save_best)
    