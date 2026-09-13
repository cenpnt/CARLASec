"""Shared model definition for the XAI adversarial-detection pipeline.

The ImageNet normalisation is folded INTO the model as a buffer so that the
network's external interface is plain [0,1] pixel space. ART and the attribution
methods can then operate on real images with clip_values=(0,1), and the
perturbation budget epsilon keeps its natural interpretation in image units.
"""
import torch
import torch.nn as nn
import timm

NUM_CLASSES = 43
IMG_SIZE = 96
BACKBONE = "resnet34"

# ImageNet statistics, since the backbone is pretrained.
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


class SignNet(nn.Module):
    """GTSRB classifier that accepts [0,1] images and normalises internally."""

    def __init__(self, backbone=BACKBONE, num_classes=NUM_CLASSES, pretrained=False):
        super().__init__()
        self.register_buffer("mean", torch.tensor(_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(_STD).view(1, 3, 1, 1))
        self.backbone = timm.create_model(
            backbone, pretrained=pretrained, num_classes=num_classes
        )

    def forward(self, x):
        return self.backbone((x - self.mean) / self.std)


def load_trained(path, device="cuda"):
    """Load a checkpoint saved by train_sota.py and return an eval-mode model."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = SignNet(
        backbone=ckpt.get("backbone", BACKBONE),
        num_classes=ckpt.get("num_classes", NUM_CLASSES),
        pretrained=False,
    )
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt
