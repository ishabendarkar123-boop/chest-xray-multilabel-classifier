"""
DenseNet-121 model definition for multi-label chest X-ray classification.
"""

import torch.nn as nn
from torchvision import models
from torchvision.models import DenseNet121_Weights


def build_densenet121(num_classes=14, pretrained=True):
    """
    Loads DenseNet-121 pretrained on ImageNet and replaces the final
    classifier layer with one producing `num_classes` outputs.
    No sigmoid here -- BCEWithLogitsLoss expects raw logits, not probabilities.
    """
    weights = DenseNet121_Weights.DEFAULT if pretrained else None
    model = models.densenet121(weights=weights)

    num_features = model.classifier.in_features
    model.classifier = nn.Linear(num_features, num_classes)

    return model

def build_resnet50(num_classes=14, pretrained=True):
    """
    Loads ResNet-50 pretrained on ImageNet, replaces the final FC layer
    with one producing `num_classes` outputs. Same logic as DenseNet-121:
    no sigmoid here, BCEWithLogitsLoss handles that internally.
    """
    from torchvision.models import ResNet50_Weights
    weights = ResNet50_Weights.DEFAULT if pretrained else None
    model = models.resnet50(weights=weights)

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    return model