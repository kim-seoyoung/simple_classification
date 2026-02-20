import torch
import torch.nn as nn
from .efficientnet_v2 import get_efficientnet_v2
from .convnext import convnext_tiny, convnext_small, convnext_base

class BirdDroneUAVClassifier(nn.Module):
    """
    Classifier for Bird, Drone, and UAV images.
    Supports EfficientNetV2 and ConvNeXt backbones.
    """
    def __init__(self, model_type='convnext_tiny', pretrained=True, num_classes=3, dropout_rate=0.3):
        super(BirdDroneUAVClassifier, self).__init__()
        
        # 1. Backbone Selection
        if 'efficientnet_v2' in model_type:
            # EfficientNetV2 returns a model with a head. 
            # We initialize it with nclass=0 to get the feature extractor or replace the head.
            self.backbone = get_efficientnet_v2(
                model_name=model_type, 
                pretrained=pretrained, 
                nclass=0 # Returns Identity for the classifier part
            )
            self.num_features = self.backbone.out_channels
            
        elif 'convnext' in model_type:
            if model_type == 'convnext_tiny':
                self.backbone = convnext_tiny(pretrained=pretrained)
            elif model_type == 'convnext_small':
                self.backbone = convnext_small(pretrained=pretrained)
            elif model_type == 'convnext_base':
                self.backbone = convnext_base(pretrained=pretrained)
            else:
                raise ValueError(f"Unsupported ConvNeXt variant: {model_type}")
            
            # ConvNeXt's head is a Linear layer, we replace it with Identity to extract features
            self.num_features = self.backbone.head.in_features
            self.backbone.head = nn.Identity()
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        # 2. Custom Classification Head
        # Designed for Bird, Drone, UAV classification with Dropout for regularization
        self.classifier_head = nn.Sequential(
            nn.BatchNorm1d(self.num_features),
            nn.Linear(self.num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # Extract features from the backbone
        features = self.backbone(x)
        # Pass through the custom classification head
        logits = self.classifier_head(features)
        return logits
