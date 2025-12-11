"""
Vision Backbones for ML4D

이 모듈은 다양한 pre-trained vision backbone을 제공합니다:
- EfficientNet (기존)
- RegNet
- BiFPN (EfficientDet 기반)
"""

from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class RegNetBackbone(nn.Module):
    """RegNet backbone with pre-trained weights.
    
    RegNet은 효율적인 이미지 분류를 위해 설계된 아키텍처입니다.
    timm 라이브러리를 통해 다양한 RegNet 변형을 사용할 수 있습니다.
    
    Available models:
        - regnetx_002, regnetx_004, regnetx_006, regnetx_008, regnetx_016, etc.
        - regnety_002, regnety_004, regnety_006, regnety_008, regnety_016, etc.
    """
    
    def __init__(
        self,
        model_name: str = "regnetx_004",
        pretrained: bool = True,
        features_only: bool = True,
        out_indices: Optional[List[int]] = None
    ):
        """
        Args:
            model_name: RegNet model name (e.g., 'regnetx_004', 'regnety_008')
            pretrained: Whether to use pre-trained weights
            features_only: If True, return feature maps instead of classification output
            out_indices: Which feature levels to return (e.g., [0, 1, 2, 3, 4])
                        If None, returns all levels
        """
        super().__init__()
        try:
            import timm
        except ImportError as e:
            raise RuntimeError("timm이 필요합니다: pip install timm") from e
        
        self.model_name = model_name
        self.pretrained = pretrained
        self.features_only = features_only
        
        # Create model
        if features_only:
            self.backbone = timm.create_model(
                model_name,
                pretrained=pretrained,
                features_only=True,
                out_indices=out_indices
            )
            # Get output channels for each feature level
            self.feature_info = self.backbone.feature_info
            if out_indices is None:
                self.out_ch = self.feature_info.channels()[-1]
            else:
                self.out_ch = self.feature_info.channels()[out_indices[-1]]
        else:
            self.backbone = timm.create_model(
                model_name,
                pretrained=pretrained
            )
            # For classification mode, get the last conv layer channels
            # This is approximate and may vary by model
            self.out_ch = 512  # Default, will be updated based on model
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, 3, H, W]
            
        Returns:
            If features_only: List of feature maps or single feature map
            If not features_only: Classification logits
        """
        if self.features_only:
            features = self.backbone(x)
            # If multiple levels, return the last one
            if isinstance(features, (list, tuple)):
                return features[-1]
            return features
        else:
            return self.backbone(x)


class BiFPN(nn.Module):
    """Bi-directional Feature Pyramid Network (BiFPN).
    
    BiFPN은 EfficientDet에서 사용되는 feature pyramid network로,
    다양한 스케일의 특징을 효과적으로 통합합니다.
    """
    
    def __init__(
        self,
        num_channels: int = 64,
        num_levels: int = 5,
        num_bifpn_layers: int = 3,
        epsilon: float = 1e-4
    ):
        """
        Args:
            num_channels: Number of channels in BiFPN
            num_levels: Number of feature pyramid levels
            num_bifpn_layers: Number of BiFPN layers to stack
            epsilon: Small value for numerical stability in weighted fusion
        """
        super().__init__()
        self.num_channels = num_channels
        self.num_levels = num_levels
        self.epsilon = epsilon
        
        # Stack multiple BiFPN layers
        self.layers = nn.ModuleList([
            BiFPNLayer(num_channels, num_levels, epsilon)
            for _ in range(num_bifpn_layers)
        ])
    
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            features: List of feature maps at different scales
                     [P3, P4, P5, P6, P7] where each is [B, C, H, W]
                     
        Returns:
            List of refined feature maps at the same scales
        """
        x = features
        for layer in self.layers:
            x = layer(x)
        return x


class BiFPNLayer(nn.Module):
    """Single BiFPN layer."""
    
    def __init__(self, num_channels: int, num_levels: int, epsilon: float = 1e-4):
        super().__init__()
        self.num_channels = num_channels
        self.num_levels = num_levels
        self.epsilon = epsilon
        
        # Top-down pathway (P7 -> P3)
        self.top_down_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(num_channels, num_channels, 3, padding=1, groups=num_channels),
                nn.BatchNorm2d(num_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(num_channels, num_channels, 1),
                nn.BatchNorm2d(num_channels)
            ) for _ in range(num_levels - 1)
        ])
        
        # Bottom-up pathway (P3 -> P7)
        self.bottom_up_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(num_channels, num_channels, 3, padding=1, groups=num_channels),
                nn.BatchNorm2d(num_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(num_channels, num_channels, 1),
                nn.BatchNorm2d(num_channels)
            ) for _ in range(num_levels - 1)
        ])
        
        # Weighted fusion weights (learnable)
        # Top-down: 2 inputs (current level + upsampled higher level)
        self.top_down_weights = nn.ParameterList([
            nn.Parameter(torch.ones(2) / 2) for _ in range(num_levels - 1)
        ])
        
        # Bottom-up: 2 inputs (current level + downsampled lower level)
        self.bottom_up_weights = nn.ParameterList([
            nn.Parameter(torch.ones(2) / 2) for _ in range(num_levels - 1)
        ])
    
    def _weighted_fusion(self, features: List[torch.Tensor], weights: torch.Tensor) -> torch.Tensor:
        """Weighted fusion of features with normalization."""
        weights = F.relu(weights)
        weights = weights / (weights.sum() + self.epsilon)
        return sum(w * f for w, f in zip(weights, features))
    
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            features: List of feature maps [P3, P4, P5, P6, P7]
            
        Returns:
            Refined feature maps at the same scales
        """
        # Top-down pathway
        top_down = [features[-1]]  # Start with P7
        for i in range(self.num_levels - 2, -1, -1):
            # Upsample higher level feature
            upsampled = F.interpolate(
                top_down[0],
                size=features[i].shape[2:],
                mode='nearest'
            )
            # Weighted fusion
            fused = self._weighted_fusion(
                [features[i], upsampled],
                self.top_down_weights[i]
            )
            # Apply conv block
            refined = self.top_down_blocks[i](fused)
            top_down.insert(0, refined)
        
        # Bottom-up pathway
        bottom_up = [top_down[0]]  # Start with P3
        for i in range(1, self.num_levels):
            # Downsample lower level feature
            downsampled = F.avg_pool2d(
                bottom_up[-1],
                kernel_size=2,
                stride=2
            )
            # Adjust size if needed
            if downsampled.shape[2:] != top_down[i].shape[2:]:
                downsampled = F.interpolate(
                    downsampled,
                    size=top_down[i].shape[2:],
                    mode='nearest'
                )
            # Weighted fusion
            fused = self._weighted_fusion(
                [top_down[i], downsampled],
                self.bottom_up_weights[i - 1]
            )
            # Apply conv block
            refined = self.bottom_up_blocks[i - 1](fused)
            bottom_up.append(refined)
        
        return bottom_up


class EfficientDetBackbone(nn.Module):
    """EfficientDet backbone with BiFPN.
    
    EfficientDet은 EfficientNet backbone과 BiFPN을 결합한 객체 탐지 모델입니다.
    timm 라이브러리를 통해 pre-trained EfficientDet 모델을 사용할 수 있습니다.
    """
    
    def __init__(
        self,
        model_name: str = "efficientdet_d0",
        pretrained: bool = True,
        features_only: bool = True
    ):
        """
        Args:
            model_name: EfficientDet model name (e.g., 'efficientdet_d0', 'efficientdet_d1')
            pretrained: Whether to use pre-trained weights
            features_only: If True, return feature maps from BiFPN
        """
        super().__init__()
        try:
            import timm
        except ImportError as e:
            raise RuntimeError("timm이 필요합니다: pip install timm") from e
        
        self.model_name = model_name
        self.pretrained = pretrained
        
        # Create EfficientDet model
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=features_only
        )
        
        if features_only:
            # Get feature info
            if hasattr(self.backbone, 'feature_info'):
                self.feature_info = self.backbone.feature_info
                self.out_ch = self.feature_info.channels()[-1]
            else:
                # Fallback: try to get from model structure
                self.out_ch = 64  # Default, will be updated
        else:
            self.out_ch = 1000  # Classification head output
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, 3, H, W]
            
        Returns:
            Feature maps from BiFPN (if features_only) or classification logits
        """
        output = self.backbone(x)
        
        if isinstance(output, (list, tuple)):
            # Return the last feature map (highest level)
            return output[-1]
        return output


def create_backbone(
    backbone_type: str = "efficientnet",
    model_name: Optional[str] = None,
    pretrained: bool = True,
    **kwargs
) -> nn.Module:
    """Factory function to create different backbone types.
    
    Args:
        backbone_type: Type of backbone ('efficientnet', 'regnet', 'efficientdet', 'bifpn')
        model_name: Specific model name (optional, uses defaults if None)
        pretrained: Whether to use pre-trained weights
        **kwargs: Additional arguments passed to backbone constructor
        
    Returns:
        Backbone model instance
        
    Examples:
        >>> # RegNet
        >>> backbone = create_backbone('regnet', 'regnetx_004', pretrained=True)
        >>> 
        >>> # EfficientDet (with BiFPN)
        >>> backbone = create_backbone('efficientdet', 'efficientdet_d0', pretrained=True)
        >>> 
        >>> # Custom BiFPN
        >>> backbone = create_backbone('bifpn', num_channels=64, num_levels=5)
    """
    if backbone_type.lower() == "regnet":
        if model_name is None:
            model_name = "regnetx_004"
        return RegNetBackbone(model_name=model_name, pretrained=pretrained, **kwargs)
    
    elif backbone_type.lower() == "efficientdet":
        if model_name is None:
            model_name = "efficientdet_d0"
        return EfficientDetBackbone(model_name=model_name, pretrained=pretrained, **kwargs)
    
    elif backbone_type.lower() == "bifpn":
        # Standalone BiFPN (requires input features)
        return BiFPN(**kwargs)
    
    elif backbone_type.lower() == "efficientnet":
        # Import from rt1_model for backward compatibility
        from ..rt1_model import EfficientNetBackbone
        if model_name is None:
            model_name = "tf_efficientnet_b1"
        # Note: EfficientNetBackbone requires prompt_dim, so we need to handle this
        prompt_dim = kwargs.pop('prompt_dim', 512)
        return EfficientNetBackbone(prompt_dim=prompt_dim, model_name=model_name)
    
    else:
        raise ValueError(
            f"Unknown backbone type: {backbone_type}. "
            f"Supported types: 'efficientnet', 'regnet', 'efficientdet', 'bifpn'"
        )

