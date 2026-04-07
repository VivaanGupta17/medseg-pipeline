"""Model architectures for medical image segmentation."""

from .unet import UNet
from .attention_unet import AttentionUNet

__all__ = ["UNet", "AttentionUNet"]


def get_model(model_name: str, **kwargs):
    """Factory function to instantiate a model by name.

    Args:
        model_name: One of 'unet', 'attention_unet'.
        **kwargs: Additional arguments forwarded to the model constructor.

    Returns:
        Instantiated model.

    Raises:
        ValueError: If model_name is not recognized.
    """
    registry = {
        "unet": UNet,
        "attention_unet": AttentionUNet,
    }
    if model_name not in registry:
        raise ValueError(
            f"Unknown model '{model_name}'. Available: {list(registry.keys())}"
        )
    return registry[model_name](**kwargs)
