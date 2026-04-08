"""Shared pytest fixtures for medseg-pipeline tests."""

import pytest
import torch
import numpy as np


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def batch_size():
    return 2


@pytest.fixture
def num_classes():
    return 4


@pytest.fixture
def sample_logits(batch_size, num_classes):
    """Random logits tensor (B, C, H, W) for 2D segmentation tests."""
    torch.manual_seed(0)
    return torch.randn(batch_size, num_classes, 64, 64)


@pytest.fixture
def sample_targets(batch_size, num_classes):
    """Random integer label map (B, H, W)."""
    torch.manual_seed(1)
    return torch.randint(0, num_classes, (batch_size, 64, 64))


@pytest.fixture
def sample_volume_logits(batch_size, num_classes):
    """Random 3D logits (B, C, D, H, W) for volumetric tests."""
    torch.manual_seed(2)
    return torch.randn(batch_size, num_classes, 16, 64, 64)


@pytest.fixture
def sample_volume_targets(batch_size, num_classes):
    """Random 3D integer labels (B, D, H, W)."""
    torch.manual_seed(3)
    return torch.randint(0, num_classes, (batch_size, 16, 64, 64))
