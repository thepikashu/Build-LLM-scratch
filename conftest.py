"""
Shared pytest configuration.
Sets a global random seed before every test for reproducibility.
"""
import torch
import pytest


@pytest.fixture(autouse=True)
def set_random_seed():
    torch.manual_seed(42)
    yield
