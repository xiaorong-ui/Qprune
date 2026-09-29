"""Small inference utilities from the LLaVA runtime."""


def disable_torch_init():
    """Skip redundant parameter initialization before loading a checkpoint."""
    import torch

    torch.nn.Linear.reset_parameters = lambda self: None
    torch.nn.LayerNorm.reset_parameters = lambda self: None
