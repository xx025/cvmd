import torch


def normalize_device(device) -> torch.device:
    return torch.device("cpu") if device is None else torch.device(device)