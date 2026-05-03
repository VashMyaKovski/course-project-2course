def resolve_device(preferred_device: str | None = None) -> str:
    """
    Return computation device string for transformer models.
    """
    if preferred_device:
        if preferred_device.startswith("cuda"):
            try:
                import torch
            except ImportError:
                return "cpu"
            return preferred_device if torch.cuda.is_available() else "cpu"
        return preferred_device

    try:
        import torch
    except ImportError:
        return "cpu"

    return "cuda" if torch.cuda.is_available() else "cpu"
