def resolve_device() -> str:
    """
    Return computation device string for transformer models.
    Use CUDA when available, else CPU.
    """
    try:
        import torch
    except ImportError:
        return "cpu"

    return "cuda" if torch.cuda.is_available() else "cpu"
