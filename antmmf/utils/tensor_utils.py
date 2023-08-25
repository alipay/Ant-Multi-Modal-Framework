import torch


def get_device_of(tensor: torch.Tensor) -> int:
    """
    Returns the device of the tensor.
    """
    if not tensor.is_cuda:
        return -1
    else:
        return tensor.get_device()


def find_embedding_layer(model, module_name) -> torch.nn.Module:

    model_ptr = model
    for ml in module_name.split("."):
        model_ptr = getattr(model_ptr, ml, None)
        if model_ptr is None:
            return model_ptr

    return model_ptr
