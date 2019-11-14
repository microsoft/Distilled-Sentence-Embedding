import torch
import torch.utils.data


def get_use_cuda(disable_cuda=False):
    """
    Returns true if cuda is available and no explicit disable cuda flag given.
    """
    return torch.cuda.is_available() and not disable_cuda


def get_device(disable_cuda=False, cuda_id=0):
    """
    Returns a gpu cuda device if available and cpu device otherwise.
    """
    if get_use_cuda(disable_cuda):
        return torch.device(f"cuda:{cuda_id}")
    return torch.device("cpu")


def set_requires_grad(module, requires_grad):
    """
    Sets the requires grad flag for all of the modules parameters.
    :param module: pytorch module.
    :param requires_grad: requires grad flag value.
    """
    for param in module.parameters():
        param.requires_grad = requires_grad
