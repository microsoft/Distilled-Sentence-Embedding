def to_numpy(tensor):
    """
    Converts tensor to numpy ndarray. Will move tensor to cpu and detach it before converison. Numpy ndarray will share memory
    of the tensor.
    :param tensor: input pytorch tensor.
    :return: numpy ndarray with shared memory of the given tensor.
    """
    return tensor.cpu().detach().numpy()
