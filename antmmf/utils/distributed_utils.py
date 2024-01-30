# Copyright (c) 2023 Ant Group and its affiliates.
# Inspired from maskrcnn_benchmark
import pickle
import platform
from contextlib import contextmanager

import torch
from torch import distributed as dist

from antmmf.utils.general import nullcontext


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def synchronize():
    if platform.system() == "Darwin":
        return
    if not dist.is_nccl_available():
        return
    if not dist.is_initialized():
        return

    world_size = dist.get_world_size()

    if world_size == 1:
        return

    dist.barrier()


def get_rank():
    if platform.system() == "Darwin":
        return 0
    if not dist.is_nccl_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def get_world_size():
    if platform.system() == "Darwin":
        return 1
    if not dist.is_nccl_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def broadcast_tensor(tensor, src=0):
    world_size = get_world_size()
    if world_size < 2:
        return tensor

    with torch.no_grad():
        dist.broadcast(tensor, src=0)

    return tensor


def broadcast_scalar(scalar, src=0, device="cpu"):
    scalar_tensor = torch.tensor(scalar).to(device)
    scalar_tensor = broadcast_tensor(scalar_tensor, src)
    return scalar_tensor.item()


def reduce_tensor(tensor):
    world_size = get_world_size()

    if world_size < 2:
        return tensor

    with torch.no_grad():
        dist.reduce(tensor, dst=0)
        if dist.get_rank() == 0:
            tensor = tensor.div(world_size)

    return tensor


class GradientAllGather(torch.autograd.Function):
    """
    copy from: https://github.com/vlkit/vlkit/blob/master/vlkit/ops/distributed.py#L4-L25
    all_gather with gradient back-propagation
    """

    @staticmethod
    def forward(ctx, tensor_list, tensor):
        dist.all_gather(tensor_list, tensor)
        return tuple(tensor_list)

    @staticmethod
    def backward(ctx, *grad_list):
        grad_list = list(grad_list)
        rank = dist.get_rank()

        dist_ops = [
            dist.reduce(grad_list[i], i, async_op=True)
            for i in range(dist.get_world_size())
        ]

        for op in dist_ops:
            op.wait()

        return None, grad_list[rank]


gradient_all_gather = GradientAllGather.apply


def gather_tensor(
    tensor: torch.Tensor,
    method: str = "stack",
    back_gradient: bool = False,
    pad_tensors: bool = False,
) -> torch.Tensor:
    world_size = get_world_size()

    if world_size < 2:
        return tensor

    if tensor.ndim == 0 and (method != "stack"):
        raise ValueError(
            'gather_tensor not support 0-dim tensor with method is not "stack"'
        )
    if tensor.ndim == 0 and pad_tensors:
        raise ValueError(
            "gather_tensor not support 0-dim tensor with padding_tensors is True"
        )

    context = nullcontext() if back_gradient else torch.no_grad()

    with context:
        if pad_tensors:
            # dist.all_gather require tensors on all GPUs must be the same shape. If not well
            # followed, will cause hang during training. See detail at:
            # https://discuss.pytorch.org/t/how-to-concatenate-different-size-tensors-from-distributed-processes/44819
            # step1: obtain Tensor size of each rank
            tensor = tensor.contiguous()
            local_tensor_size = tensor.shape[0]
            size_tens = torch.tensor(
                local_tensor_size, dtype=torch.int64, device=tensor.device
            )
            size_tens = gather_tensor(size_tens, pad_tensors=False)

            # step2: all ranks allocate a tensor of max size
            max_size = size_tens.max().item()
            # pad tensor to ensure all tensor have the save shape
            padded_tensor = torch.empty(
                max_size, *tensor.shape[1:], dtype=tensor.dtype, device=tensor.device
            )
            padded_tensor[:local_tensor_size] = tensor
            tensor = padded_tensor

        # step3: run the real gather on max size tensor
        tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
        if back_gradient:
            gradient_all_gather(tensor_list, tensor)
        else:
            dist.all_gather(tensor_list, tensor)

        if pad_tensors:
            # step4: remove padding
            for idx, sz in enumerate(size_tens):
                tensor_list[idx] = tensor_list[idx][:sz]

        # Gathered tensors have no gradient, so we overwrite the gathered tensor for
        # the current replica. with the embeddings produced on this replica, which do
        # have gradients.
        # see detail: https://www.aiuai.cn/aifarm1762.html
        # tensor_list[get_rank()] = (
        #     tensor[:local_tensor_size] if tensor.ndim > 0 else tensor
        # )
        if method == "stack":
            tensor_list = torch.stack(tensor_list, dim=0)
        else:
            tensor_list = torch.cat(tensor_list, dim=0)
    return tensor_list


@contextmanager
def torch_distributed_zero_first():
    """
    Decorator to make all processes in distributed training wait for each local_master to do something.
    """
    local_rank = get_rank()
    if local_rank not in [-1, 0]:
        torch.distributed.barrier()
    yield
    if local_rank == 0:
        torch.distributed.barrier()


def reduce_dict(dictionary):
    """
    Args:
        dictionary (dict): all the values will be reduced
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return dictionary

    with torch.no_grad():
        keys, values = zip(*sorted(dictionary.items()))
        values = torch.stack(values, dim=0).cuda()

        dist.reduce(values, dst=0)

        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            values /= world_size
        reduced_dict = {k: v for k, v in zip(keys, values)}
    return reduced_dict


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.tensor([tensor.numel()], device="cuda")
    size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device="cuda"))
    if local_size != max_size:
        padding = torch.empty(
            size=(max_size - local_size,), dtype=torch.uint8, device="cuda"
        )
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list
