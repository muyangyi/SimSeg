import os
import torch
import pickle
from typing import Any

import torch.distributed as dist

from . import logger
from .context import ENV


__all__ = [
    'all_gather_with_grad',
    'all_gather',
    'all_gather_group',
    'all_gather_object',
    'all_reduce',
    'broadcast',
    'broadcast_list',
    'barrier',
    'broadcast_object_list',
    'GatherLayer',
    'concat_all_gather',
    'generate_local_groups'
]


class all_gather_with_grad(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        output = [torch.empty_like(input) for _ in range(ENV.size)]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        grad_out = torch.empty_like(grads[0])
        grad_out[:] = grads[ENV.rank]
        return grad_out


def all_gather(tensor: Any, name=None):
    r""" A simple wrapper for all gather for both horovod and torch.distributed.

    Args:
        tensor: tensor to be all_gathered. Its type must be compatible with
        torch.Tensor
        name: The name of the tensor, for horovod.

    Return: The gathered tensor in form of List[torch.tensor]
    """

    if ENV.size == 1:
        return [tensor]

    if not isinstance(tensor, torch.Tensor):
        tensor = torch.tensor(tensor, device='cuda')

    gathered_tensor = [torch.empty_like(tensor) for _ in range(ENV.size)]
    dist.all_gather(gathered_tensor, tensor)
    return gathered_tensor


def all_gather_group(tensor, group):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(dist.get_world_size(group))]
    dist.all_gather(
        tensors_gather, tensor.contiguous(), async_op=False, group=group)
    return tensors_gather


def all_reduce(tensor: Any,
               name: str = None,
               average: bool = True,
               op: dist.ReduceOp = dist.ReduceOp.SUM):
    r""" A simple wrapper for all reduce for both horovod and torch.distributed.

    Args:
        tensor: tensor to be all_reduced. Its type must be compatible with torch.Tensor
        name: The name of the tensor.
        average: Whether to average the result.

    Return: The all_reduced tensor.
    """

    if not isinstance(tensor, torch.Tensor):
        tensor = torch.tensor(tensor, device='cuda')
    tensor = tensor.clone().detach()

    if ENV.size == 1:
        return tensor
    if op != dist.ReduceOp.SUM:
        average = False  # average only works when the op is dist.ReduceOp.SUM

    dist.all_reduce(tensor, op=op)
    reduced_tensor = tensor / ENV.size if average is True else tensor
    return reduced_tensor


def broadcast(tensor: torch.Tensor, name: str = None, src: int = 0):
    r""" A simple wrapper for broadcast tensor for both horovod and torch.distributed.
         Inplace broadcast mode is taken, aligned with torch official ddp broadcast

    Args:
        tensor: Tensor to be broadcast.
        name: The name of the tensor.
        src: Source rank id.

    Returns:
        Inplace update the tensor based on the src one.
    """

    if ENV.size == 1:
        return

    dist.broadcast(tensor, src=src)


def broadcast_list(x: list, name: str = None, src: int = 0):
    r""" A simple wrapper for broadcast list with nccl for both horovod and torch.distributed.

    Args:
        x: List of Tensor/int/float to be broadcast
        name: The name of the tensor.
        src: Source rank id.

    Returns:
        the list on rank src. Note that this is not an inplace opearation.
    """

    assert isinstance(x, list), 'broadcast_list only takes list as input.'
    tensor = torch.tensor(x).to(ENV.device)
    broadcast(tensor, name=name, src=src)
    return tensor.tolist()


def barrier():
    r""" A simple wrapper for barrier.
    """

    if ENV.size == 1:
        return

    dist.barrier()

def _object_to_tensor(obj):
    buffer = pickle.dumps(obj)
    byte_storage = torch.ByteStorage.from_buffer(buffer)
    byte_tensor = torch.ByteTensor(byte_storage)
    local_size = torch.LongTensor([byte_tensor.numel()])
    return byte_tensor, local_size


def _tensor_to_object(tensor, tensor_size):
    buf = tensor.numpy().tobytes()[:tensor_size]
    out = pickle.loads(buf)
    return out


def all_gather_object(object_list, obj, group, my_local_rank, group_size):
    """
    Gathers picklable objects from the whole group into a list. Similar to
    :func:`all_gather`, but Python objects can be passed in. Note that the object
    must be picklable in order to be gathered.

    Arguments:
        object_list (list[Any]): Output list. It should be correctly sized as the
            size of the group for this collective and will contain the output.
        object (Any): Pickable Python object to be broadcast from current process.
        group (ProcessGroup, optional): The process group to work on

    Returns:
        None. If the calling rank is part of this group, the output of the
        collective will be populated into the input ``object_list``. If the
        calling rank is not part of the group, the passed in ``object_list`` will
        be unmodified.

    .. note:: Note that this API differs slightly from the :func:`all_gather`
        collective since it does not provide an ``async_op`` handle and thus
        will be a blocking call.

    .. warning::
        :func:`all_gather_object` uses ``pickle`` module implicitly, which is
        known to be insecure. It is possible to construct malicious pickle data
        which will execute arbitrary code during unpickling. Only call this
        function with data you trust.
    """

    input_tensor, local_size = _object_to_tensor(obj)
    input_tensor, local_size = input_tensor.to(my_local_rank), local_size.to(my_local_rank)
    # Gather all local sizes. This is so that we can find the max size, and index
    # until the correct size when deserializing the tensors.
    object_sizes_tensor = torch.zeros(group_size, dtype=int).to(
        my_local_rank
    )
    object_size_list = [
        object_sizes_tensor[i].unsqueeze(dim=0) for i in range(group_size)
    ]
    # Allgather tensor sizes
    dist.all_gather(object_size_list, local_size, group=group)
    max_object_size = max(object_size_list)
    # Resize tensor to max size across all ranks.
    input_tensor.resize_(max_object_size)
    coalesced_output_tensor = torch.empty(
        max_object_size * group_size, dtype=torch.uint8
    ).to(my_local_rank)
    # Output tensors are nonoverlapping views of coalesced_output_tensor
    output_tensors = [
        coalesced_output_tensor[max_object_size * i: max_object_size * (i + 1)]
        for i in range(group_size)
    ]
    dist.all_gather(output_tensors, input_tensor, group=group)
    # Deserialize outputs back to object.
    for i, tensor in enumerate(output_tensors):
        tensor = tensor.type(torch.ByteTensor)
        tensor_size = object_size_list[i]
        object_list[i] = _tensor_to_object(tensor, tensor_size)


def broadcast_object_list(object_list, src=0, group=None):
    """
    Broadcasts picklable objects in ``object_list`` to the whole group. Similar
    to :func:`broadcast`, but Python objects can be passed in.
    Note that all objects in ``object_list`` must be picklable in order to be
    broadcasted.

    Args:
        object_list (List[Any]): List of input objects to broadcast.
            Each object must be picklable. Only objects on the ``src`` rank will
            be broadcast, but each rank must provide lists of equal sizes.
        src (int): Source rank from which to broadcast ``object_list``.
        group: (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used. Default is ``None``.

    Returns:
        ``None``. If rank is part of the group, ``object_list`` will contain the
        broadcasted objects from ``src`` rank.

    .. note:: For NCCL-based processed groups, internal tensor representations
        of objects must be moved to the GPU device before communication takes
        place. In this case, the device used is given by
        ``torch.cuda.current_device()`` and it is the user's responsiblity to
        ensure that this is set so that each rank has an individual GPU, via
        ``torch.cuda.set_device()``.

    .. note:: Note that this API differs slightly from the :func:`all_gather`
        collective since it does not provide an ``async_op`` handle and thus
        will be a blocking call.

    .. warning::
        :func:`broadcast_object_list` uses ``pickle`` module implicitly, which
        is known to be insecure. It is possible to construct malicious pickle
        data which will execute arbitrary code during unpickling. Only call this
        function with data you trust.

    Example::
        >>> # Note: Process group initialization omitted on each rank.
        >>> import torch.distributed as dist
        >>> if dist.get_rank() == 0:
        >>>     # Assumes world_size of 3.
        >>>     objects = ["foo", 12, {1: 2}] # any picklable object
        >>> else:
        >>>     objects = [None, None, None]
        >>> dist.broadcast_object_list(objects, src=0)
        >>> broadcast_objects
        ['foo', 12, {1: 2}]
    """
    if not group:
        group = dist.group.WORLD
    my_rank = dist.get_rank()
    # Serialize object_list elements to tensors on src rank.
    if my_rank == src:
        tensor_list, size_list = zip(
            *[_object_to_tensor(obj) for obj in object_list])
        object_sizes_tensor = torch.cat(size_list)
    else:
        object_sizes_tensor = torch.empty(len(object_list), dtype=torch.long)

    group_backend = dist.get_backend(group)
    is_nccl_backend = group_backend == dist.Backend.NCCL
    current_device = torch.device("cpu")
    if is_nccl_backend:
        # See note about using torch.cuda.current_device() here in docstring.
        # We cannot simply use my_rank since rank == device is not necessarily
        # true.
        current_device = torch.device('cuda', torch.cuda.current_device())
        object_sizes_tensor = object_sizes_tensor.to(current_device)
        object_sizes_tensor = object_sizes_tensor.to(current_device)

    # Broadcast object sizes
    dist.broadcast(object_sizes_tensor, src=src, group=group)

    # Concatenate and broadcast serialized object tensors
    if my_rank == src:
        object_tensor = torch.cat(tensor_list)
    else:
        object_tensor = torch.empty(
            torch.sum(object_sizes_tensor).int().item(),
            # type: ignore[arg-type]
            dtype=torch.uint8
        )

    if is_nccl_backend:
        object_tensor = object_tensor.to(current_device)
    dist.broadcast(object_tensor, src=src, group=group)
    # Deserialize objects using their stored sizes.
    offset = 0
    if my_rank != src:
        for i, obj_size in enumerate(object_sizes_tensor):
            obj_view = object_tensor[offset: offset + obj_size]
            obj_view = obj_view.type(torch.uint8)  # type: ignore[call-overload]
            if obj_view.device != torch.device("cpu"):
                obj_view = obj_view.cpu()
            offset += obj_size
            object_list[i] = _tensor_to_object(obj_view, obj_size)


class GatherLayer(torch.autograd.Function):
    """
        :class:`GatherLayer` is a module wrapper that realizes backward op in all_gather
        Usage:
        feat_global = torch.cat(all_gather(feat, group), 0)
        # equals to
        feat_global = GatherLayer.apply(feat, group, rank)
    """

    @staticmethod
    def forward(ctx, tensor, group, rank):
        ctx.batch_size = tensor.shape[0]
        ctx.group = group
        ctx.rank = rank

        gathered_tensor = [torch.zeros_like(tensor) for _ in
                           range(dist.get_world_size(group))]

        dist.all_gather(gathered_tensor, tensor, group=group)
        gathered_tensor = torch.cat(gathered_tensor, 0)

        return gathered_tensor

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        dist.all_reduce(grad_input, op=dist.ReduceOp.SUM, async_op=False,
                        group=ctx.group)

        idx_from = ctx.rank * ctx.batch_size
        idx_to = (ctx.rank + 1) * ctx.batch_size
        return grad_input[idx_from:idx_to], None, None


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(ENV.size)]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


def generate_local_groups(local_group_size):
    if not dist.is_initialized():
        logger.error("this function is only supported by pytorch distributed training")
        exit(1)
    local_group_rank = 0
    new_group_times = 0
    host_this_rank = os.environ['HOSTNAME']
    rank_info = (host_this_rank, ENV.rank)
    ranks_info_list = [None for _ in range(ENV.size)]
    all_gather_object(ranks_info_list, rank_info, dist.group.WORLD, ENV.local_rank, ENV.size)
    ranks_info_dict = {}
    for each in ranks_info_list:
        host = each[0]
        if host in ranks_info_dict:
            ranks_info_dict[host].append(each[1])
        else:
            ranks_info_dict[host] = [each[1]]

    # first try to construct local_group inside a node
    for host in ranks_info_dict:
        while len(ranks_info_dict[host]) >= local_group_size:
            # pop [local_group_size] item to form a group
            tmp_rank_list = ranks_info_dict[host][:local_group_size]
            ranks_info_dict[host] = ranks_info_dict[host][local_group_size:]
            if host_this_rank == host and ENV.rank in tmp_rank_list:
                should_return = dist.new_group(tmp_rank_list)
                local_group_rank = tmp_rank_list.index(ENV.rank)
                new_group_times += 1
                logger.info("Generate a local group {}, local_group_rank {} for rank {}".format(
                    tmp_rank_list, local_group_rank, ENV.rank))
            else:
                dist.new_group(tmp_rank_list)
                new_group_times += 1

    # if some ranks are remained, we form the local group by flatting the dict
    tmp_rank_list = []
    this_rank_flag = False
    for host in ranks_info_dict:
        while ranks_info_dict[host]:
            rank = ranks_info_dict[host].pop(0)
            if host_this_rank == host and ENV.rank == rank:
                this_rank_flag = True
            tmp_rank_list.append(rank)
            if len(tmp_rank_list) == local_group_size:
                if this_rank_flag:
                    should_return = dist.new_group(tmp_rank_list)
                    local_group_rank = tmp_rank_list.index(ENV.rank)
                    new_group_times += 1
                    logger.info("Generate a local group {}, local_group_rank {} for rank {}".format(
                        tmp_rank_list, local_group_rank, ENV.rank))
                    tmp_rank_list = []
                    this_rank_flag = False
                else:
                    dist.new_group(tmp_rank_list)
                    new_group_times += 1
                    tmp_rank_list = []
    assert new_group_times == ENV.size / local_group_size
    return should_return, local_group_rank