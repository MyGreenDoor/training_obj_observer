
import os
import subprocess
import typing

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import builtins

# Check for MLU (Cambricon) availability
try:
    import torch_mlu
    IS_MLU_AVAILABLE = torch.mlu.is_available()
except ImportError:
    IS_MLU_AVAILABLE = False

# Check for NPU (Ascend) availability  
try:
    import torch_npu
    IS_NPU_AVAILABLE = torch.npu.is_available()
except ImportError:
    IS_NPU_AVAILABLE = False
    
    
def init_dist(launcher: str, backend: str = 'nccl', **kwargs) -> None:
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')
    if launcher == 'pytorch':
        _init_dist_pytorch(backend, **kwargs)
    elif launcher == 'mpi':
        _init_dist_mpi(backend, **kwargs)
    elif launcher == 'slurm':
        _init_dist_slurm(backend, **kwargs)
    else:
        raise ValueError(f'Invalid launcher type: {launcher}')


def _init_dist_pytorch(backend: str, **kwargs) -> None:
    # TODO: use local_rank instead of rank % num_gpus
    rank = int(os.environ['RANK'])
    if IS_MLU_AVAILABLE:
        import torch_mlu  # noqa: F401
        torch.mlu.set_device(rank)
        dist.init_process_group(
            backend='cncl',
            rank=rank,
            world_size=int(os.environ['WORLD_SIZE']),
            **kwargs)
    elif IS_NPU_AVAILABLE:
        import torch_npu  # noqa: F401
        num_npus = torch.npu.device_count()
        torch.npu.set_device(rank % num_npus)
        dist.init_process_group(
            backend='hccl',
            rank=rank,
            world_size=int(os.environ['WORLD_SIZE']),
            **kwargs)
    else:
        num_gpus = torch.cuda.device_count()
        torch.cuda.set_device(rank % num_gpus)
        dist.init_process_group(backend=backend, **kwargs)


def _init_dist_mpi(backend: str, **kwargs) -> None:
    local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    if 'MASTER_PORT' not in os.environ:
        # 29500 is torch.distributed default port
        os.environ['MASTER_PORT'] = '29500'
    if 'MASTER_ADDR' not in os.environ:
        raise KeyError('The environment variable MASTER_ADDR is not set')
    os.environ['WORLD_SIZE'] = os.environ['OMPI_COMM_WORLD_SIZE']
    os.environ['RANK'] = os.environ['OMPI_COMM_WORLD_RANK']
    dist.init_process_group(backend=backend, **kwargs)


def _init_dist_slurm(backend: str, port: typing.Optional[int] = None) -> None:
    """Initialize slurm distributed training environment.

    If argument ``port`` is not specified, then the master port will be system
    environment variable ``MASTER_PORT``. If ``MASTER_PORT`` is not in system
    environment variable, then a default port ``29500`` will be used.

    Args:
        backend (str): Backend of torch.distributed.
        port (int, optional): Master port. Defaults to None.
    """
    proc_id = int(os.environ['SLURM_PROCID'])
    ntasks = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(proc_id % num_gpus)
    addr = subprocess.getoutput(
        f'scontrol show hostname {node_list} | head -n1')
    # specify master port
    if port is not None:
        os.environ['MASTER_PORT'] = str(port)
    elif 'MASTER_PORT' in os.environ:
        pass  # use MASTER_PORT in the environment variable
    else:
        # if torch.distributed default port(29500) is available
        # then use it, else find a free port
        if _is_free_port(29500):
            os.environ['MASTER_PORT'] = '29500'
        else:
            os.environ['MASTER_PORT'] = str(_find_free_port())
    # use MASTER_ADDR in the environment variable if it already exists
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = addr
    os.environ['WORLD_SIZE'] = str(ntasks)
    os.environ['LOCAL_RANK'] = str(proc_id % num_gpus)
    os.environ['RANK'] = str(proc_id)
    dist.init_process_group(backend=backend)
    
    
def init_distributed(backend="nccl"):
    # If launched with torchrun, env vars are set
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group(backend=backend, init_method="env://")
        torch_device = f"cuda:{local_rank}" if backend == "nccl" else "cpu"
        return True
    else:
        return False


def get_rank():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0

def get_world_size():
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1

def is_main_process():
    return get_rank() == 0

def barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

def setup_for_distributed(is_master: bool):
    """
    This function disables printing when not in master process to avoid clutter.
    """
    import builtins as __builtins__
    builtin_print = __builtins__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtins__.print = print
