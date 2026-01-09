import torch
from typing import Mapping, Dict, Tuple

def move_to_cuda(sample):
    if len(sample) == 0:
        return {}

    def _move_to_cuda(maybe_tensor):
        if torch.is_tensor(maybe_tensor):
            return maybe_tensor.cuda(non_blocking=True)
        elif isinstance(maybe_tensor, dict):
            return {key: _move_to_cuda(value) for key, value in maybe_tensor.items()}
        elif isinstance(maybe_tensor, list):
            return [_move_to_cuda(x) for x in maybe_tensor]
        elif isinstance(maybe_tensor, tuple):
            return tuple([_move_to_cuda(x) for x in maybe_tensor])
        elif isinstance(maybe_tensor, Mapping):
            return type(maybe_tensor)({k: _move_to_cuda(v) for k, v in maybe_tensor.items()})
        else:
            return maybe_tensor

    return _move_to_cuda(sample)

def _unpack_qp(inputs: Dict[str, torch.Tensor]) -> Tuple:
    target_batch_dict = {k[len('t_'):]: v for k, v in inputs.items() if k.startswith('t_')}
    encoder_batch_dict = {k[len('e_'):]: v for k, v in inputs.items() if k.startswith('e_')}

    if not target_batch_dict:
        target_batch_dict = None
    if not encoder_batch_dict:
        encoder_batch_dict = None
    
    return target_batch_dict, encoder_batch_dict