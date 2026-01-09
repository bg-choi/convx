import os
import torch
from typing import Optional, Dict, Tuple

from models import End2EndModel
from logger_config import logger
from .trainer import Trainer

def _unpack_qp(inputs: Dict[str, torch.Tensor]) -> Tuple:
    target_batch_dict = {k[len('t_'):]: v for k, v in inputs.items() if k.startswith('t_')}
    encoder_batch_dict = {k[len('e_'):]: v for k, v in inputs.items() if k.startswith('e_')}

    if not target_batch_dict:
        target_batch_dict = None
    if not encoder_batch_dict:
        encoder_batch_dict = None
    
    return target_batch_dict, encoder_batch_dict


class CustomTrainer(Trainer):
    def __init__(self, *pargs, **kwargs):
        super(CustomTrainer, self).__init__(*pargs, **kwargs)
        self.model: End2EndModel
        self.last_epoch = 0
    
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to {}".format(output_dir))
        self.model.save(output_dir)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        target, encode = _unpack_qp(inputs)
        outputs = model(
            target=target,
            encode=encode
        )
        loss = outputs.loss
        
        return (loss, outputs) if return_outputs else loss