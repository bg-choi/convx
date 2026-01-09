import os
import torch
from typing import Optional, Dict, Tuple

from models import ReconstructModel
from logger_config import logger
from .trainer import Trainer

def _unpack_qp(inputs: Dict[str, torch.Tensor]) -> Tuple:
    enc_batch_dict = {k[len('e_'):]: v for k, v in inputs.items() if k.startswith('e_')}
    dec_batch_dict = {k[len('d_'):]: v for k, v in inputs.items() if k.startswith('d_')}

    if not enc_batch_dict:
        enc_batch_dict = None
    if not dec_batch_dict:
        dec_batch_dict = None
    
    return enc_batch_dict, dec_batch_dict


class PretrainingTrainer(Trainer):
    def __init__(self, *pargs, **kwargs):
        super(PretrainingTrainer, self).__init__(*pargs, **kwargs)
        self.model: ReconstructModel
        self.last_epoch = 0
    
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to {}".format(output_dir))
        self.model.save(output_dir)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        encoder, decoder = _unpack_qp(inputs)
        outputs = model(encoder=encoder, decoder=decoder)
        loss = outputs[0]
        
        return (loss, outputs) if return_outputs else loss
