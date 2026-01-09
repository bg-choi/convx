import os
import torch
from typing import Optional, Dict, Tuple

from models import ReconstructModel, PretrainOutput
from logger_config import logger
from utils import AverageMeter
from .trainer import Trainer

def _unpack_qp(inputs: Dict[str, torch.Tensor]) -> Tuple:
    e_batch_dict = {k[len('e_'):]: v for k, v in inputs.items() if k.startswith('e_')}
    d_batch_dict = {k[len('d_'):]: v for k, v in inputs.items() if k.startswith('d_')}

    if not e_batch_dict:
        e_batch_dict = None
    if not d_batch_dict:
        d_batch_dict = None
    
    return e_batch_dict, d_batch_dict


class PretrainingTrainer(Trainer):
    def __init__(self, *pargs, **kwargs):
        super(PretrainingTrainer, self).__init__(*pargs, **kwargs)
        self.model: ReconstructModel
        self.last_epoch = 0

        self.sparse_meter = AverageMeter('sparse', round_digits=2)
        self.dense_meter = AverageMeter('dense', round_digits=2)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to {}".format(output_dir))
        self.model.save(output_dir)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        encoder, decoder = _unpack_qp(inputs)
        outputs: PretrainOutput = model(encoder=encoder, decoder=decoder)

        if self.model.training:
            self.sparse_meter.update(outputs.sparse_loss.item())
            self.dense_meter.update(outputs.dense_loss.item())

        return outputs.loss
    
    def log(self, logs: Dict[str, float], start_time: Optional[float]=None) -> None:
        """
        Intercepts the trainer's logging call.
        Adds custom metrics from our meters to the logs, then resets the meters.
        """
        # Add the averaged values from your meters to the logs dictionary
        if self.model.training:
            logs['sparse_loss'] = round(self.sparse_meter.avg, 2)
            logs['dense_loss'] = round(self.dense_meter.avg, 2)

        # The parent `log` method handles the actual logging (e.g., to console, W&B, TensorBoard)
        super().log(logs)

        # Reset meters for the next logging window
        if self.model.training:
            self._reset_meters_if_needed()

    def _reset_meters_if_needed(self):
        if int(self.state.epoch) != self.last_epoch:
            self.last_epoch = int(self.state.epoch)
            self.sparse_meter.reset()
            self.dense_meter.reset()