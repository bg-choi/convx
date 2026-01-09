import os
import torch
from dataclasses import dataclass, field
from typing import Optional
from transformers import TrainingArguments

from logger_config import logger

@dataclass
class Arguments(TrainingArguments):
    stage: str = field(
        default='stage_1', metadata={"help": "Path to source code"}
    )
    model_name_or_path: str = field(
        default='mistralai/Mistral-7B-Instruct-v0.2',
        metadata={"help": "Path to backbone model or model identifier from huggingface.co/models"}
    )
    retriever_name_or_path: str = field(
        default='Salesforce/SFR-Embedding-Mistral',
        metadata={"help": "Path to retriever or model identifier from huggingface.co/models"}
    )
    data_dir: str = field(
        default='./data/', metadata={"help": "Path to data directory"}
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a jsonl file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the metrics on (a jsonlines file)."
        },
    )
    
    max_grad_norm: float = field(default=1.0, metadata={"help": "gradient clipping"})
    weight_decay: float = field(default=0.05, metadata={"help": "weight decay"})
    
    mem_size: int = field(
        default=16,
        metadata={
            "help": "The number of memory slots."
        },
    )
    p_max_len: int = field(
        default=256,
        metadata={
            "help": "The maximum context length after tokenization for passage."
        },
    )
    dry_run: Optional[bool] = field(
        default=False,
        metadata={'help': 'Set dry_run to True for debugging purpose'}
    )

    def __post_init__(self):
        assert os.path.exists(self.data_dir)
        assert torch.cuda.is_available(), 'Only support running on GPUs'

        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)

        if self.dry_run:
            self.logging_steps = 1
            self.max_train_samples = 128
            self.per_device_train_batch_size = min(1, self.per_device_train_batch_size)
            self.gradient_accumulation_steps = 1
            self.max_steps = 30
            self.save_steps = self.eval_steps = 30
            logger.warning('Dry run: set logging_steps=1')
        
        super(Arguments, self).__post_init__()
        self.label_names = ['labels']
