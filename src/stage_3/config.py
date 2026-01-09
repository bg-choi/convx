import os
import torch
from dataclasses import dataclass, field
from typing import Optional
from transformers import TrainingArguments

from logger_config import logger

@dataclass
class Arguments(TrainingArguments):
    stage: str = field(
        default='stage_3', metadata={"help": "Path to source code"}
    )
    model_name_or_path: str = field(
        default='mistralai/Mistral-7B-Instruct-v0.2',
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    retriever_name_or_path: str = field(
        default='Salesforce/SFR-Embedding-Mistral',
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    data_dir: str = field(
        default='./data/', metadata={"help": "Path to train directory"}
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a jsonlines file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the metrics on (a jsonlines file)."
        },
    )

    max_grad_norm: float = field(default=1.0, metadata={"help": "gradient clipping"})
    weight_decay: float = field(default=0.00, metadata={"help": "weight decay"})
    mem_size: int = field(
        default=16,
        metadata={
            "help": "The maximum total input sequence length after tokenization for passage."
        },
    )
    n_passages: int = field(
        default=5,
        metadata={"help": "number of passages for each example (including both positive and negative passages)"}
    )
    p_max_len: int = field(
        default=256,
        metadata={
            "help": "The maximum total input sequence length after tokenization for passage."
        },
    )
    gradient_checkpointing: bool = field(
        default=False,
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
                    "value if set."
        },
    )
    lora_rank: int = field(
        default=32,
        metadata={"help": "LoRA r"}
    )
    lora_alpha: int = field(
        default=128,
        metadata={"help": "LoRA a"}
    )
    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "LoRA dropout"}
    )

    # used for index search
    do_inference: bool = field(default=False, metadata={"help": "Inference with ConvX"})
    file_name: Optional[str] = field(default='nq_dev_top5', metadata={"help": "inference file"})
    max_new_tokens: int = field(
        default=100,
        metadata={"help": "number of passages for each example (including both positive and negative passages)"}
    )

    dry_run: Optional[bool] = field(
        default=False,
        metadata={'help': 'Set dry_run to True for debugging purpose'}
    )
    def __post_init__(self):
        assert os.path.exists(self.data_dir)
        assert torch.cuda.is_available(), 'Only support running on GPUs'
        assert self.output_dir

        if self.dry_run:
            self.logging_steps = 1
            self.max_train_samples = 128
            self.per_device_train_batch_size = min(2, self.per_device_train_batch_size)
            self.train_n_passages = min(1, self.train_n_passages)
            self.gradient_accumulation_steps = 1
            self.max_steps = 30
            self.save_steps = self.eval_steps = 30
            logger.warning('Dry run: set logging_steps=1')

        if torch.cuda.device_count() <= 1:
            self.logging_steps = min(10, self.logging_steps)

        super(Arguments, self).__post_init__()

        self.label_names = ['labels']
