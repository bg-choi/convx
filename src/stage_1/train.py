import os
import logging
import torch.distributed as dist

from transformers.utils.logging import enable_explicit_format
from transformers.trainer_callback import PrinterCallback
from transformers import (
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    set_seed,
)

from logger_config import logger, LoggerCallback
from config import Arguments
from collators import PretrainingCollator
from loaders import PretrainingDataLoader
from models import ReconstructModel
from trainers import PretrainingTrainer

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def _common_setup(args: Arguments):
    if args.process_index > 0:
        logger.setLevel(logging.WARNING)
    enable_explicit_format()
    set_seed(args.seed)

def main():
    parser = HfArgumentParser(Arguments)
    args, remaining_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    _common_setup(args)
    logger.info('Args={}'.format(str(args)))

    config = AutoConfig.from_pretrained(args.model_name_or_path)
    ret_tokenizer = AutoTokenizer.from_pretrained(args.retriever_name_or_path, padding_side='left')
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, padding_side='left')
    tokenizer.pad_token_id = 0
    
    model: ReconstructModel = ReconstructModel.build(args=args, config=config, tokenizer=tokenizer)
    logger.info(model)

    data_collator = PretrainingCollator(
        tokenizer=tokenizer,
        pad_to_multiple_of=8 if args.fp16 else None,
        args=args,
        ret_tokenizer=ret_tokenizer,
    )

    data_loader = PretrainingDataLoader(args=args, ret_tokenizer=ret_tokenizer)
    train_dataset = data_loader.train_dataset

    trainer: Trainer = PretrainingTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset if args.do_train else None,
        data_collator=data_collator,
    )
    trainer.remove_callback(PrinterCallback)
    trainer.add_callback(LoggerCallback)
    data_loader.trainer = trainer
    model.trainer = trainer

    if args.do_train:
        train_result = trainer.train()
        trainer.save_model()

        metrics = train_result.metrics
        metrics['train_samples'] = len(train_dataset)

        trainer.log_metrics('train', metrics)
        trainer.save_metrics('train', metrics)
        
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
    
    return

if __name__ == "__main__":
    main()
