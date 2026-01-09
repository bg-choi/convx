import os
import shutil
import logging
import torch.distributed as dist

from transformers.utils.logging import enable_explicit_format
from transformers.trainer_callback import PrinterCallback
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    set_seed,
)

from config import Arguments
from logger_config import logger, LoggerCallback
from models import End2EndModel
from collators import CustomCollator
from loaders import CustomDataLoader
from trainers import CustomTrainer

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

    ret_tokenizer = AutoTokenizer.from_pretrained(args.retriever_name_or_path, padding_side='left')
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, padding_side='left')
    # tokenizer.pad_token_id = 0

    model: End2EndModel = End2EndModel.build(args=args)
    logger.info(model)
    
    data_collator = CustomCollator(
        tokenizer=tokenizer,
        pad_to_multiple_of=8 if args.fp16 else None,
        args=args,
        ret_tokenizer=ret_tokenizer
    )

    data_loader = CustomDataLoader(
        args=args,
        tokenizer=tokenizer,
        ret_tokenizer=ret_tokenizer
    )
    train_dataset = data_loader.train_dataset

    trainer: Trainer = CustomTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset if args.do_train else None,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    trainer.remove_callback(PrinterCallback)
    trainer.add_callback(LoggerCallback)
    data_loader.trainer = trainer
    model.trainer = trainer

    if args.do_train:
        train_result = trainer.train()
        trainer.save_model()
        finetuned = model.llm.merge_and_unload()
        finetuned.save_pretrained(os.path.join(args.output_dir))

        metrics = train_result.metrics
        metrics['train_samples'] = len(train_dataset)

        trainer.log_metrics('train', metrics)
        trainer.save_metrics('train', metrics)
        
        for fname in os.listdir(os.path.join(args.output_dir)):
            if 'checkpoint-' in fname:
                shutil.rmtree(os.path.join(args.output_dir, fname))
        
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
    
    return


if __name__ == "__main__":
    main()
