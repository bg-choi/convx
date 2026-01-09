import os
import random
import re

from collections import defaultdict
from datasets import DatasetDict, load_dataset
from typing import Optional, Tuple, Dict, List

from transformers import PreTrainedTokenizerFast, Trainer
from transformers.file_utils import PaddingStrategy

from config import Arguments
from logger_config import logger
from .dataloader_utils import group_psg_ids

class CustomDataLoader:
    def __init__(
            self,
            args: Arguments,
            tokenizer: PreTrainedTokenizerFast,
            ret_tokenizer: PreTrainedTokenizerFast,
        ):
        self.args = args
        self.tokenizer = tokenizer
        self.ret_tokenizer = ret_tokenizer

        self.train_dataset = self._get_transformed_datasets()

        self.trainer: Optional[Trainer] = None

    def _transform_func(self, examples: Dict[str, List]) -> Dict[str, List]:
        passage_lists: List[str] = group_psg_ids(
            examples=examples,
            n_passages=self.args.n_passages
        )

        pids = self.ret_tokenizer(
            passage_lists,
            add_special_tokens=False,
            padding=PaddingStrategy.DO_NOT_PAD,
            max_length=self.args.p_max_len,
            truncation=True,
            return_attention_mask=False
        )
        trunc_passages = self.ret_tokenizer.batch_decode(pids['input_ids'])

        merged_dict = defaultdict()
        merged_dict['passage'] = []
        step_size = self.args.n_passages
        for idx in range(0, len(trunc_passages), step_size):
            merged_dict['passage'].append(trunc_passages[idx:(idx + step_size)])

        all_target_prompts = []
        all_labels = []

        inst = "Refer to the background document and answer the questions:"
        emb = "[MEM]" + "".join("<xRAG>" for i in range(self.args.mem_size)) + "[/MEM]"
        background = "\n".join(f"Background {i+1}: {emb}" for i in range(self.args.n_passages))

        for qry, ans in zip(examples['question'], examples['pred_answer']):
            target_messages = []
            
            target_q = {
                'role': 'user',
                'content': f'{inst}\n\n{background}\n\nQuestion: {qry}\n'
            }
            target_messages.append(target_q)
            
            target_prompt = self.tokenizer.apply_chat_template(
                target_messages,
                tokenize=False,    
                add_generation_prompt=True,
                enable_thinking=False,
            )
            target_prompt += " The answer is: "
            all_target_prompts.append(target_prompt)

            label = ans
            all_labels.append(label)
        
        merged_dict['target'] = all_target_prompts
        merged_dict['answer'] = all_labels
        
        return merged_dict


    def _get_transformed_datasets(self) -> Tuple:
        data_files = {}
        if self.args.train_file is not None:
            data_files['train'] = self.args.train_file.split(',')
        if self.args.validation_file is not None:
            data_files['validation'] = self.args.validation_file
        raw_datasets: DatasetDict = load_dataset('json', data_files=data_files)

        train_dataset = None

        if self.args.do_train:
            if 'train' not in raw_datasets:
                raise ValueError("--do_train requires a train dataset")
            train_dataset = raw_datasets['train']
            if self.args.max_train_samples is not None:
                train_dataset = train_dataset.select(range(self.args.max_train_samples))
            
            for index in random.sample(range(len(train_dataset)), 3):
                logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
            train_dataset.set_transform(self._transform_func)
        
        return train_dataset