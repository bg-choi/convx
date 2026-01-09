import os

from datasets import Dataset, load_dataset
from typing import Optional, Dict, List

from transformers import PreTrainedTokenizerFast, Trainer
from transformers.file_utils import PaddingStrategy

from config import Arguments


class PretrainingDataLoader:
    def __init__(
            self,
            args: Arguments,
            ret_tokenizer: PreTrainedTokenizerFast,
        ):
        self.args = args
        self.ret_tokenizer = ret_tokenizer

        corpus_path = os.path.join(args.data_dir, 'collection', 'kilt-128-sample.jsonl')
        self.train_dataset: Dataset = load_dataset('json', data_files=corpus_path)['train']
        self.train_dataset.set_transform(self._transform_func)

        self.trainer: Optional[Trainer] = None

    def __getitem__(self, index):
        return self.train_dataset[index]
    
    def __len__(self):
        return len(self.train_dataset)

    def _transform_func(self, examples: Dict[str, List]) -> Dict[str, List]:
        e_collated = self.ret_tokenizer(
            examples['text'],
            add_special_tokens=False,
            max_length=self.args.p_max_len,
            padding=PaddingStrategy.DO_NOT_PAD,
            truncation=True,
            return_attention_mask=False
        )
        trun_passages = self.ret_tokenizer.batch_decode(e_collated['input_ids'])
        
        merged_dict = {
            'passage': trun_passages
        }

        return merged_dict