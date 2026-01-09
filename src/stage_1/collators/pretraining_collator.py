import torch
from copy import copy
from dataclasses import dataclass
from typing import List, Dict, Any

from transformers import DataCollatorWithPadding, BatchEncoding
from transformers import PreTrainedTokenizerFast

from config import Arguments

@dataclass
class PretrainingCollator(DataCollatorWithPadding):
    args: Arguments = None
    ret_tokenizer: PreTrainedTokenizerFast = None

    def __call__(self, features: List[Dict[str, Any]]) -> BatchEncoding:
        all_e_inputs = []
        all_d_inputs = []

        for f in features:
            p = f.get('passage', [])
            all_e_inputs.append(p)
            all_d_inputs.append(p)

        e_collated = self.ret_tokenizer(
            all_e_inputs,
            padding='longest',
            max_length=self.args.p_max_len,
            return_tensors=self.return_tensors,
            return_attention_mask=True,
            truncation=True,
        )
        d_collated = self.tokenizer(
            all_d_inputs,
            add_special_tokens=False,
            padding='longest',
            max_length=self.args.p_max_len,
            return_tensors=self.return_tensors,
            return_attention_mask=True,
            truncation=True,
        )

        for k in list(e_collated.keys()):
            e_collated['e_' + k] = e_collated[k]
            del e_collated[k]
        for k in list(d_collated.keys()):
            e_collated['d_' + k] = d_collated[k]
        
        merged_batch_dict = e_collated
        labels = torch.zeros(len(e_collated['e_input_ids']), dtype=torch.long)
        merged_batch_dict['labels'] = labels

        return merged_batch_dict