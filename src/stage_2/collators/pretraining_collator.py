import torch
from dataclasses import dataclass
from typing import List, Dict, Any

from transformers import (
    DataCollatorWithPadding,
    BatchEncoding
)
from transformers import PreTrainedTokenizerFast

from config import Arguments

@dataclass
class PretrainingCollator(DataCollatorWithPadding):
    args: Arguments = None
    ret_tokenizer: PreTrainedTokenizerFast = None

    def __call__(self, features: List[Dict[str, Any]]) -> BatchEncoding:
        tasks = []
        enc_inputs = []
        dec_inputs = []
        dec_labels = []

        for f in features:
            task = f.get('task', [])
            tasks.append(task)

            for p in f.get('passage', []):
                enc_inputs.append(p)

            prompt = f.get('prompt', [])
            label = f.get('label')

            dec_inputs.append(prompt)
            dec_labels.append(label)
            
        e_collated = self.ret_tokenizer(
            enc_inputs,
            padding='longest',
            return_tensors=self.return_tensors,
            return_attention_mask=True,
        )

        d_collated = self.tokenizer.pad(
            {'input_ids': dec_inputs},
            padding='longest',
            return_tensors=self.return_tensors,
            return_attention_mask=True,
        )
        dl_collated = self.tokenizer.pad(
            {'input_ids': dec_labels},
            padding='longest',
            return_tensors=self.return_tensors,
            return_attention_mask=False,
        )
        d_collated['labels'] = dl_collated['input_ids'].clone()
        d_collated['labels'][d_collated['labels']==self.tokenizer.pad_token_id] = -100
        
        for k in list(d_collated.keys()):
            d_collated['d_' + k] = d_collated[k]
            del d_collated[k]
        for k in e_collated:
            d_collated['e_' + k] = e_collated[k]

        merged_batch_dict = d_collated
        labels = torch.zeros(len(d_collated['d_input_ids']), dtype=torch.long)
        merged_batch_dict['labels'] = labels
        
        return merged_batch_dict