import torch

from dataclasses import dataclass
from typing import List, Dict, Any
from transformers import DataCollatorWithPadding, BatchEncoding
from transformers import PreTrainedTokenizerFast

from config import Arguments


@dataclass
class CustomCollator(DataCollatorWithPadding):
    args: Arguments = None
    ret_tokenizer: PreTrainedTokenizerFast = None

    def __call__(self, features: List[Dict[str, Any]]) -> BatchEncoding:
        encoder_inputs = []
        target_inputs = []
        target_labels = []

        for f in features:
            for p in f.get('passage', []):
                encoder_inputs.append(p)

            t = f.get('target', [])
            a = f.get('answer', [])

            t_prompt_ids = self.tokenizer.encode(t, add_special_tokens=False)
            a_ids = self.tokenizer.encode(a, add_special_tokens=False)
            t_ids = t_prompt_ids + a_ids
            tl_ids = ([-100] * len(t_prompt_ids)) + a_ids
            target_inputs.append(t_ids)
            target_labels.append(tl_ids)

        t_collated = self.tokenizer.pad(
            {'input_ids': target_inputs},
            padding='longest',
            return_tensors=self.return_tensors,
            return_attention_mask=True,
        )
        tl_collated = self.tokenizer.pad(
            {'input_ids': target_labels},
            padding='longest',
            return_tensors=self.return_tensors,
            return_attention_mask=False,
        )
        t_collated['labels'] = tl_collated['input_ids'].clone()
        t_collated['labels'][t_collated['labels']==self.tokenizer.pad_token_id] = -100

        del tl_collated

        e_collated = self.ret_tokenizer(
            encoder_inputs,
            padding='longest',
            return_tensors=self.return_tensors,
            return_attention_mask=True,
        )
        
        for k in list(t_collated.keys()):
            t_collated['t_' + k] = t_collated[k]
            del t_collated[k]
        for k in e_collated:
            t_collated['e_' + k] = e_collated[k]

        merged_batch_dict = t_collated
        labels = torch.zeros(len(t_collated['t_input_ids']), dtype=torch.long)
        merged_batch_dict['labels'] = labels

        return merged_batch_dict
    
@dataclass
class CustomCollatorForInference(DataCollatorWithPadding):
    args: Arguments = None
    ret_tokenizer: PreTrainedTokenizerFast = None
    
    def __call__(self, features: List[Dict[str, Any]]) -> BatchEncoding:
        encoder_inputs = []
        target_inputs = []
        for f in features:
            for p in f.get('passage', []):
                encoder_inputs.append(p)

            prompt = f.get('prompt', [])
            prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            target_inputs.append(prompt_ids)

        t_collated = self.tokenizer.pad(
            {'input_ids': target_inputs},
            padding='longest',
            return_tensors=self.return_tensors,
            return_attention_mask=True,
        )
        
        e_collated = self.ret_tokenizer(
            encoder_inputs,
            padding='longest',
            return_tensors=self.return_tensors,
            return_attention_mask=True,
        )
        
        for k in list(t_collated.keys()):
            t_collated['t_' + k] = t_collated[k]
            del t_collated[k]
        for k in e_collated:
            t_collated['e_' + k] = e_collated[k]

        return t_collated
