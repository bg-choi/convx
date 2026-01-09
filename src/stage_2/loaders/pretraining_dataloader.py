import os
import random
from copy import copy

from datasets import Dataset, load_dataset
from typing import Optional, Dict, Any

from transformers import PreTrainedTokenizerFast, Trainer

from config import Arguments
from .dataloader_utils import (
    AEInstructions,
    LMInstructions,
    AESameInstructions,
    AEDiffInstructions
)

class PretrainingDataLoader:
    def __init__(
            self,
            args: Arguments,
            tokenizer: PreTrainedTokenizerFast,
            ret_tokenizer: PreTrainedTokenizerFast
        ):
        self.args = args
        self.tokenizer = tokenizer
        self.ret_tokenizer = ret_tokenizer
        self.global_step = 0

        self.tasks = ['ae', 'ae-same', 'ae-diff', 'lm']
        self.warmup_task_steps = 1000
        self.main_task_steps = 7000
        self.warmup_ratios = [0.6, 0.1, 0.1, 0.2]
        self.main_ratios = [0.4, 0.3, 0.2, 0.1]
        self.hard_ratios = [0.2, 0.3, 0.4, 0.1]

        corpus_path = os.path.join(args.data_dir, 'collection', 'kilt-128-sample.jsonl')
        self.train_dataset: Dataset = load_dataset('json', data_files=corpus_path)['train']

        self.train_dataset.set_transform(self._transform_func)

        self.trainer: Optional[Trainer] = None
    
    def get_task(self):
        step = self.trainer.state.global_step
        if step < self.warmup_task_steps:
            ratios = self.warmup_ratios
        elif step >= self.warmup_task_steps and step < self.main_task_steps:
            ratios = self.main_ratios
        else:
            ratios = self.hard_ratios
        
        return random.choices(self.tasks, weights=ratios, k=1)[0]

    def _transform_func(self, examples: Dict[str, Any]) -> Dict[str, Any]:
        indices = examples["id"]
        tasks = [self.get_task() for _ in indices]
        
        passages = []
        prompts = []
        labels = []

        emb_token = "[MEM]" + "".join("<xRAG>" for i in range(self.args.mem_size)) + "[/MEM]"
        for idx, t in enumerate(tasks):
            if t == "ae":
                psg = [examples['text'][idx]]
                passages.append(psg)

                inst = random.choice(AEInstructions)
                inst = inst.format_map(dict(emb_token=emb_token))
                messages = [{'role': 'user', 'content': inst}]
                prompt_ids = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                )
                pid = self.tokenizer.encode(examples['text'][idx], add_special_tokens=False)
                d_input_ids = prompt_ids + pid + [self.tokenizer.eos_token_id]
                label_ids = copy(d_input_ids)
                label_ids[:len(prompt_ids)] = [-100] * len(prompt_ids)

                prompts.append(d_input_ids)
                labels.append(label_ids)

            elif t == "ae-same":
                chunk_a, chunk_b = examples['chunk_a'][idx], examples['chunk_b'][idx]
                if random.random() >= 0.8:
                    chunk_a, chunk_b = examples['chunk_b'][idx], examples['chunk_a'][idx]
                
                psg = [
                    chunk_a,
                    chunk_b
                ]
                passages.append(psg)
                
                inst = random.choice(AESameInstructions)
                inst = inst.format_map(dict(emb_token_1=emb_token, emb_token_2=emb_token))
                messages = [{'role': 'user', 'content': inst}]
                prompt_ids = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                )
                pid = self.tokenizer.encode(examples['text'][idx], add_special_tokens=False)
                d_input_ids = prompt_ids + pid + [self.tokenizer.eos_token_id]
                label_ids = copy(d_input_ids)
                label_ids[:len(prompt_ids)] = [-100] * len(prompt_ids)

                prompts.append(d_input_ids)
                labels.append(label_ids)

            elif t == "ae-diff":
                chunk_a = examples['chunk_a'][idx]
                p = copy(examples['text'])
                p.pop(idx)
                a = copy(examples['chunk_a'])
                a.pop(idx)
                b = copy(examples['chunk_b'])
                b.pop(idx)
                batch_samples = p + a + b
                chunk_b = random.choice(batch_samples)

                psg = [
                    chunk_a,
                    chunk_b
                ]
                passages.append(psg)

                inst = random.choice(AEDiffInstructions)
                inst = inst.format_map(dict(emb_token_1=emb_token, emb_token_2=emb_token))
                messages = [{'role': 'user', 'content': inst}]
                prompt_ids = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                )

                aid = self.tokenizer.encode(chunk_a, add_special_tokens=False)
                bid = self.tokenizer.encode(chunk_b, add_special_tokens=False)

                a_prefix_id = self.tokenizer.encode("Background 1: ", add_special_tokens=False)
                b_prefix_id = self.tokenizer.encode("\nBackground 2: ", add_special_tokens=False)

                answer_ids = a_prefix_id + aid + b_prefix_id + bid
                d_input_ids = prompt_ids + answer_ids + [self.tokenizer.eos_token_id]
                label_ids = copy(d_input_ids)
                label_ids[:len(prompt_ids)] = [-100] * len(prompt_ids)

                prompts.append(d_input_ids)
                labels.append(label_ids)

            elif t == "lm":
                chunk_a, chunk_b = examples['chunk_a'][idx], examples['chunk_b'][idx]
                psg = [chunk_a]
                passages.append(psg)

                inst = random.choice(LMInstructions)
                inst = inst.format_map(dict(emb_token=emb_token))
                messages = [{'role': 'user', 'content': inst}]
                prompt_ids = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                )
                pid = self.tokenizer.encode(chunk_b, add_special_tokens=False)
                d_input_ids = prompt_ids + pid
                label_ids = copy(d_input_ids)
                label_ids[:len(prompt_ids)] = [-100] * len(prompt_ids)

                prompts.append(d_input_ids)
                labels.append(label_ids)
            
            else:
                raise ValueError(f"Undefined task: {t}")

        return {
            "task": tasks,
            "passage": passages,
            "prompt": prompts,
            "label": labels,
        }