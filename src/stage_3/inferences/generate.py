import json
import os
import logging
import tqdm
import torch

from contextlib import nullcontext
from torch.utils.data import DataLoader
from functools import partial
from collections import defaultdict
from datasets import Dataset
from typing import Dict, List, Tuple
from datasets import load_dataset

from transformers.utils.logging import enable_explicit_format
from transformers.file_utils import PaddingStrategy
from transformers import (
    AutoTokenizer,
    BatchEncoding,
    HfArgumentParser,
    PreTrainedTokenizerFast,
    set_seed
)

from config import Arguments
from collators import CustomCollatorForInference
from logger_config import logger
from utils import move_to_cuda, _unpack_qp
from metrics import (
    get_exact_match_score,
    get_substring_match_score,
    eval_truthfulqa
)
from models import End2EndModelForInference

parser = HfArgumentParser(Arguments)
args, remaining_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)

def _common_setup(args: Arguments):
    if args.process_index > 0:
        logger.setLevel(logging.WARNING)
    enable_explicit_format()
    set_seed(args.seed)

def _get_cuda_result_save_path(worker_idx: int) -> str:
    return f'{args.output_dir}/{args.stage}_{args.mem_size}x_{args.file_name}_cuda:{worker_idx}.jsonl'

def _get_merged_result_save_path() -> str:
    return f'{args.output_dir}/{args.stage}_{args.mem_size}x_{args.file_name}.json'

def _query_transform_func(
        args: Arguments,
        tokenizer: PreTrainedTokenizerFast,
        ret_tokenizer: PreTrainedTokenizerFast,
        examples: Dict[str, List]
    ) -> BatchEncoding:
    
    all_prompts = []
    inst = "Refer to the background document and answer the questions."
    emb = "[MEM]" + "".join("<xRAG>" for i in range(args.mem_size)) + "[/MEM]"
    background = "\n".join(f"Background {i+1}: {emb}" for i in range(args.n_passages))
    
    for qry in examples['query']:
        messages = []

        content = f'{inst}\n\n{background}\n\nQuestion: {qry}\n'
        rag = {
            'role': 'user',
            'content': content
        }
        messages.append(rag)
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,    
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompt += " The answer is: "
        all_prompts.append(prompt)

    passage_lists = []
    for batch_psg in examples['passages']:
        passage_lists += batch_psg

    psg_batch_dict = ret_tokenizer(
        passage_lists,
        max_length=args.p_max_len,
        padding=PaddingStrategy.DO_NOT_PAD,
        truncation=True,
        return_attention_mask=False,
        add_special_tokens=False,
    )
    passages = ret_tokenizer.batch_decode(psg_batch_dict['input_ids'])
    
    merged_dict = defaultdict()
    merged_dict['passage'] = []

    step_size = args.n_passages
    for idx in range(0, len(passages), step_size):
        merged_dict['passage'].append(passages[idx:(idx + step_size)])
    
    merged_dict['prompt'] = all_prompts
    
    return merged_dict


@torch.no_grad()
def _worker_encode_queries(gpu_idx: int) -> Tuple:
    
    query_id_to_text = defaultdict()
    query_id_to_topk = defaultdict()

    test_data = os.path.join(args.data_dir, f'{args.file_name}.jsonl')
    test_dataset = load_dataset(
        'json',
        data_files=test_data
    )['train']

    for d in test_dataset:
        if 'query' in d.keys():
            query = d['query']
        elif 'question' in d.keys():
            query = d['question']
        else:
            raise ValueError("Neither query nor question exists!!!")
        
        if 'id' in d.keys():
            query_id = d['id']
        elif 'query_id' in d.keys():
            query_id = d['query_id']
        else:
            raise ValueError("Neither id nor query_id exists!!!")
        
        query_id_to_text[query_id] = query
        query_id_to_topk[query_id] = d['topk']['contents'][:args.n_passages]

    query_ids = sorted(list(query_id_to_text.keys()))
    queries = [query_id_to_text[query_id] for query_id in query_ids]
    passages = [query_id_to_topk[query_id] for query_id in query_ids]
    dataset = Dataset.from_dict({'query_id': query_ids,
                                 'query': queries,
                                 'passages': passages})
    dataset = dataset.shard(num_shards=torch.cuda.device_count(),
                            index=gpu_idx,
                            contiguous=True)

    query_ids = dataset['query_id']
    query_id_to_text = {qid: query_id_to_text[qid] for qid in query_ids}

    logger.info('GPU {} needs to process {} examples'.format(gpu_idx, len(dataset)))
    torch.cuda.set_device(gpu_idx)

    ret_tokenizer = AutoTokenizer.from_pretrained(args.retriever_name_or_path, padding_side='left')
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, padding_side='left')
    
    model: End2EndModelForInference = End2EndModelForInference.build(args, tokenizer)
    model.eval()
    model.cuda()
    
    dataset.set_transform(partial(_query_transform_func, args, tokenizer, ret_tokenizer))

    data_collator = CustomCollatorForInference(
        tokenizer=tokenizer,
        pad_to_multiple_of=8 if args.fp16 else None,
        args=args,
        ret_tokenizer=ret_tokenizer
    )
    data_loader = DataLoader(
        dataset,
        batch_size=args.per_device_eval_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.dataloader_num_workers,
        collate_fn=data_collator,
        pin_memory=True)

    out_path = _get_cuda_result_save_path(worker_idx=gpu_idx)
    with open(os.path.join(out_path), 'w+') as fout:
        for batch_idx, batch_dict in enumerate(tqdm.tqdm(data_loader, desc='answer generation')):
            batch_dict = move_to_cuda(batch_dict)

            target, encode = _unpack_qp(batch_dict)

            with torch.cuda.amp.autocast() if args.fp16 else nullcontext():
                with torch.no_grad():
                    outputs = model(target=target, encode=encode)
                    
                    generated = tokenizer.batch_decode(
                        outputs,
                        skip_special_tokens=True
                    )
                    
                    batch_query_ids = query_ids[
                        batch_idx*args.per_device_eval_batch_size:(batch_idx+1)*args.per_device_eval_batch_size
                    ]
                    for g, qid in zip(generated, batch_query_ids):
                        print(f"{qid}\n{g}\n")
                        temp = {
                            'id': qid,
                            'generated': g
                        }
                        fout.write(f'{json.dumps(temp, ensure_ascii=False)}\n')
                    
    logger.info('Done answer generation for worker {}'.format(gpu_idx))

    return

def _merge_generated(worker_cnt: int):
    ground_truth = defaultdict(list)
    with open(os.path.join(args.data_dir, f'{args.file_name}.jsonl'), 'r') as fin:
        for line in fin.readlines():
            d = json.loads(line)

            if 'id' in d.keys():
                query_id = d['id']
            elif 'query_id' in d.keys():
                query_id = d['query_id']
            else:
                raise ValueError("Neither id nor query_id exists!!!")

            ground_truth[query_id] = d['answers']

    merged_data = defaultdict(dict)
    for worker_idx in range(worker_cnt):
        path = _get_cuda_result_save_path(worker_idx)
        
        with open(path, 'r') as fin:
            for line in fin.readlines():
                d = json.loads(line)
                merged_data[d['id']]['ground_truth'] = ground_truth[d['id']]
                merged_data[d['id']]['generated'] = d['generated'].strip()

    out_path = _get_merged_result_save_path()
    with open(out_path, 'w') as fout:
        json.dump(merged_data, fout, ensure_ascii=False, indent='\t')
    logger.info('Merge done: save {} generations to {}'.format(len(merged_data), out_path))

    # do some cleanup
    if len(merged_data) != 0:
        for worker_idx in range(worker_cnt):
            path = _get_cuda_result_save_path(worker_idx)
            os.remove(path)


def calculate_score():
    saved_path = _get_merged_result_save_path()
    with open(os.path.join(saved_path), 'r') as fin:
        data = json.load(fin)

    all_preds = []
    all_labels = []
    for qid, v in data.items():
        
        all_preds.append(v['generated'])
        all_labels.append(v['ground_truth'])
    
    em_score, em_score_per_sample = get_exact_match_score(all_preds, all_labels)
    m_score, m_score_per_sample = get_substring_match_score(all_preds, all_labels)
    f1_score, f1_score_per_sample, rl_scores, rl_score_per_sample = eval_truthfulqa(all_preds, all_labels)
    logger.info(f"EM score: {em_score}")
    logger.info(f"MATCH score: {m_score}")
    logger.info(f"F1 score: {f1_score}")
    logger.info(f"RL score: {rl_scores}")

    with open(os.path.join(f'{args.output_dir}', f'{args.stage}_{args.mem_size}x_{args.file_name}.txt'), 'a') as fout:
        fout.write(f"EM score: {em_score}\tMATCH score: {m_score}\tF1 score: {f1_score}\tRL score: {rl_scores}\n")


@torch.no_grad()
def _worker_batch_search(gpu_idx: int):
    _worker_encode_queries(gpu_idx)

def _batch_search_queries():
    _common_setup(args)

    logger.info('Args={}'.format(str(args)))
    gpu_count = torch.cuda.device_count()
    if gpu_count == 0:
        logger.error('No gpu available')
        return

    logger.info('Use {} gpus'.format(gpu_count))
    torch.multiprocessing.spawn(_worker_batch_search, args=(), nprocs=gpu_count)
    logger.info('Done batch search queries')

    _merge_generated(gpu_count)
    calculate_score()


if __name__ == '__main__':
    _batch_search_queries()
