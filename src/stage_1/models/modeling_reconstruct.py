import os
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass

from transformers.activations import ACT2FN
from transformers.models.mistral.modeling_mistral import MistralRMSNorm
from transformers.modeling_outputs import ModelOutput
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    MistralForCausalLM,
)

from config import Arguments
from logger_config import logger
from utils import get_stop_ids


@dataclass
class PretrainOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    sparse_loss: Optional[torch.Tensor] = None
    dense_loss: Optional[torch.Tensor] = None


class Converter(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.M = config.M
        in_dim = int(config.retriever_hidden_size / config.M)
        out_dim = config.hidden_size

        self.w1 = nn.Linear(in_dim, out_dim, bias=False)
        self.w2 = nn.Linear(out_dim, out_dim, bias=False)

        self.attn = nn.MultiheadAttention(
            embed_dim=out_dim,
            num_heads=config.num_attention_heads,
            batch_first=True,
        )
        
        self.act = ACT2FN[config.hidden_act]
        self.norm = MistralRMSNorm(out_dim, eps=config.rms_norm_eps)
    
    def forward(self, x):
        B = x.shape[0]
        h = self.act(self.w1(x.reshape(B, self.M, -1)))
        attn_out, _ = self.attn(h, h, h) # (B, M, out_dim)
        h = h + attn_out
        h = self.norm(h)
        h = self.w2(h)
        return h

class xLLM(MistralForCausalLM):
    def __init__(self,config):
        super().__init__(config)
        if hasattr(config,"retriever_hidden_size") and config.retriever_hidden_size > 0: 
            self.converter = Converter(config)
        self.post_init()

class ReconstructModel(nn.Module):
    def __init__(
            self,
            args: Arguments,
            retriever: AutoModel,
            llm: xLLM,
            tokenizer
        ):
        super().__init__()

        self.args = args
        self.stopids = get_stop_ids(self.args)

        self.retriever: AutoModel = retriever.eval()
        self.llm: xLLM = llm

        self.M = self.args.mem_size
        self.tokenizer = tokenizer

        from trainers import PretrainingTrainer
        self.trainer: Optional[PretrainingTrainer] = None
    

    def forward(
            self,
            encoder: Dict[str, torch.Tensor]=None,
            decoder: Dict[str, torch.Tensor]=None
        ):
        with torch.no_grad():
            p_reps = self.retriever(
                input_ids=encoder['input_ids'],
                attention_mask=encoder['attention_mask']
            ).last_hidden_state[:, -1]

            llm_outputs = self.llm(**decoder, output_hidden_states=True)
            llm_hidden = llm_outputs.hidden_states[-1] # (B, L, H)
            llm_logits = llm_outputs.logits # (B, L, V)

        B, H = p_reps.shape
        V = llm_logits.shape[-1]

        converted_p_reps = self.llm.converter(p_reps) # (B, M, H)

        """ Dense Loss """
        llm_hidden = llm_hidden * decoder['attention_mask'].unsqueeze(-1)
        avg_llm_hidden = llm_hidden.sum(dim=1) / decoder['attention_mask'].sum(dim=1).unsqueeze(1) # (B, H)
        dense_loss = F.mse_loss(converted_p_reps.mean(dim=1), avg_llm_hidden, reduction='mean')
        
        """ Sparse loss """        
        lm_weight = self.llm.lm_head.weight
        lm_bias = self.llm.lm_head.bias if self.llm.lm_head.bias is not None else None
        c_logits = F.linear(converted_p_reps, lm_weight, lm_bias) # (B, M, V)

        target_ids = decoder['input_ids']
        pos_target_mask = torch.zeros((B, V), device=c_logits.device)
        pos_target_mask.scatter_(1, target_ids, 1.0)
        pos_target_mask[:, self.stopids] = 0.0

        llm_logits = llm_logits * decoder['attention_mask'].unsqueeze(-1)

        kl_loss = F.kl_div(
            F.log_softmax(c_logits.logsumexp(dim=1), dim=-1),
            F.softmax(llm_logits.logsumexp(dim=1), dim=-1),
            reduction="batchmean"
        )

        pos_logp = -F.log_softmax(c_logits, dim=-1).max(dim=1)[0] # (B, V)
        valid_pos_logp = pos_logp * pos_target_mask

        sum_per_sample = pos_target_mask.sum(dim=1) # (B, )
        valid_mask = sum_per_sample > 0
        valid_pos_logp = valid_pos_logp[valid_mask]
        sum_per_sample = sum_per_sample[valid_mask]
        batch_sum_pos_logp = (valid_pos_logp.sum(dim=1) / sum_per_sample).mean() # (B, )

        sparse_loss = kl_loss + batch_sum_pos_logp

        loss = sparse_loss + dense_loss

        return PretrainOutput(
            loss=loss, sparse_loss=sparse_loss, dense_loss=dense_loss
        )
    

    @classmethod
    def build(cls, args: Arguments, config, tokenizer):
        def init_converter_weights(module):
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

            if isinstance(module, MistralRMSNorm):
                nn.init.ones_(module.weight)

        logger.info(f'loading RETRIEVER weights from {args.retriever_name_or_path}')
        retriever = AutoModel.from_pretrained(
            args.retriever_name_or_path,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
        )
        retriever.requires_grad_(False)

        logger.info(f'loading LLM weights from {args.model_name_or_path}')
        backbone: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            config=config
        )
        backbone.config.retriever_hidden_size = retriever.config.hidden_size 
        backbone.config.M = args.mem_size
        converter: Converter = Converter(backbone.config)
        converter.apply(init_converter_weights)
        backbone.converter = converter
        for n, p in backbone.named_parameters():
            if 'converter' not in n:
                p.requires_grad_(False)
        
        model = cls(args=args, retriever=retriever, llm=backbone, tokenizer=tokenizer)
        
        return model
    
    def save(self, output_dir: str):
        self.llm.save_pretrained(os.path.join(output_dir))