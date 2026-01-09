import os
from typing import Optional, Dict

import torch
import torch.nn as nn

from peft import LoraConfig, get_peft_model

from transformers.activations import ACT2FN
from transformers.models.mistral.modeling_mistral import MistralRMSNorm
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    MistralForCausalLM
)

from config import Arguments
from logger_config import logger


class Converter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.M = config.M
        self.in_dim = int(config.retriever_hidden_size / config.M)
        self.out_dim = config.hidden_size

        self.w1 = nn.Linear(self.in_dim, self.out_dim, bias=False)
        self.w2 = nn.Linear(self.out_dim, self.out_dim, bias=False)

        self.attn = nn.MultiheadAttention(
            embed_dim=self.out_dim,
            num_heads=config.num_attention_heads,
            batch_first=True,
        )
        
        self.act = ACT2FN[config.hidden_act]
        self.norm = MistralRMSNorm(self.out_dim, eps=config.rms_norm_eps)
    
    def forward(self, x):
        B = x.shape[0]
        h = self.act(self.w1(x.reshape(B, self.M, -1)))
        attn_out, _ = self.attn(h, h, h) # (B, M, out_dim)
        h = h + attn_out
        h = self.norm(h)
        h = self.w2(h)
        return h.view(-1, self.out_dim)


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
            llm: xLLM
        ):
        super().__init__()

        self.args = args
        self.retriever: AutoModel = retriever

        self.llm: xLLM = llm
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name_or_path)
        
        from trainers import PretrainingTrainer
        self.trainer: Optional[PretrainingTrainer] = None
    

    def forward(
            self,
            encoder: Dict[str, torch.Tensor]=None,
            decoder: Dict[str, torch.Tensor]=None,
        ):
        B = encoder['input_ids'].shape[0]
        with torch.no_grad():
            p_reps = self.retriever(encoder['input_ids'], encoder['attention_mask']).last_hidden_state[:, -1]
        converted_p_reps = self.llm.converter(p_reps) # (B, M, H)

        inputs_embeds = self.llm.model.model.embed_tokens(decoder['input_ids'])
        inputs_embeds[decoder['input_ids']==32000] = converted_p_reps
        
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=decoder['attention_mask'],
            labels=decoder['labels']
        )

        loss = outputs.loss

        return (loss, ) + (outputs, )

    @classmethod
    def build(cls, args: Arguments, tokenizer):
        logger.info(f'loading retriever weights from {args.retriever_name_or_path}')
        retriever = AutoModel.from_pretrained(
            args.retriever_name_or_path,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
        )
        retriever.requires_grad_(False)

        logger.info(f'loading LLM weights from {args.model_name_or_path}')
        backbone = xLLM.from_pretrained(
            args.model_name_or_path,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
        )
        backbone.resize_token_embeddings(len(tokenizer))

        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", 
                "up_proj", "down_proj"],
            task_type="CAUSAL_LM"
        )
        
        llm = get_peft_model(backbone, lora_config)

        for name, param in llm.named_parameters():
            if "converter" in name:
                logger.info(f"{name}, {param.requires_grad}")
        
        model = cls(args=args, retriever=retriever, llm=llm)
        
        return model
    
    def save(self, output_dir: str):
        self.llm.save_pretrained(os.path.join(output_dir, 'adapter'))
