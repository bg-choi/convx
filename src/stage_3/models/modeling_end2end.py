import os
from typing import Optional, Dict

import torch
import torch.nn as nn
from peft import (
    LoraConfig,
    get_peft_model
)

from transformers.activations import ACT2FN
from transformers import  (
    AutoModel,
    PreTrainedTokenizer,
    MistralForCausalLM,
)
from transformers.models.mistral.modeling_mistral import MistralRMSNorm

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
        x = x.reshape(B, self.M, -1)
        h = self.act(self.w1(x))
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


class End2EndModel(nn.Module):
    def __init__(
            self,
            args: Arguments,
            retriever: AutoModel,
            llm: xLLM,
        ):
        super().__init__()

        self.args = args
        self.retriever: AutoModel = retriever.eval()

        self.llm: xLLM = llm

        from trainers import CustomTrainer
        self.trainer: Optional[CustomTrainer] = None

    def forward(
            self,
            target: Dict[str, torch.Tensor]=None,
            encode: Dict[str, torch.Tensor]=None,
        ):
        with torch.no_grad():
            p_reps = self.retriever(
                input_ids=encode['input_ids'],
                attention_mask=encode['attention_mask'],
            ).last_hidden_state[:, -1]
        p_reps = self.llm.converter(p_reps)
        
        """ Put memory slots """
        qa_inputs_embeds = self.llm.model.model.embed_tokens(target['input_ids'])
        qa_inputs_embeds[target['input_ids']==32000] = p_reps

        qa_outputs = self.llm(
            inputs_embeds=qa_inputs_embeds,
            attention_mask=target['attention_mask'],
            labels=target['labels'],
            output_hidden_states=True,
            return_dict=True
        )

        return qa_outputs

    @classmethod
    def build(cls, args: Arguments):
        logger.info(f'loading RETRIEVER weights from {args.retriever_name_or_path}')
        retriever = AutoModel.from_pretrained(
            args.retriever_name_or_path,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
        )
        retriever.requires_grad_(False)

        logger.info(f'loading LLM weights from {args.model_name_or_path}')
        pretrained = xLLM.from_pretrained(
            args.model_name_or_path,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
        )

        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", 
                "up_proj", "down_proj"],
        )
        llm = get_peft_model(pretrained, lora_config)

        model = cls(args=args, retriever=retriever, llm=llm)

        for name, param in llm.named_parameters():
            if "converter" in name:
                logger.info(f"{name}, {param.requires_grad}")

        return model
    
    def save(self, output_dir: str):
        self.llm.save_pretrained(os.path.join(output_dir, 'adapter'))


class End2EndModelForInference(End2EndModel):
    def __init__(
            self, args: Arguments,
            retriever: AutoModel,
            llm: xLLM,
            tokenizer: PreTrainedTokenizer,
        ):
        nn.Module.__init__(self)
        self.args = args
        self.retriever: AutoModel = retriever.eval()
        self.tokenizer: PreTrainedTokenizer = tokenizer

        self.llm: xLLM = llm.eval()
    
    @torch.no_grad()
    def forward(
            self,
            target: Dict[str, torch.Tensor] = None,    
            encode: Dict[str, torch.Tensor]=None,
        ):
        """ Encod passages """
        p_reps = self.retriever(
            input_ids=encode['input_ids'],
            attention_mask=encode['attention_mask']
        ).last_hidden_state[:, -1]
        p_reps = self.llm.converter(p_reps)
        
        """ Put memory slots """
        target_inputs_embeds = self.llm.model.embed_tokens(target['input_ids'])
        target_inputs_embeds[target['input_ids']==32000] = p_reps
        qa_outputs = self.llm.generate(
            inputs_embeds=target_inputs_embeds,
            attention_mask=target['attention_mask'],
            max_new_tokens=self.args.max_new_tokens,
            do_sample=False,
            use_cache=True,
        )

        return qa_outputs
    
    @classmethod
    def build(cls, args: Arguments, tokenizer: PreTrainedTokenizer):
        logger.info(f'loading RETRIEVER weights from {args.retriever_name_or_path}')
        retriever = AutoModel.from_pretrained(
            args.retriever_name_or_path,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
        )
        retriever.requires_grad_(False)

        logger.info(f'loading fine-tuned LLM from {args.model_name_or_path}')
        llm = xLLM.from_pretrained(
            args.model_name_or_path,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
        )
        llm.requires_grad_(False)

        model = cls(args=args, retriever=retriever, llm=llm, tokenizer=tokenizer)

        return model
