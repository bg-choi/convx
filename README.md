# ConvX: A Lightweight Converter to Bridge Indexed Dense Representations and Large Language Models for Retrieval-Augmented Generation

## Training and Inference Environments
I recommend to install requirements with:
```
pip install requirements.txt
```

# For Training
For pretraining, we incorporated dmrau/kilt-128 from huggingface but merged both chunk to construct an entire passage.
The corpus is saved in jsonline file which contains keys 'text' for the merged passage.

## Stage 1: Pretraining converter
run stage 1:
```
bash scripts/stage1.sh stage_1 $output_dir $backbone_llm $mem_size
```

## Stage 2: Pretraining the target LLM
run stage 2:
```
bash scripts/stage2.sh stage_2 $output_dir $backbone_llm $mem_size
```

## Stage 3: Self-Distillation
For self-distillation, we adopted dmrau/multi_qa dataset from huggingface and retrieved top-5 passages from kilt-128 using SPLADE-v3.
Then, we generated answers of length is 100 using the vanilla LLM (backbone) and used them as supervision signals.
Samples for which generated answers do not contain original golden labels are filtered out.

run stage 3:
```
bash scripts/stage3.sh stage_3 $output_dir $path_to_stage2_checkpoint $mem_size $n_passages_per_sample
```

# For Inference
## Inference
```
bash scripts/inference.sh stage_3 $path_to_stage3_checkpoint $eval_data $n_passages_per_sample
```
