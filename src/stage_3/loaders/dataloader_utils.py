from typing import Dict, List

def group_psg_ids(
        examples: Dict[str, List],
        n_passages: int,
    ) -> List[int]:

    input_docs: List[str] = []
    topk: List[Dict[str, List]] = examples['topk']
    for _, ex in enumerate(topk):
        docs = ex['contents'][:n_passages]
        input_docs += docs
        
    return input_docs