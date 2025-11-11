# main.py
import argparse
from tqdm import tqdm
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from bqer import (
    calibrate_bqer_Ks_from_loader,
    apply_bqer_wrapper,
)
from evaluator import eval_ppl

# -------------------- ONE-CALL PIPELINE --------------------------------------

@torch.no_grad()
def calibrate_and_wrap_bqer(
    fp_model: nn.Module,
    q_model: nn.Module,
    dataloader,
    device: str = "cuda",
    max_batches: int = 64,
    lambda1: float = 1e-6,
    window_m: int = 0,
    clip: float = 0.5,
    alpha: float = 0.5,
    group_size: int = -1,      # NEW
):
    Ks_cur, Ks_prev = calibrate_bqer_Ks_from_loader(
        fp_model, q_model, dataloader,
        device=device, max_batches=max_batches, lambda1=lambda1, clip=clip, group_size=group_size,
    )
    apply_bqer_wrapper(q_model, Ks_cur, Ks_prev,
        window_m=window_m, alpha=alpha, group_size=group_size)
    return q_model

# -------------------- DATA LOADING -----------------------------------------

def prepare_dataloader(encodings, max_seqlen=2048, num_chunks=128):
    input_ids = encodings["input_ids"][0]
    attention_mask = encodings["attention_mask"][0] if "attention_mask" in encodings else torch.ones_like(input_ids)
    chunks = []
    for i in range(0, input_ids.shape[0] - max_seqlen + 1, max_seqlen):
        chunk_input_ids = input_ids[i:i+max_seqlen].unsqueeze(0)
        chunk_attention_mask = attention_mask[i:i+max_seqlen].unsqueeze(0)
        chunks.append({"input_ids": chunk_input_ids, "attention_mask": chunk_attention_mask})
    return chunks[:num_chunks]

# -------------------- EVALUATION -----------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", "-m",type=str, default="Llama-3.2-3B")
    parser.add_argument("--seqlen", "-s", type=int, default=2048)
    parser.add_argument("--num_chunks", "-n", type=int, default=256)
    parser.add_argument("--lambda1", "-l", type=float, default=1e-5)
    parser.add_argument("--window_m", "-w", type=int, default=0)
    parser.add_argument("--clip", "-c", type=float, default=0.5)
    parser.add_argument("--alpha", "-a", type=float, default=0.5)
    parser.add_argument("--group_size", "-g", type=int, default=512)
    args = parser.parse_args()

    # Load models
    fp_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B", dtype=torch.bfloat16, device_map="auto")
    q_model = AutoModelForCausalLM.from_pretrained("/data/hf_cache/hub/Llama-3.2-3B-2bits-g64", dtype=torch.bfloat16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")

    dataset = load_dataset("allenai/c4", data_files={"train": "en/c4-train.00000-of-01024.json.gz"}, split="train[:1%]")
    encodings = tokenizer("\n\n".join([x["text"] for x in dataset if len(x["text"].strip()) > 0]), return_tensors="pt")
    dataloader = prepare_dataloader(encodings, max_seqlen=args.seqlen, num_chunks=args.num_chunks)

    print(f"Given config: " +
          f"[lambda1={args.lambda1}, window_m={args.window_m}, clip={args.clip}, alpha={args.alpha}, group_size={args.group_size}]")

    print(f"Evaluating perplexity of original model")
    ppl_test = eval_ppl(q_model, tokenizer)
    print(f"wikitext perplexity of original model: {ppl_test}")

    print(f"Calibrating and wrapping BQER...")
    q_model = calibrate_and_wrap_bqer(
        fp_model, q_model, dataloader, device="cuda", max_batches=args.num_chunks,
        lambda1=args.lambda1, window_m=args.window_m, clip=args.clip, alpha=args.alpha, group_size=args.group_size
    )
    del fp_model
    torch.cuda.empty_cache()

    print(f"Evaluating perplexity of BQER wrapped model")
    ppl_test = eval_ppl(q_model, tokenizer)
    print(f"wikitext perplexity of BQER wrapped model: {ppl_test}")

# -------------------- EXAMPLE USAGE -----------------------------------------
# from transformers import AutoModelForCausalLM
# fp_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-8B", torch_dtype=torch.bfloat16)
# q_model = AutoModelForCausalLM.from_pretrained("your/quantized-llama", torch_dtype=torch.bfloat16)
#
# # dataloader should yield dicts ready for the model (**batch), e.g.:
# # {'input_ids': LongTensor[B,T], 'attention_mask': LongTensor[B,T], ...}
# # Keep max_batches modest (e.g., 20~100) for a quick fit.
#
# q_model, Ks_cur, Ks_prev = calibrate_and_wrap_bqer(
#     fp_model, q_model, dataloader,
#     device="cuda", max_batches=50, lambda1=1e-6, window_m=1
# )
#
# # At inference: before each new prompt
# for layer in q_model.model.layers:
#     if isinstance(layer, BQERDecoderLayer):
#         layer.reset_cache()