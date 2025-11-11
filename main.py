# main.py
import os
import argparse
from tqdm import tqdm
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from bqer import (
    calibrate_bqer_Ks,
    apply_bqer,
)
from evaluator import eval_ppl

# -------------------- ONE-CALL PIPELINE --------------------------------------

@torch.no_grad()
def calibrate_and_wrap_bqer(
    fp_model: nn.Module,
    q_model: nn.Module,
    dataloader,
    device: str = "cuda",
    args: argparse.Namespace = None,
):
    Ks_cur, Ks_prev = calibrate_bqer_Ks(
        fp_model, q_model, dataloader, device=device,
        max_batches=args.num_chunks, lambda1=args.lambda1, clip=args.clip, group_size=args.group_size,
    )
    apply_bqer(q_model, Ks_cur, Ks_prev,
        window_m=args.window_m, alpha=args.alpha, group_size=args.group_size)
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
    # model
    parser.add_argument("--model_name", "-m",type=str, default="meta-llama/Llama-3.2-3B")
    parser.add_argument("--qbits", "-q", type=int, default=2)
    # calibration data
    parser.add_argument("--seqlen", "-s", type=int, default=2048)
    parser.add_argument("--num_chunks", "-n", type=int, default=256)
    # BQER parameters
    parser.add_argument("--bsz", "-b", type=int, default=32)
    parser.add_argument("--lambda1", "-l", type=float, default=1e-5)
    parser.add_argument("--window_m", "-w", type=int, default=0)
    parser.add_argument("--clip", "-c", type=float, default=0.5)
    parser.add_argument("--alpha", "-a", type=float, default=0.5)
    parser.add_argument("--group_size", "-g", type=int, default=512)
    args = parser.parse_args()

    HF_HOME = os.getenv("HF_HOME", "/data/hf_cache")
    # Load models
    fp_model = AutoModelForCausalLM.from_pretrained(args.model_name, dtype=torch.bfloat16, device_map="auto")
    q_model = AutoModelForCausalLM.from_pretrained(f"{HF_HOME}/hub/{args.model_name.split('/')[-1]}-{args.qbits}bits-g64", 
                                                    dtype=torch.bfloat16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    dataset = load_dataset("allenai/c4", data_files={"train": "en/c4-train.00000-of-01024.json.gz"}, split="train[:1%]")
    encodings = tokenizer("\n\n".join([x["text"] for x in dataset if len(x["text"].strip()) > 0]), return_tensors="pt")
    dataloader = prepare_dataloader(encodings, max_seqlen=args.seqlen, num_chunks=args.num_chunks)

    print(f"Given config: " +
          f"[lambda1={args.lambda1}, window_m={args.window_m}, clip={args.clip}, alpha={args.alpha}, group_size={args.group_size}]")

    # print(f"Evaluating perplexity of original model")
    # ppl_test = eval_ppl(q_model, tokenizer)
    # print(f"wikitext perplexity of original model: {ppl_test}")

    print(f"Calibrating and wrapping BQER...")
    q_model = calibrate_and_wrap_bqer(
        fp_model, q_model, dataloader, device="cuda", args=args)
    del fp_model
    torch.cuda.empty_cache()

    print(f"Evaluating perplexity of BQER wrapped model")
    ppl_test = eval_ppl(q_model, tokenizer)
    print(f"wikitext perplexity of BQER wrapped model: {ppl_test}")
