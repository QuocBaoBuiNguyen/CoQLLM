import argparse
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def pull_model(model_path="Qwen/Qwen2-7B", save_dir="./ckpt/llm/qwen2-7b-base"):
    """Download and save an HF causal LM + tokenizer in fp16.

    Uses ``AutoTokenizer`` / ``AutoModelForCausalLM`` so any LLaMA-family,
    Qwen2-family or other compatible backbone resolves automatically. The
    default targets Qwen2-7B-Base (branch `feat/swap-llm-qwen2`); pass
    ``model_path="lmsys/vicuna-7b-v1.5"``, ``save_dir="./ckpt/llm/base"`` to
    pull the original Vicuna backbone instead.
    """
    print(f"Pulling model from {model_path}...")

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map={"": "cpu"},
        trust_remote_code=True,
    )

    if model.generation_config is not None:
        model.generation_config.temperature = 1.0
        model.generation_config.top_p = 1.0

    os.makedirs(save_dir, exist_ok=True)

    print(f"Saving model and tokenizer to {save_dir}...")
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print("Done!")
    return save_dir


def smoke_test_model(model_dir, prompt="Q: What is the largest animal?\nA:", max_new_tokens=32):
    """Load the saved model and run one quick generation test."""

    use_cuda = torch.cuda.is_available()
    device_map = {"": 0} if use_cuda else {"": "cpu"}
    dtype = torch.float16 if use_cuda else torch.float32

    print(f"Running smoke test from {model_dir} on {'cuda' if use_cuda else 'cpu'}...")

    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    generation_output = model.generate(input_ids=input_ids, max_new_tokens=max_new_tokens)
    print(tokenizer.decode(generation_output[0], skip_special_tokens=True))

if __name__ == "__main__":
    # Default = Vicuna-7B-v1.5 to match the active CoLLM-comparison config
    # (model.llm_model: .../ckpt/llm/vicuna-7b-v1.5). Run from the repo root so
    # ./ckpt resolves to /content/CoQLLM/ckpt. To pull the Qwen base instead:
    #   python pull_llm_model.py --model-path Qwen/Qwen2-7B --save-dir ./ckpt/llm/qwen2-7b-base
    parser = argparse.ArgumentParser(description="Download an HF causal LM + tokenizer in fp16.")
    parser.add_argument("--model-path", default="lmsys/vicuna-7b-v1.5", help="HF repo id to pull.")
    parser.add_argument("--save-dir", default="./ckpt/llm/vicuna-7b-v1.5", help="Local dir to save into (must match model.llm_model in the config).")
    parser.add_argument("--skip-smoke-test", action="store_true", help="Skip the post-download generation check.")
    args = parser.parse_args()

    saved_dir = pull_model(model_path=args.model_path, save_dir=args.save_dir)
    if not args.skip_smoke_test:
        smoke_test_model(saved_dir)
