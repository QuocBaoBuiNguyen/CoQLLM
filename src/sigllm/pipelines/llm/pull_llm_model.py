import os
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM

def pull_model(model_path="lmsys/vicuna-7b-v1.5", save_dir="./ckpt/llm/base"):
    """
    Download and save the Vicuna LLaMA-family model and tokenizer.

    Default matches the InstructBLIP paper backbone (Vicuna-7B-v1.5). Vicuna
    is an instruction-tuned LLaMA-2 derivative, which is required by this
    pipeline since the LLM is kept frozen (no LoRA) — only an instruction-
    tuned model can follow the Yes/No prompts at Stage 3.
    """
    print(f"Pulling model from {model_path}...")
    
    # Initialize tokenizer
    tokenizer = LlamaTokenizer.from_pretrained(model_path, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize model
    model = LlamaForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map={"": "cpu"}  # Load to CPU for saving
    )

    if model.generation_config is not None:
        model.generation_config.temperature = 1.0
        model.generation_config.top_p = 1.0

    # Create directory if it doesn't exist
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

    tokenizer = LlamaTokenizer.from_pretrained(model_dir, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    model = LlamaForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=dtype,
        device_map=device_map,
    )

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    generation_output = model.generate(input_ids=input_ids, max_new_tokens=max_new_tokens)
    print(tokenizer.decode(generation_output[0], skip_special_tokens=True))

if __name__ == "__main__":
    saved_dir = pull_model()
    smoke_test_model(saved_dir)
