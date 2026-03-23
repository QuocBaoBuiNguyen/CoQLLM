import os
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM

def pull_model(model_path="openlm-research/open_llama_3b", save_dir="./content/ckpt/llm/base"):
    """
    Download and save the base LLaMA model and tokenizer.
    """
    print(f"Pulling model from {model_path}...")
    
    # Initialize tokenizer
    tokenizer = LlamaTokenizer.from_pretrained(model_path, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize model
    model = LlamaForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="cpu" # Load to CPU for saving
    )
    
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Saving model and tokenizer to {save_dir}...")
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print("Done!")

if __name__ == "__main__":
    # Default to open_llama_3b as seen in the existing snippet
    pull_model()

prompt = 'Q: What is the largest animal?\nA:'
input_ids = tokenizer(prompt, return_tensors="pt").input_ids

generation_output = model.generate(
    input_ids=input_ids.cuda(), max_new_tokens=32
)
print(tokenizer.decode(generation_output[0]))

pretrained_path = "/content/open_llama_3b"
model.save_pretrained(pretrained_path)
tokenizer.save_pretrained(pretrained_path)