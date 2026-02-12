import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "./pragna-1b"

print(f"Loading model from {model_path}...")

# 1. Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

# 2. Load Model 
# Using float16
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    torch_dtype=torch.float16, 
    device_map="cuda",
    local_files_only=True
)

print(f"Success! Model loaded on {torch.cuda.get_device_name(0)}")
print(f"Memory Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
