import torch
import os

from transformers import AutoModelForCausalLM, AutoTokenizer, logging

logging.set_verbosity_error()

model_path = "./pragna-1b"
device = "cuda"

# Load Model
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype=torch.float16, 
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Prepare Inference
prompt = "The capital of India is"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

print("Generating...")
with torch.no_grad():
    outputs = model.generate(
        **inputs, 
        max_new_tokens=50,
        do_sample=True,      
        temperature=0.7,     
        top_k=50,            
        repetition_penalty=1.1 
    )

print("\n" + tokenizer.decode(outputs[0], skip_special_tokens=True))
