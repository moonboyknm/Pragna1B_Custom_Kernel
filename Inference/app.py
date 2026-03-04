import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Model setup 
model_path = "./pragna-1b"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    dtype=torch.float16, 
    device_map="auto"
)

def predict(message, history):
    inputs = tokenizer(message, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            repetition_penalty=1.1
        )
    
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return full_text[len(message):].strip()

demo = gr.ChatInterface(
    fn=predict,
    title="Pragna-1B Chat",
    description="Local inference running on RTX 2050",
)

if __name__ == "__main__":
    demo.launch()
