---
base_model: meta-llama/Llama-3.2-1B-Instruct
library_name: transformers
model_name: small-talk-1.3
tags:
- generated_from_trainer
- trl
- dpo
---

## Model history
- V1 [https://huggingface.co/Luke-griggs/small-talk-1.1](https://huggingface.co/Luke-griggs/small-talk-1.1)
- V2 [https://huggingface.co/Luke-griggs/small-talk-1.2](https://huggingface.co/Luke-griggs/small-talk-1.2)


# Model Card for small-talk-1.3

This model is a fine-tuned version of [meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct).
It has been trained using [TRL](https://github.com/huggingface/trl).

# Quick start

## Install dependencies
```bash
!pip -q install transformers accelerate safetensors bitsAndBytes
```

## Load the model and tokenizer
```python
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
repo = "Luke-griggs/small-talk-1.3"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,  # or float16 if needed
)
tokenizer = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(
    repo,
    torch_dtype=torch.float16,
    device_map="auto"
    )
model.eval()
```

## Generate a response
```python
chat = [
    {"role": "system", 
     "content": """
You are a helpful, polite, and friendly assistant. Answer questions to the best of your ability.
If you don't know something, be honest and say so. Keep responses clear and concise."},
"""},
     {"role": "user", "content": "What's your favorite thing to do?"}
]
prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )
full_text = tokenizer.decode(output[0], skip_special_tokens=True)
prompt_text = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
print(full_text[len(prompt_text):])
```

This model was trained with DPO, a method introduced in [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://huggingface.co/papers/2305.18290).

### Framework versions

- TRL: 0.19.0
- Transformers: 4.53.1
- Pytorch: 2.4.1+cu124
- Datasets: 3.6.0
- Tokenizers: 0.21.2
