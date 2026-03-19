import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

_model = None
_tokenizer = None

def load():
    global _model, _tokenizer

    if _model is not None:
        return _model, _tokenizer

    print("[LLM] loading local model...")

    try:
        model_name = "microsoft/Phi-3-mini-4k-instruct"

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        _tokenizer = AutoTokenizer.from_pretrained(model_name)

        _model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            quantization_config=bnb,
            torch_dtype=torch.float16,
        )

        return _model, _tokenizer

    except Exception as e:
        print("[LLM LOAD FAILED]", e)
        raise RuntimeError("LLM not usable")
    


def generate(prompt: str):
    model, tokenizer = load()

    messages = [
        {"role": "system", "content": "Rewrite economic analysis clearly and concisely."},
        {"role": "user", "content": prompt},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=200,          # ↑ increase
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = out[0][inputs["input_ids"].shape[-1]:]

    result = tokenizer.decode(generated, skip_special_tokens=True)

    # ----------------------------
    # HARD GUARD (IMPORTANT)
    # ----------------------------
    if len(result.strip()) < 10:
        print("[LLM WARNING] Too short → fallback")
        return prompt  # or original answer

    return result