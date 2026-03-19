from .base import BaseLLM

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


class Phi3LLM(BaseLLM):
    def __init__(self):
        self.model = None
        self.tokenizer = None

    def load(self):
        if self.model is not None:
            return self.model, self.tokenizer

        print("[LLM] loading Phi-3...")

        try:
            model_name = "microsoft/Phi-3-mini-4k-instruct"

            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                quantization_config=bnb,
                torch_dtype=torch.float16,
            )

            return self.model, self.tokenizer

        except Exception as e:
            print("[LLM LOAD FAILED]", e)
            raise RuntimeError("LLM not usable")

    def generate(self, messages, mode: str = "system") -> str:
        model, tokenizer = self.load()

        # ----------------------------
        # APPLY CHAT TEMPLATE
        # ----------------------------
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        # ----------------------------
        # GENERATION CONFIG
        # ----------------------------
        do_sample = (mode != "system")

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=do_sample,
                temperature=0.0 if not do_sample else 0.4,
                top_p=0.9 if do_sample else 1.0,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated = out[0][inputs["input_ids"].shape[-1]:]

        result = tokenizer.decode(generated, skip_special_tokens=True)

        if len(result.strip()) < 5:
            return "I don't have enough context to answer that."

        return result.strip()