import os

def load_llm():
    name = os.getenv("TECTO_LLM", "none")

    if name == "phi3":
        from .phi3 import Phi3LLM
        return Phi3LLM()

    if name == "none":
        return None

    raise ValueError(f"Unknown LLM: {name}")