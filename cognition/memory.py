from collections import deque

MEMORY = deque(maxlen=50)

def add_memory(doc):
    MEMORY.append(doc)

def get_memory():
    return list(MEMORY)