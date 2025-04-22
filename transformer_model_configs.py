"""
Pre-defined configurations for various transformer models.

TODO: Currently, these are LLM-generated. They are accurate for older, public models,
but recent closed models are likely not accurate. Put more thought into refining these.
"""

def gpt2_config():
    """GPT-2 (124M) configuration."""
    return {
        "vocab_size": 50257,
        "seq_length": 1024,
        "embedding_dim": 768,
        "num_heads": 12,
        "ffn_dim": 3072,
        "num_layers": 12,
        "attention_type": "MHA"
    }

def gpt2_medium_config():
    """GPT-2 Medium (355M) configuration."""
    return {
        "vocab_size": 50257,
        "seq_length": 1024,
        "embedding_dim": 1024,
        "num_heads": 16,
        "ffn_dim": 4096,
        "num_layers": 24,
        "attention_type": "MHA"
    }

def gpt2_large_config():
    """GPT-2 Large (774M) configuration."""
    return {
        "vocab_size": 50257,
        "seq_length": 1024,
        "embedding_dim": 1280,
        "num_heads": 20,
        "ffn_dim": 5120,
        "num_layers": 36,
        "attention_type": "MHA"
    }

def gpt2_xl_config():
    """GPT-2 XL (1.5B) configuration."""
    return {
        "vocab_size": 50257,
        "seq_length": 1024,
        "embedding_dim": 1600,
        "num_heads": 25,
        "ffn_dim": 6400,
        "num_layers": 48,
        "attention_type": "MHA"
    }

def gpt3_config():
    """GPT-3 (175B) configuration."""
    return {
        "vocab_size": 50257,
        "seq_length": 2048,
        "embedding_dim": 12288,
        "num_heads": 96,
        "ffn_dim": 49152,
        "num_layers": 96,
        "attention_type": "MHA"
    }

def gpt4_estimate_config():
    """GPT-4 (estimated) configuration."""
    return {
        "vocab_size": 100000,
        "seq_length": 8192,
        "embedding_dim": 25600,
        "num_heads": 160,
        "ffn_dim": 102400,
        "num_layers": 120,
        "attention_type": "MHA"
    }

def llama2_7b_config():
    """Llama 2 (7B) configuration."""
    return {
        "vocab_size": 32000,
        "seq_length": 4096,
        "embedding_dim": 4096,
        "num_heads": 32,
        "ffn_dim": 11008,
        "num_layers": 32,
        "attention_type": "MHA",
    }

def llama2_13b_config():
    """Llama 2 (13B) configuration."""
    return {
        "vocab_size": 32000,
        "seq_length": 4096,
        "embedding_dim": 5120,
        "num_heads": 40,
        "ffn_dim": 13824,
        "num_layers": 40,
        "attention_type": "MHA",
    }

def llama2_70b_config():
    """Llama 2 (70B) configuration."""
    return {
        "vocab_size": 32000,
        "seq_length": 4096,
        "embedding_dim": 8192,
        "num_heads": 64,
        "ffn_dim": 22016,
        "num_layers": 80,
        "attention_type": "GQA",
        "group_size": 8
    }

def llama3_8b_config():
    """Llama 3 (8B) configuration."""
    return {
        "vocab_size": 128000,
        "seq_length": 8192,
        "embedding_dim": 4096,
        "num_heads": 32,
        "ffn_dim": 14336,
        "num_layers": 32,
        "attention_type": "GQA",
        "group_size": 4
    }

def llama3_70b_config():
    """Llama 3 (70B) configuration."""
    return {
        "vocab_size": 128000,
        "seq_length": 8192,
        "embedding_dim": 8192,
        "num_heads": 64,
        "ffn_dim": 28672,
        "num_layers": 80,
        "attention_type": "GQA",
        "group_size": 8
    }

def llama4_8b_config():
    """Llama 4 (8B) estimated configuration."""
    return {
        "vocab_size": 128000,
        "seq_length": 128000,
        "embedding_dim": 4096,
        "num_heads": 32,
        "ffn_dim": 14336,
        "num_layers": 32,
        "attention_type": "GQA",
        "group_size": 8
    }

def llama4_70b_config():
    """Llama 4 (70B) estimated configuration."""
    return {
        "vocab_size": 128000,
        "seq_length": 128000,
        "embedding_dim": 8192,
        "num_heads": 64,
        "ffn_dim": 28672,
        "num_layers": 80,
        "attention_type": "GQA",
        "group_size": 8
    }

def tinyllama_1_1b_config():
    """TinyLlama (1.1B) configuration."""
    return {
        "vocab_size": 32000,
        "seq_length": 2048,
        "embedding_dim": 2048,
        "num_heads": 32,
        "ffn_dim": 5632,
        "num_layers": 22,
        "attention_type": "GQA",
        "group_size": 8
    }

def mixtral_8x7b_config():
    """Mixtral 8x7B configuration."""
    return {
        "vocab_size": 32000,
        "seq_length": 32768,
        "embedding_dim": 4096,
        "num_heads": 32,
        "ffn_dim": 14336,
        "num_layers": 32,
        "attention_type": "GQA",
        "group_size": 8,
        "moe_enabled": True,
        "num_experts": 8,
        "active_experts": 2
    }

def mixtral_8x22b_config():
    """Mixtral 8x22B (estimated) configuration."""
    return {
        "vocab_size": 32000,
        "seq_length": 8192,
        "embedding_dim": 6144,
        "num_heads": 48,
        "ffn_dim": 20480,
        "num_layers": 48,
        "attention_type": "GQA",
        "group_size": 4,
        "moe_enabled": True,
        "num_experts": 8,
        "active_experts": 2
    }

def deepseek_7b_config():
    """DeepSeek 7B configuration."""
    return {
        "vocab_size": 100672,
        "seq_length": 4096,
        "embedding_dim": 4096,
        "num_heads": 32,
        "ffn_dim": 11008,
        "num_layers": 32,
        "attention_type": "MHA"
    }

def deepseek_67b_config():
    """DeepSeek 67B configuration."""
    return {
        "vocab_size": 100672,
        "seq_length": 4096,
        "embedding_dim": 8192,
        "num_heads": 64,
        "ffn_dim": 22016,
        "num_layers": 80,
        "attention_type": "MHA"
    }

def deepseek_coder_v2_config():
    """DeepSeek Coder V2 (16B) estimated configuration."""
    return {
        "vocab_size": 100672,
        "seq_length": 16384,
        "embedding_dim": 5120,
        "num_heads": 40,
        "ffn_dim": 20480,
        "num_layers": 40,
        "attention_type": "MHA"
    }

def falcon_7b_config():
    """Falcon 7B configuration."""
    return {
        "vocab_size": 65024,
        "seq_length": 2048,
        "embedding_dim": 4544,
        "num_heads": 71,
        "ffn_dim": 11712,
        "num_layers": 32,
        "attention_type": "MQA"  # Multi-Query Attention
    }

def falcon_40b_config():
    """Falcon 40B configuration."""
    return {
        "vocab_size": 65024,
        "seq_length": 2048,
        "embedding_dim": 8192,
        "num_heads": 64,
        "ffn_dim": 22016,
        "num_layers": 60,
        "attention_type": "MQA"  # Multi-Query Attention (1 KV head)
    }

def phi_2_config():
    """Phi-2 (2.7B) configuration."""
    return {
        "vocab_size": 51200,
        "seq_length": 2048,
        "embedding_dim": 2560,
        "num_heads": 32,
        "ffn_dim": 10240,
        "num_layers": 32,
        "attention_type": "MHA"
    }

def phi_3_mini_config():
    """Phi-3 Mini (3.8B) configuration."""
    return {
        "vocab_size": 32064,
        "seq_length": 4096,
        "embedding_dim": 3072,
        "num_heads": 32,
        "ffn_dim": 8192,
        "num_layers": 32,
        "attention_type": "MHA"
    }

def phi_3_medium_config():
    """Phi-3 Medium (14B) configuration."""
    return {
        "vocab_size": 100000,
        "seq_length": 4096,
        "embedding_dim": 5120,
        "num_heads": 40,
        "ffn_dim": 14336,
        "num_layers": 40,
        "attention_type": "GQA",
        "group_size": 4
    }

def gemma_2b_config():
    """Gemma 2B configuration."""
    return {
        "vocab_size": 256000,
        "seq_length": 8192,
        "embedding_dim": 2048,
        "num_heads": 8,
        "ffn_dim": 16384,
        "num_layers": 18,
        "attention_type": "MQA",
        "group_size": 1
    }

def gemma_7b_config():
    """Gemma 7B configuration."""
    return {
        "vocab_size": 256000,
        "seq_length": 8192,
        "embedding_dim": 3072,
        "num_heads": 16,
        "ffn_dim": 24576,
        "num_layers": 28,
        "attention_type": "MHA",
        "group_size": 1
    }

def gemma2_9b_config():
    """Gemma 2 9B configuration."""
    return {
        "vocab_size": 256000,
        "seq_length": 8192,
        "embedding_dim": 3584,
        "num_heads": 16,
        "ffn_dim": 14336,
        "num_layers": 42,
        "attention_type": "GQA",
        "group_size": 2
    }

def gemma2_27b_config():
    """Gemma 2 27B configuration."""
    return {
        "vocab_size": 256000,
        "seq_length": 8192,
        "embedding_dim": 4608,
        "num_heads": 32,
        "ffn_dim": 18432,
        "num_layers": 46,
        "attention_type": "GQA",
        "group_size": 2
    }

def all_configs():
    """Return all configurations as a dictionary."""
    return {
        "GPT-2": gpt2_config(),
        "GPT-2-Medium": gpt2_medium_config(),
        "GPT-2-Large": gpt2_large_config(),
        "GPT-2-XL": gpt2_xl_config(),
        "GPT-3": gpt3_config(),
        "GPT-4 (est)": gpt4_estimate_config(),
        "LLaMA-2-7B": llama2_7b_config(),
        "LLaMA-2-13B": llama2_13b_config(),
        "LLaMA-2-70B": llama2_70b_config(),
        "LLaMA-3-8B": llama3_8b_config(),
        "LLaMA-3-70B": llama3_70b_config(),
        "LLaMA-4-8B (est)": llama4_8b_config(),
        "LLaMA-4-70B (est)": llama4_70b_config(),
        "TinyLLaMA-1.1B": tinyllama_1_1b_config(),
        "Mixtral-8x7B": mixtral_8x7b_config(),
        "Mixtral-8x22B": mixtral_8x22b_config(),
        "DeepSeek-7B": deepseek_7b_config(),
        "DeepSeek-67B": deepseek_67b_config(),
        "DeepSeek-Coder-V2": deepseek_coder_v2_config(),
        "Falcon-7B": falcon_7b_config(),
        "Falcon-40B": falcon_40b_config(),
        "Phi-2": phi_2_config(),
        "Phi-3-Mini": phi_3_mini_config(),
        "Phi-3-Medium": phi_3_medium_config(),
        "Gemma-2B": gemma_2b_config(),
        "Gemma-7B": gemma_7b_config(),
        "Gemma-2-9B": gemma2_9b_config(),
        "Gemma-2-27B": gemma2_27b_config()
    }