"""
This is a simple model the for analyzing the computational scale and memory requirements of transformer-based language models.

TODO: Add DeepSeek's Latent Attention (LQA)
"""

class TransformerScaleModel:
    """
    A simple model the for analyzing the computational scale and memory requirements of transformer-based language models.
    """
    def __init__(
        self,
        vocab_size=50000,
        seq_length=1024,
        embedding_dim=768,
        num_heads=12,
        ffn_dim=3072,
        num_layers=12,
        batch_size=1,
        attention_type="MHA",  # MHA, GQA, MQA
        group_size=1,  # For GQA, number of heads per group
        moe_enabled=False,
        num_experts=8,      # Total number of experts in standard MoE
        active_experts=2,   # Number of active experts per token
        num_shared_experts=0, # Number of shared experts (processed by all tokens)
        num_routed_experts=0  # Number of experts to route between (used if num_shared_experts > 0)
    ):
        """
        Initialize a transformer model scale calculator with given parameters.
        
        Args:
            vocab_size: Size of the vocabulary
            seq_length: Maximum sequence length
            embedding_dim: Dimension of embeddings
            num_heads: Number of attention heads
            ffn_dim: Dimension of feed-forward network (per expert if MoE)
            num_layers: Number of transformer layers
            batch_size: Batch size for training/inference
            attention_type: Type of attention mechanism (MHA, GQA, MQA)
            group_size: For GQA, number of heads per group
            moe_enabled: Whether to use Mixture of Experts
            num_experts: Total number of experts in standard MoE (used if num_shared_experts == 0)
            active_experts: Number of active experts per token (routed experts)
            num_shared_experts: Number of shared experts processed by all tokens.
            num_routed_experts: Number of experts to route between (used if num_shared_experts > 0).
                                If num_shared_experts > 0, num_experts is ignored for FFN calculation.
        """
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.ffn_dim = ffn_dim
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.attention_type = attention_type
        self.group_size = group_size
        self.moe_enabled = moe_enabled
        self.num_experts = num_experts
        self.active_experts = active_experts
        self.num_shared_experts = num_shared_experts
        self.num_routed_experts = num_routed_experts if num_shared_experts > 0 else 0 # Only relevant if shared experts exist

        # Use num_routed_experts for routing calculation if shared experts are present
        self.effective_num_routed_experts = self.num_routed_experts if self.num_shared_experts > 0 else self.num_experts

        # Validate parameters
        self._validate_parameters()
    
    def _validate_parameters(self):
        """Validate that the parameters make sense."""
        assert self.embedding_dim % self.num_heads == 0, "Embedding dimension must be divisible by number of heads"
        if self.attention_type == "GQA":
            assert self.num_heads % self.group_size == 0, "Number of heads must be divisible by group size for GQA"
            assert self.num_heads // self.group_size > 0, "Number of K/V heads must be positive for GQA"
        if self.moe_enabled:
            if self.num_shared_experts > 0:
                assert self.num_routed_experts > 0, "If num_shared_experts > 0, num_routed_experts must also be > 0"
                assert self.active_experts <= self.num_routed_experts, "Active experts cannot exceed the number of routed experts"
                assert self.num_shared_experts + self.num_routed_experts > 0, "Total number of experts must be positive" # Sanity check
            else:
                assert self.num_experts > 0, "If MoE is enabled without shared experts, num_experts must be > 0"
                assert self.active_experts <= self.num_experts, "Active experts cannot exceed the total number of experts"
    
    def _validate_matmul(self, input1_shape, input2_shape, output_shape, op_name="matmul"):
        """Validate shapes for a standard batch matrix multiplication."""
        assert len(input1_shape) >= 2 and len(input2_shape) >= 2, f"[{op_name}] Inputs must have at least 2 dimensions."

        # Standard Matmul checks
        contract_dim1 = input1_shape[-1]
        contract_dim2 = input2_shape[-2]
        assert contract_dim1 == contract_dim2, \
            f"[{op_name}] Inner dimensions must match for matmul: {input1_shape} vs {input2_shape}"

        expected_output_shape_end = (*input1_shape[:-1], input2_shape[-1])

        # Check batch dimensions (simple check for now, assumes broadcasting or direct match)
        # A full broadcasting check is complex, this checks basic compatibility.
        len_diff = abs(len(input1_shape) - len(input2_shape))
        shape1_batch = input1_shape[:-2]
        shape2_batch = input2_shape[:-2]

        if len(shape1_batch) > len(shape2_batch):
            assert shape1_batch[len_diff:] == shape2_batch or not shape2_batch, f"[{op_name}] Batch dimensions mismatch: {shape1_batch} vs {shape2_batch}"
        elif len(shape2_batch) > len(shape1_batch):
             assert shape2_batch[len_diff:] == shape1_batch or not shape1_batch, f"[{op_name}] Batch dimensions mismatch: {shape1_batch} vs {shape2_batch}"
        else:
             assert shape1_batch == shape2_batch, f"[{op_name}] Batch dimensions mismatch: {shape1_batch} vs {shape2_batch}"


        # Check if the calculated output shape matches the expected output shape structure
        assert len(output_shape) == len(expected_output_shape_end), \
             f"[{op_name}] Output dimension mismatch: {output_shape} vs expected {expected_output_shape_end}"
        assert output_shape[-2:] == expected_output_shape_end[-2:], \
             f"[{op_name}] Output core shape mismatch: {output_shape} vs expected {expected_output_shape_end}"
        # Simple check on batch dims, assuming the larger one dictates the output
        assert output_shape[:-2] == (shape1_batch if len(shape1_batch) >= len(shape2_batch) else shape2_batch), \
             f"[{op_name}] Output batch shape mismatch: {output_shape} vs expected based on inputs"


    def _validate_gqa_mqa_attn_v_matmul(self, attn_weights_shape, v_shape, output_shape):
        """Validate shapes specifically for GQA/MQA AttentionWeights @ V."""
        op_name = "attention_v_matmul (GQA/MQA)"
        assert len(attn_weights_shape) == 4, f"[{op_name}] Attention weights shape must be 4D."
        assert len(v_shape) == 4, f"[{op_name}] V shape must be 4D."
        assert len(output_shape) == 4, f"[{op_name}] Output shape must be 4D."

        b1, h, s1a, s1b = attn_weights_shape
        b2, kv_h, s2a, d_h = v_shape
        b_out, h_out, s_out, d_h_out = output_shape

        assert b1 == b2 == b_out, f"[{op_name}] Batch dimensions must match: {b1}, {b2}, {b_out}"
        assert h > 0 and kv_h > 0, f"[{op_name}] Head counts must be positive: h={h}, kv_h={kv_h}"
        assert h % kv_h == 0, f"[{op_name}] num_heads ({h}) must be divisible by num_kv_heads ({kv_h})"
        assert s1a == s1b, f"[{op_name}] Attention weights must be square in sequence length: {s1a} vs {s1b}"
        assert s1a == s2a, f"[{op_name}] Sequence lengths must match: {s1a} vs {s2a}"
        assert s1b == s2a, f"[{op_name}] Inner dimensions for matmul must match: {s1b} vs {s2a}" # Contraction dim check

        # Check output shape consistency
        assert h == h_out, f"[{op_name}] Output num_heads mismatch: {h_out} vs expected {h}"
        assert s1a == s_out, f"[{op_name}] Output sequence length mismatch: {s_out} vs expected {s1a}"
        assert d_h == d_h_out, f"[{op_name}] Output head dimension mismatch: {d_h_out} vs expected {d_h}"


    def get_embedding_tensor_size(self):
        """Calculate the size of the embedding tensor."""
        # Input embeddings: [batch_size, seq_length, embedding_dim]
        return {
            "shape": (self.batch_size, self.seq_length, self.embedding_dim),
            "elements": self.batch_size * self.seq_length * self.embedding_dim
        }
    
    def get_attention_tensor_sizes(self):
        """Calculate the sizes of tensors in the attention mechanism."""
        results = {}
        
        # Input to attention: [batch_size, seq_length, embedding_dim]
        results["input"] = {
            "shape": (self.batch_size, self.seq_length, self.embedding_dim),
            "elements": self.batch_size * self.seq_length * self.embedding_dim
        }
        
        # For MHA: each of Q, K, V has a separate projection for each head
        if self.attention_type == "MHA":
            # Q, K, V projections
            results["q_projection_matmul"] = {
                "input": {
                    "shape": [(self.batch_size, self.seq_length, self.embedding_dim), (self.embedding_dim, self.embedding_dim)],
                    "elements": self.batch_size * self.seq_length * self.embedding_dim * self.embedding_dim
                },
                "output": {
                    "shape": (self.batch_size, self.seq_length, self.embedding_dim),
                    "elements": self.batch_size * self.seq_length * self.embedding_dim
                }
            }
            self._validate_matmul(results["q_projection_matmul"]["input"]["shape"][0],
                                  results["q_projection_matmul"]["input"]["shape"][1],
                                  results["q_projection_matmul"]["output"]["shape"],
                                  "q_projection_matmul")
            results["k_projection_matmul"] = {
                "input": {
                    "shape": [(self.batch_size, self.seq_length, self.embedding_dim), (self.embedding_dim, self.embedding_dim)],
                    "elements": self.batch_size * self.seq_length * self.embedding_dim * self.embedding_dim
                },
                "output": {
                    "shape": (self.batch_size, self.seq_length, self.embedding_dim),
                    "elements": self.batch_size * self.seq_length * self.embedding_dim
                }
            }
            self._validate_matmul(results["k_projection_matmul"]["input"]["shape"][0],
                                  results["k_projection_matmul"]["input"]["shape"][1],
                                  results["k_projection_matmul"]["output"]["shape"],
                                  "k_projection_matmul")
            results["v_projection_matmul"] = {
                "input": {
                    "shape": [(self.batch_size, self.seq_length, self.embedding_dim), (self.embedding_dim, self.embedding_dim)],
                    "elements": self.batch_size * self.seq_length * self.embedding_dim * self.embedding_dim
                },
                "output": {
                    "shape": (self.batch_size, self.seq_length, self.embedding_dim),
                    "elements": self.batch_size * self.seq_length * self.embedding_dim
                }
            }
            self._validate_matmul(results["v_projection_matmul"]["input"]["shape"][0],
                                  results["v_projection_matmul"]["input"]["shape"][1],
                                  results["v_projection_matmul"]["output"]["shape"],
                                  "v_projection_matmul")
            
            # Reshaped Q, K, V: [batch_size, num_heads, seq_length, head_dim]
            results["q_reshaped"] = {
                "shape": (self.batch_size, self.num_heads, self.seq_length, self.head_dim),
                "elements": self.batch_size * self.num_heads * self.seq_length * self.head_dim
            }
            self._validate_matmul(results["q_projection_matmul"]["input"]["shape"][0],
                                  results["q_projection_matmul"]["input"]["shape"][1],
                                  results["q_projection_matmul"]["output"]["shape"],
                                  "q_projection_matmul (GQA)")
            results["k_reshaped"] = {
                "shape": (self.batch_size, self.num_heads, self.seq_length, self.head_dim),
                "elements": self.batch_size * self.num_heads * self.seq_length * self.head_dim
            }
            results["v_reshaped"] = {
                "shape": (self.batch_size, self.num_heads, self.seq_length, self.head_dim),
                "elements": self.batch_size * self.num_heads * self.seq_length * self.head_dim
            }
            
        # For GQA: K and V projections are shared among groups of heads
        elif self.attention_type == "GQA":
            num_kv_heads = self.num_heads // self.group_size
            
            # Q projection
            results["q_projection_matmul"] = {
                "input": {
                    "shape": [(self.batch_size, self.seq_length, self.embedding_dim), (self.embedding_dim, self.embedding_dim)],
                    "elements": self.batch_size * self.seq_length * self.embedding_dim * self.embedding_dim
                },
                "output": {
                    "shape": (self.batch_size, self.seq_length, self.embedding_dim),
                    "elements": self.batch_size * self.seq_length * self.embedding_dim
                }
            }
            self._validate_matmul(results["q_projection_matmul"]["input"]["shape"][0],
                                  results["q_projection_matmul"]["input"]["shape"][1],
                                  results["q_projection_matmul"]["output"]["shape"],
                                  "q_projection_matmul (GQA)")
            
            # K, V projections (shared)
            kv_dim = self.head_dim * num_kv_heads
            results["k_projection_matmul"] = {
                "input": {
                    "shape": [(self.batch_size, self.seq_length, self.embedding_dim), (self.embedding_dim, kv_dim)],
                    "elements": self.batch_size * self.seq_length * self.embedding_dim * kv_dim
                },
                "output": {
                    "shape": (self.batch_size, self.seq_length, kv_dim),
                    "elements": self.batch_size * self.seq_length * kv_dim
                }
            }
            self._validate_matmul(results["k_projection_matmul"]["input"]["shape"][0],
                                  results["k_projection_matmul"]["input"]["shape"][1],
                                  results["k_projection_matmul"]["output"]["shape"],
                                  "k_projection_matmul (GQA)")
            results["v_projection_matmul"] = {
                "input": {
                    "shape": [(self.batch_size, self.seq_length, self.embedding_dim), (self.embedding_dim, kv_dim)],
                    "elements": self.batch_size * self.seq_length * self.embedding_dim * kv_dim
                },
                "output": {
                    "shape": (self.batch_size, self.seq_length, kv_dim),
                    "elements": self.batch_size * self.seq_length * kv_dim
                }
            }
            self._validate_matmul(results["v_projection_matmul"]["input"]["shape"][0],
                                  results["v_projection_matmul"]["input"]["shape"][1],
                                  results["v_projection_matmul"]["output"]["shape"],
                                  "v_projection_matmul (GQA)")
            
            # Reshaped Q, K, V
            results["q_reshaped"] = {
                "shape": (self.batch_size, self.num_heads, self.seq_length, self.head_dim),
                "elements": self.batch_size * self.num_heads * self.seq_length * self.head_dim
            }
            self._validate_matmul(results["q_projection_matmul"]["input"]["shape"][0],
                                  results["q_projection_matmul"]["input"]["shape"][1],
                                  results["q_projection_matmul"]["output"]["shape"],
                                  "q_projection_matmul (MQA)")
            results["k_reshaped"] = {
                "shape": (self.batch_size, num_kv_heads, self.seq_length, self.head_dim),
                "elements": self.batch_size * num_kv_heads * self.seq_length * self.head_dim
            }
            results["v_reshaped"] = {
                "shape": (self.batch_size, num_kv_heads, self.seq_length, self.head_dim),
                "elements": self.batch_size * num_kv_heads * self.seq_length * self.head_dim
            }
            
        # For MQA (Multi-Query Attention): Only one K and V head
        elif self.attention_type == "MQA":
            # Q projection
            results["q_projection_matmul"] = {
                "input": {
                    "shape": [(self.batch_size, self.seq_length, self.embedding_dim), (self.embedding_dim, self.embedding_dim)],
                    "elements": self.batch_size * self.seq_length * self.embedding_dim * self.embedding_dim
                },
                "output": {
                    "shape": (self.batch_size, self.seq_length, self.embedding_dim),
                    "elements": self.batch_size * self.seq_length * self.embedding_dim
                }
            }
            
            # K, V projections (single head)
            results["k_projection_matmul"] = {
                "input": {
                    "shape": [(self.batch_size, self.seq_length, self.embedding_dim), (self.embedding_dim, self.head_dim)],
                    "elements": self.batch_size * self.seq_length * self.embedding_dim * self.head_dim
                },
                "output": {
                    "shape": (self.batch_size, self.seq_length, self.head_dim),
                    "elements": self.batch_size * self.seq_length * self.head_dim
                }
            }
            self._validate_matmul(results["k_projection_matmul"]["input"]["shape"][0],
                                  results["k_projection_matmul"]["input"]["shape"][1],
                                  results["k_projection_matmul"]["output"]["shape"],
                                  "k_projection_matmul (MQA)")
            results["v_projection_matmul"] = {
                "input": {
                    "shape": [(self.batch_size, self.seq_length, self.embedding_dim), (self.embedding_dim, self.head_dim)],
                    "elements": self.batch_size * self.seq_length * self.embedding_dim * self.head_dim
                },
                "output": {
                    "shape": (self.batch_size, self.seq_length, self.head_dim),
                    "elements": self.batch_size * self.seq_length * self.head_dim
                }
            }
            self._validate_matmul(results["v_projection_matmul"]["input"]["shape"][0],
                                  results["v_projection_matmul"]["input"]["shape"][1],
                                  results["v_projection_matmul"]["output"]["shape"],
                                  "v_projection_matmul (MQA)")
            
            # Reshaped Q, K, V
            results["q_reshaped"] = {
                "shape": (self.batch_size, self.num_heads, self.seq_length, self.head_dim),
                "elements": self.batch_size * self.num_heads * self.seq_length * self.head_dim
            }
            self._validate_matmul(results["q_projection_matmul"]["input"]["shape"][0],
                                  results["q_projection_matmul"]["input"]["shape"][1],
                                  results["q_projection_matmul"]["output"]["shape"],
                                  "q_projection_matmul (MQA)")
            results["k_reshaped"] = {
                "shape": (self.batch_size, 1, self.seq_length, self.head_dim),
                "elements": self.batch_size * self.seq_length * self.head_dim
            }
            results["v_reshaped"] = {
                "shape": (self.batch_size, 1, self.seq_length, self.head_dim),
                "elements": self.batch_size * self.seq_length * self.head_dim
            }
        
        # QK^T matrix multiplication (attention scores)
        if self.attention_type == "MHA":
            results["qk_matmul"] = {
                "input": {
                    "shape": [(self.batch_size, self.num_heads, self.seq_length, self.head_dim), 
                             (self.batch_size, self.num_heads, self.head_dim, self.seq_length)],
                    "elements": self.batch_size * self.num_heads * self.seq_length * self.head_dim * self.seq_length
                },
                "output": {
                    "shape": (self.batch_size, self.num_heads, self.seq_length, self.seq_length),
                    "elements": self.batch_size * self.num_heads * self.seq_length * self.seq_length
                }
            }
            self._validate_matmul(results["qk_matmul"]["input"]["shape"][0],
                                  results["qk_matmul"]["input"]["shape"][1],
                                  results["qk_matmul"]["output"]["shape"],
                                  "qk_matmul (MHA)")
        elif self.attention_type == "GQA":
            num_kv_heads = self.num_heads // self.group_size
            results["qk_matmul"] = {
                "input": {
                    "shape": [(self.batch_size, self.num_heads, self.seq_length, self.head_dim), 
                             (self.batch_size, num_kv_heads, self.head_dim, self.seq_length)],
                    "elements": self.batch_size * self.num_heads * self.seq_length * self.head_dim * self.seq_length
                },
                "output": {
                    "shape": (self.batch_size, self.num_heads, self.seq_length, self.seq_length),
                    "elements": self.batch_size * self.num_heads * self.seq_length * self.seq_length
                }
            }
        elif self.attention_type == "MQA":
            results["qk_matmul"] = {
                "input": {
                    "shape": [(self.batch_size, self.num_heads, self.seq_length, self.head_dim), 
                             (self.batch_size, 1, self.head_dim, self.seq_length)],
                    "elements": self.batch_size * self.num_heads * self.seq_length * self.head_dim * self.seq_length
                },
                "output": {
                    "shape": (self.batch_size, self.num_heads, self.seq_length, self.seq_length),
                    "elements": self.batch_size * self.num_heads * self.seq_length * self.seq_length
                }
            }
        
        # Attention weights (after softmax)
        results["attention_weights"] = {
            "shape": (self.batch_size, self.num_heads, self.seq_length, self.seq_length),
            "elements": self.batch_size * self.num_heads * self.seq_length * self.seq_length
        }
        
        # Attention output (attn * V)
        if self.attention_type == "MHA":
            results["attention_v_matmul"] = {
                "input": {
                    "shape": [(self.batch_size, self.num_heads, self.seq_length, self.seq_length), 
                             (self.batch_size, self.num_heads, self.seq_length, self.head_dim)],
                    "elements": self.batch_size * self.num_heads * self.seq_length * self.seq_length * self.head_dim
                },
                "output": {
                    "shape": (self.batch_size, self.num_heads, self.seq_length, self.head_dim),
                    "elements": self.batch_size * self.num_heads * self.seq_length * self.head_dim
                }
            }
            self._validate_matmul(results["attention_v_matmul"]["input"]["shape"][0],
                                  results["attention_v_matmul"]["input"]["shape"][1],
                                  results["attention_v_matmul"]["output"]["shape"],
                                  "attention_v_matmul (MHA)")
        elif self.attention_type == "GQA":
            num_kv_heads = self.num_heads // self.group_size
            results["attention_v_matmul"] = {
                "input": {
                    "shape": [(self.batch_size, self.num_heads, self.seq_length, self.seq_length), 
                             (self.batch_size, num_kv_heads, self.seq_length, self.head_dim)],
                    "elements": self.batch_size * self.num_heads * self.seq_length * self.seq_length * self.head_dim
                },
                "output": {
                    "shape": (self.batch_size, self.num_heads, self.seq_length, self.head_dim),
                    "elements": self.batch_size * self.num_heads * self.seq_length * self.head_dim
                }
            }
            # Use specific GQA/MQA validation for this step
            self._validate_gqa_mqa_attn_v_matmul(results["attention_v_matmul"]["input"]["shape"][0],
                                                 results["attention_v_matmul"]["input"]["shape"][1],
                                                 results["attention_v_matmul"]["output"]["shape"])
        elif self.attention_type == "MQA":
            results["attention_v_matmul"] = {
                "input": {
                    "shape": [(self.batch_size, self.num_heads, self.seq_length, self.seq_length), 
                             (self.batch_size, 1, self.seq_length, self.head_dim)],
                    "elements": self.batch_size * self.num_heads * self.seq_length * self.seq_length * self.head_dim
                },
                "output": {
                    "shape": (self.batch_size, self.num_heads, self.seq_length, self.head_dim),
                    "elements": self.batch_size * self.num_heads * self.seq_length * self.head_dim
                }
            }
            # Use specific GQA/MQA validation for this step
            self._validate_gqa_mqa_attn_v_matmul(results["attention_v_matmul"]["input"]["shape"][0],
                                                 results["attention_v_matmul"]["input"]["shape"][1],
                                                 results["attention_v_matmul"]["output"]["shape"])
        
        # Attention output (reshaped for projection)
        results["attention_output_reshaped"] = {
            "shape": (self.batch_size, self.seq_length, self.embedding_dim),
            "elements": self.batch_size * self.seq_length * self.embedding_dim
        }
        
        # Output projection (res*O)
        results["output_projection_matmul"] = {
            "input": {
                "shape": [(self.batch_size, self.seq_length, self.embedding_dim), (self.embedding_dim, self.embedding_dim)],
                "elements": self.batch_size * self.seq_length * self.embedding_dim * self.embedding_dim
            },
            "output": {
                "shape": (self.batch_size, self.seq_length, self.embedding_dim),
                "elements": self.batch_size * self.seq_length * self.embedding_dim
            }
        }
        self._validate_matmul(results["output_projection_matmul"]["input"]["shape"][0],
                              results["output_projection_matmul"]["input"]["shape"][1],
                              results["output_projection_matmul"]["output"]["shape"],
                              "output_projection_matmul")
        
        return results
    
    def get_ffn_tensor_sizes(self):
        """Calculate the sizes of tensors in the feed-forward network."""
        results = {}
        
        # Input to FFN/MoE block
        results["input"] = {
            "shape": (self.batch_size, self.seq_length, self.embedding_dim),
            "elements": self.batch_size * self.seq_length * self.embedding_dim
        }

        if not self.moe_enabled:
            # Standard FFN
            # First linear layer
            results["linear1_matmul"] = {
                "input": {
                    "shape": [(self.batch_size, self.seq_length, self.embedding_dim), (self.embedding_dim, self.ffn_dim)],
                    "elements": self.batch_size * self.seq_length * self.embedding_dim * self.ffn_dim
                },
                "output": {
                    "shape": (self.batch_size, self.seq_length, self.ffn_dim),
                    "elements": self.batch_size * self.seq_length * self.ffn_dim
                }
            }
            self._validate_matmul(results["linear1_matmul"]["input"]["shape"][0],
                                  results["linear1_matmul"]["input"]["shape"][1],
                                  results["linear1_matmul"]["output"]["shape"],
                                  "ffn_linear1_matmul")

            # Second linear layer
            results["linear2_matmul"] = {
                "input": {
                    "shape": [(self.batch_size, self.seq_length, self.ffn_dim), (self.ffn_dim, self.embedding_dim)],
                    "elements": self.batch_size * self.seq_length * self.ffn_dim * self.embedding_dim
                },
                "output": {
                    "shape": (self.batch_size, self.seq_length, self.embedding_dim),
                    "elements": self.batch_size * self.seq_length * self.embedding_dim
                }
            }
            self._validate_matmul(results["linear2_matmul"]["input"]["shape"][0],
                                  results["linear2_matmul"]["input"]["shape"][1],
                                  results["linear2_matmul"]["output"]["shape"],
                                  "ffn_linear2_matmul")
        else:
            # Mixture of Experts FFN

            # --- Shared Experts (if any) ---
            if self.num_shared_experts > 0:
                results["shared_experts"] = {}
                # Shared expert computations run on all tokens
                shared_tokens = self.batch_size * self.seq_length

                results["shared_experts"]["linear1_matmul"] = {
                    "input": {
                        "shape": [(shared_tokens, self.embedding_dim), (self.embedding_dim, self.ffn_dim)],
                        "elements": shared_tokens * self.embedding_dim * self.ffn_dim * self.num_shared_experts # Summed over shared experts
                    },
                    "output": {
                        "shape": (shared_tokens, self.ffn_dim),
                        "elements": shared_tokens * self.ffn_dim * self.num_shared_experts # Summed over shared experts
                    }
                }
                self._validate_matmul(results["shared_experts"]["linear1_matmul"]["input"]["shape"][0],
                                      results["shared_experts"]["linear1_matmul"]["input"]["shape"][1],
                                      results["shared_experts"]["linear1_matmul"]["output"]["shape"],
                                      "moe_shared_linear1_matmul")
                results["shared_experts"]["linear2_matmul"] = {
                    "input": {
                        "shape": [(shared_tokens, self.ffn_dim), (self.ffn_dim, self.embedding_dim)],
                        "elements": shared_tokens * self.ffn_dim * self.embedding_dim * self.num_shared_experts # Summed over shared experts
                    },
                    "output": {
                        "shape": (shared_tokens, self.embedding_dim),
                        "elements": shared_tokens * self.embedding_dim * self.num_shared_experts # Summed over shared experts
                    }
                }
                self._validate_matmul(results["shared_experts"]["linear2_matmul"]["input"]["shape"][0],
                                      results["shared_experts"]["linear2_matmul"]["input"]["shape"][1],
                                      results["shared_experts"]["linear2_matmul"]["output"]["shape"],
                                      "moe_shared_linear2_matmul")

            # --- Routed Experts ---
            # Router computation targets the number of experts available for routing
            results["router_matmul"] = {
                "input": {
                    "shape": [(self.batch_size, self.seq_length, self.embedding_dim), (self.embedding_dim, self.effective_num_routed_experts)],
                    "elements": self.batch_size * self.seq_length * self.embedding_dim * self.effective_num_routed_experts
                },
                "output": {
                    "shape": (self.batch_size, self.seq_length, self.effective_num_routed_experts),
                    "elements": self.batch_size * self.seq_length * self.effective_num_routed_experts
                }
            }
            self._validate_matmul(results["router_matmul"]["input"]["shape"][0],
                                  results["router_matmul"]["input"]["shape"][1],
                                  results["router_matmul"]["output"]["shape"],
                                  "moe_router_matmul")

            # Routed expert computation (for active experts only)
            # Capacity: average number of tokens processed by each routed expert
            # Note: This is a simplification; real systems often use capacity factors.
            total_tokens = self.batch_size * self.seq_length
            if self.effective_num_routed_experts > 0:
                 tokens_per_expert = total_tokens * self.active_experts / self.effective_num_routed_experts
            else:
                 tokens_per_expert = 0 # Avoid division by zero if only shared experts exist (though validation prevents this)

            results["routed_experts"] = {}
            results["routed_experts"]["linear1_matmul"] = {
                "input": {
                    "shape": [(tokens_per_expert, self.embedding_dim), (self.embedding_dim, self.ffn_dim)],
                    # Total compute/elements across all routed experts for the active tokens
                    "elements": tokens_per_expert * self.embedding_dim * self.ffn_dim * self.effective_num_routed_experts
                },
                "output": {
                    "shape": (tokens_per_expert, self.ffn_dim),
                     # Total elements across all routed experts for the active tokens
                    "elements": tokens_per_expert * self.ffn_dim * self.effective_num_routed_experts
                }
            }
            self._validate_matmul(results["routed_experts"]["linear1_matmul"]["input"]["shape"][0],
                                  results["routed_experts"]["linear1_matmul"]["input"]["shape"][1],
                                  results["routed_experts"]["linear1_matmul"]["output"]["shape"],
                                  "moe_routed_linear1_matmul")

            results["routed_experts"]["linear2_matmul"] = {
                "input": {
                    "shape": [(tokens_per_expert, self.ffn_dim), (self.ffn_dim, self.embedding_dim)],
                     # Total compute/elements across all routed experts for the active tokens
                    "elements": tokens_per_expert * self.ffn_dim * self.embedding_dim * self.effective_num_routed_experts
                },
                "output": {
                    "shape": (tokens_per_expert, self.embedding_dim),
                    # Total elements across all routed experts for the active tokens
                    "elements": tokens_per_expert * self.embedding_dim * self.effective_num_routed_experts
                }
            }
            self._validate_matmul(results["routed_experts"]["linear2_matmul"]["input"]["shape"][0],
                                  results["routed_experts"]["linear2_matmul"]["input"]["shape"][1],
                                  results["routed_experts"]["linear2_matmul"]["output"]["shape"],
                                  "moe_routed_linear2_matmul")

            # Final output after combining expert outputs (shared + routed)
            # The shape/elements remain the same as the input to the FFN block
            results["output"] = {
                "shape": (self.batch_size, self.seq_length, self.embedding_dim),
                "elements": self.batch_size * self.seq_length * self.embedding_dim
            }

        return results
    
    def get_layer_tensor_sizes(self):
        """Calculate the sizes of tensors in a single transformer layer."""
        results = {}
        
        # Attention block
        results["attention"] = self.get_attention_tensor_sizes()
        
        # Feed-forward network
        results["ffn"] = self.get_ffn_tensor_sizes()
        
        return results
    
    def get_model_tensor_sizes(self):
        """Calculate the sizes of tensors in the entire model."""
        results = {}
        
        # Embedding layer
        results["embedding"] = self.get_embedding_tensor_size()
        
        # Transformer layers
        layer_sizes = self.get_layer_tensor_sizes()
        results["layer"] = layer_sizes
        
        # Summary statistics
        results["summary"] = self.get_summary_statistics(layer_sizes)
        
        return results
    
    def get_summary_statistics(self, layer_sizes):
        """Calculate summary statistics across layers."""
        summary = {}
        
        # Total elements in attention
        total_attention_elements = 0
        total_attention_compute = 0
        
        # Count elements in tensors and compute operations in matrix multiplications
        for key, value in layer_sizes["attention"].items():
            if key != "input" and isinstance(value, dict) and "elements" in value:
                total_attention_elements += value["elements"]
            
            if isinstance(value, dict) and "input" in value and "output" in value:
                if "matmul" in key:
                    input_shapes = value["input"]["shape"]
                    output_elements = value["output"]["elements"]
                    
                    # Determine the contraction dimension (common dimension in the matmul)
                    if key == "qk_matmul":
                        # Q*K^T: [b, h, s, d] @ [b, h, d, s] -> [b, h, s, s]
                        contraction_dim = self.head_dim
                    elif key == "attention_v_matmul":
                        # Attn*V: [b, h, s, s] @ [b, h, s, d] -> [b, h, s, d]
                        contraction_dim = self.seq_length
                    else:
                        # For projection matmuls, use the embedding dimension
                        contraction_dim = self.embedding_dim
                    
                    # Compute operations = output_elements * contraction_dim
                    compute_ops = output_elements * contraction_dim
                    total_attention_compute += compute_ops
        
        # Total elements and compute in FFN
        total_ffn_elements = 0
        total_ffn_compute = 0

        for key, value in layer_sizes["ffn"].items():
            # Correctly sum elements, checking nested 'output' dict for matmuls
            if key != "input" and isinstance(value, dict):
                # Handle standard FFN or final MoE output
                if "elements" in value and not ("input" in value and "output" in value):
                    total_ffn_elements += value["elements"]
                # Handle matmul outputs (standard or router)
                elif "output" in value and "elements" in value["output"]:
                    total_ffn_elements += value["output"]["elements"]
                # Handle nested MoE structures (shared_experts, routed_experts)
                elif key in ["shared_experts", "routed_experts"]:
                     for expert_key, expert_value in value.items():
                         if "output" in expert_value and "elements" in expert_value["output"]:
                             total_ffn_elements += expert_value["output"]["elements"]


            # Count compute operations in matrix multiplications
            if isinstance(value, dict):
                 # Standard FFN or MoE router
                if "matmul" in key and "input" in value and "output" in value:
                    output_elements = value["output"]["elements"]
                    # Determine contraction dimension
                    if "linear1" in key or "router" in key:
                        contraction_dim = self.embedding_dim
                    elif "linear2" in key:
                        contraction_dim = self.ffn_dim
                    else: # Should not happen with current keys
                         contraction_dim = 1 # Default fallback

                    compute_ops = output_elements * contraction_dim
                    total_ffn_compute += compute_ops

                # MoE expert layers (shared or routed)
                elif key in ["shared_experts", "routed_experts"]:
                     for expert_key, expert_value in value.items():
                         if "matmul" in expert_key and "input" in expert_value and "output" in expert_value["output"]:
                            output_elements = expert_value["output"]["elements"]
                            if "linear1" in expert_key:
                                contraction_dim = self.embedding_dim
                            elif "linear2" in expert_key:
                                contraction_dim = self.ffn_dim
                            else: # Should not happen
                                contraction_dim = 1

                            # Note: expert_value already contains summed elements/compute for its type
                            compute_ops = output_elements * contraction_dim
                            total_ffn_compute += compute_ops


        # Total elements and compute in a single layer
        total_layer_elements = total_attention_elements + total_ffn_elements
        total_layer_compute = total_attention_compute + total_ffn_compute
        
        # Total elements and compute in all layers
        total_model_elements = total_layer_elements * self.num_layers
        total_model_compute = total_layer_compute * self.num_layers
        
        # Summary
        summary["attention_elements_per_layer"] = total_attention_elements
        summary["attention_compute_per_layer"] = total_attention_compute
        summary["ffn_elements_per_layer"] = total_ffn_elements
        summary["ffn_compute_per_layer"] = total_ffn_compute
        summary["total_elements_per_layer"] = total_layer_elements
        summary["total_compute_per_layer"] = total_layer_compute
        summary["total_model_elements"] = total_model_elements
        summary["total_model_compute"] = total_model_compute
        
        return summary
    
    def print_tensor_sizes(self, tensor_sizes, indent=0):
        """Print tensor sizes in a formatted way."""
        indent_str = "  " * indent
        for key, value in tensor_sizes.items():
            if isinstance(value, dict):
                if "shape" in value and "elements" in value:
                    shape_str = " × ".join(str(dim) for dim in value["shape"])
                    print(f"{indent_str}{key}: Shape = [{shape_str}], Elements = {value['elements']:,}")
                elif "input" in value and "output" in value:
                    print(f"{indent_str}{key}:")
                    if isinstance(value["input"]["shape"], list):
                        # Multiple input tensors
                        for i, shape in enumerate(value["input"]["shape"]):
                            shape_str = " × ".join(str(dim) for dim in shape)
                            print(f"{indent_str}  Input {i+1}: Shape = [{shape_str}]")
                    else:
                        # Single input tensor
                        shape_str = " × ".join(str(dim) for dim in value["input"]["shape"])
                        print(f"{indent_str}  Input: Shape = [{shape_str}]")
                    
                    # Output tensor
                    shape_str = " × ".join(str(dim) for dim in value["output"]["shape"])
                    print(f"{indent_str}  Output: Shape = [{shape_str}], Elements = {value['output']['elements']:,}")
                else:
                    print(f"{indent_str}{key}:")
                    self.print_tensor_sizes(value, indent + 1)
            else:
                print(f"{indent_str}{key}: {value:,}")
                
    def display_model_summary(self):
        """Display a summary of the model's tensor sizes and compute requirements."""
        model_sizes = self.get_model_tensor_sizes()
        
        print(f"Transformer Model Scale Summary")
        print(f"==============================")
        print(f"Configuration:")
        print(f"  Vocab Size: {self.vocab_size:,}")
        print(f"  Sequence Length: {self.seq_length:,}")
        print(f"  Embedding Dimension: {self.embedding_dim:,}")
        print(f"  Number of Heads: {self.num_heads:,}")
        print(f"  Head Dimension: {self.head_dim:,}")
        print(f"  FFN Dimension: {self.ffn_dim:,}")
        print(f"  Number of Layers: {self.num_layers:,}")
        print(f"  Batch Size: {self.batch_size:,}")
        print(f"  Attention Type: {self.attention_type}")
        
        if self.attention_type == "GQA":
            print(f"  Group Size: {self.group_size:,}")
        
        if self.moe_enabled:
            print(f"  MoE Enabled: Yes")
            if self.num_shared_experts > 0:
                print(f"  Shared Experts: {self.num_shared_experts:,}")
                print(f"  Routed Experts: {self.num_routed_experts:,}")
                print(f"  Active Routed Experts: {self.active_experts:,}")
            else:
                print(f"  Total Experts: {self.num_experts:,}")
                print(f"  Active Experts: {self.active_experts:,}")
        else:
            print(f"  MoE Enabled: No")
        
        print("\nModel Summary Statistics:")
        for key, value in model_sizes["summary"].items():
            print(f"  {key.replace('_', ' ').title()}: {value:,}")
        
        # Ask if the user wants to see detailed tensor sizes
        show_details = input("\nShow detailed tensor sizes per layer? (y/n): ")
        if show_details.lower() == 'y':
            print("\nDetailed Tensor Sizes per Layer:")
            print("\nAttention Block:")
            self.print_tensor_sizes(model_sizes["layer"]["attention"])
            
            print("\nFeed-Forward Network:")
            self.print_tensor_sizes(model_sizes["layer"]["ffn"])
    
    def generate_tensor_data(self):
        """Generate tensor data for visualization."""
        model_sizes = self.get_model_tensor_sizes()
        return {
            "config": {
                "vocab_size": self.vocab_size,
                "seq_length": self.seq_length,
                "embedding_dim": self.embedding_dim,
                "num_heads": self.num_heads,
                "head_dim": self.head_dim,
                "ffn_dim": self.ffn_dim,
                "num_layers": self.num_layers,
                "batch_size": self.batch_size,
                "attention_type": self.attention_type,
                "group_size": self.group_size if self.attention_type == "GQA" else None,
                "moe_enabled": self.moe_enabled,
                "num_experts": self.num_experts if self.moe_enabled and self.num_shared_experts == 0 else None,
                "active_experts": self.active_experts if self.moe_enabled else None,
                "num_shared_experts": self.num_shared_experts if self.moe_enabled and self.num_shared_experts > 0 else None,
                "num_routed_experts": self.num_routed_experts if self.moe_enabled and self.num_shared_experts > 0 else None
            },
            "summary": model_sizes["summary"],
            "detail": {
                "attention": model_sizes["layer"]["attention"],
                "ffn": model_sizes["layer"]["ffn"]
            }
        }