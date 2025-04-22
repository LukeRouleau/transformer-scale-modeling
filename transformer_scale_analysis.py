"""
Script to analyze transformer model configurations and output summary statistics as YAML files.
"""

import os
import yaml
from transformer_scale_model import TransformerScaleModel
from transformer_model_configs import all_configs

def format_number(num):
    """Format large numbers with suffixes for better readability."""
    if num >= 1_000_000_000:
        return f"{num/1_000_000_000:.2f}B"
    elif num >= 1_000_000:
        return f"{num/1_000_000:.2f}M"
    elif num >= 1_000:
        return f"{num/1_000:.2f}K"
    else:
        return f"{num:,}"

def format_tensor_sizes(tensor_data, indent=0):
    """Formats tensor size dictionary into a readable string."""
    output_lines = []
    indent_str = "  " * indent
    for key, value in tensor_data.items():
        if isinstance(value, dict):
            if "shape" in value and "elements" in value:
                # Handle tensors with shape and elements
                shape_str = ", ".join(str(dim) for dim in value["shape"])
                output_lines.append(f"{indent_str}{key}: Shape = [{shape_str}], Elements = {value['elements']:,}")
            elif "input" in value and "output" in value:
                # Handle operations with input and output (like matmuls)
                output_lines.append(f"{indent_str}{key}:")
                # Format input shape(s)
                if isinstance(value["input"]["shape"], list):
                    for i, shape in enumerate(value["input"]["shape"]):
                        shape_str = ", ".join(str(dim) for dim in shape)
                        output_lines.append(f"{indent_str}  Input {i+1}: Shape = [{shape_str}]")
                else:
                    shape_str = ", ".join(str(dim) for dim in value["input"]["shape"])
                    output_lines.append(f"{indent_str}  Input: Shape = [{shape_str}]")
                # Format output shape
                shape_str = ", ".join(str(dim) for dim in value["output"]["shape"])
                output_lines.append(f"{indent_str}  Output: Shape = [{shape_str}], Elements = {value['output']['elements']:,}")
                
                # Add GQA/MQA annotation if necessary
                if key == "attention_v_matmul":
                    input_shapes = value["input"]["shape"]
                    if isinstance(input_shapes, list) and len(input_shapes) == 2: # Ensure two inputs exist
                        input1_shape = input_shapes[0] # Attention weights
                        input2_shape = input_shapes[1] # Value tensor V
                        # Check for 4D shapes and head dimension mismatch indicating GQA/MQA
                        if len(input1_shape) == 4 and len(input2_shape) == 4 and input1_shape[1] != input2_shape[1]:
                             num_q_heads = input1_shape[1]
                             num_kv_heads = input2_shape[1]
                             if num_q_heads > 0 and num_kv_heads > 0 and num_q_heads % num_kv_heads == 0:
                                 output_lines.append(f"{indent_str}  Note: GQA/MQA broadcasts K/V heads ({num_kv_heads}) to match Q heads ({num_q_heads}).")
            else:
                # Recursively format nested dictionaries
                output_lines.append(f"{indent_str}{key}:")
                output_lines.extend(format_tensor_sizes(value, indent + 1))
        else:
            # Handle simple key-value pairs (like in summary)
            output_lines.append(f"{indent_str}{key}: {value:,}")
    return output_lines

def format_model_summary_text(model_name, data):
    """Formats the complete model data into a readable text summary string."""
    lines = []
    
    # Title with border
    title = f" Model Analysis: {model_name} "
    border = "=" * len(title)
    lines.append(border)
    lines.append(title)
    lines.append(border)
    lines.append("")

    # --- MODEL ARCHITECTURE SECTION ---
    lines.append("╔═══════════════════════════╗")
    lines.append("║   MODEL ARCHITECTURE      ║")
    lines.append("╚═══════════════════════════╝")
    
    config = data["config"]
    
    # Basic architecture details
    lines.append("")
    lines.append("  Basic Configuration:")
    lines.append(f"  • Embedding Dimension: {config['embedding_dim']:,}")
    lines.append(f"  • Number of Layers:    {config['num_layers']:,}")
    lines.append(f"  • Number of Heads:     {config['num_heads']:,}")
    lines.append(f"  • FFN Dimension:       {config['ffn_dim']:,} (per expert if MoE)")
    lines.append(f"  • Vocabulary Size:     {config['vocab_size']:,}")
    lines.append(f"  • Sequence Length:     {config['seq_length']:,}")
    
    # Attention mechanism
    lines.append("")
    lines.append("  Attention Mechanism:")
    lines.append(f"  • Type: {config['attention_type']}")
    lines.append(f"  • Heads (Query):     {config['num_heads']:,}")
    if config['attention_type'] == "GQA":
        num_kv_heads = config['num_heads'] // config['group_size']
        lines.append(f"  • Group Size:        {config['group_size']:,}")
        lines.append(f"  • Heads (KV):        {num_kv_heads:,}")
    elif config['attention_type'] == "MQA":
         lines.append(f"  • Heads (KV):        1")
    else: # MHA
         lines.append(f"  • Heads (KV):        {config['num_heads']:,}")

    # MoE details if enabled
    if config.get('moe_enabled', False): # Use .get for safety
        lines.append("")
        lines.append("  Mixture of Experts (MoE):")
        
        num_shared = config.get('num_shared_experts')
        num_routed = config.get('num_routed_experts')
        active_experts = config.get('active_experts')

        if num_shared is not None and num_shared > 0:
            # Shared + Routed configuration
            lines.append(f"  • Shared Experts:     {num_shared:,}")
            lines.append(f"  • Routed Experts:     {num_routed:,}")
            lines.append(f"  • Active Routed Exp:  {active_experts:,}")
            
            # Calculate capacity based on routed experts
            total_tokens = config['batch_size'] * config['seq_length']
            if num_routed > 0:
                expert_capacity = total_tokens * active_experts / num_routed
            else:
                expert_capacity = 0 # Should not happen based on validation
            lines.append(f"  • Avg Tokens/Routed Exp: {expert_capacity:,.1f}")
        else:
            # Standard MoE configuration
            total_experts = config.get('num_experts') # Might be None if shared > 0
            if total_experts is not None:
                lines.append(f"  • Total Experts:      {total_experts:,}")
                lines.append(f"  • Active Experts:     {active_experts:,}")
                
                # Calculate capacity based on total experts
                total_tokens = config['batch_size'] * config['seq_length']
                if total_experts > 0:
                     expert_capacity = total_tokens * active_experts / total_experts
                else:
                     expert_capacity = 0
                lines.append(f"  • Avg Tokens/Expert: {expert_capacity:,.1f}")
            else:
                 lines.append("  • Configuration: Standard (details pending config update)") # Fallback

    # --- SCALE METRICS SECTION ---
    lines.append("")
    lines.append("╔═══════════════════════════╗")
    lines.append("║   MODEL SCALE METRICS     ║")
    lines.append("╚═══════════════════════════╝")
    
    summary = data['summary']
    
    # Compute metrics
    lines.append("")
    lines.append("  Compute Requirements (per forward pass):")
    lines.append(f"  • Total Model Compute:    {format_number(summary['total_model_compute'])} FLOPs")
    lines.append(f"  • Per Layer Compute:      {format_number(summary['total_compute_per_layer'])} FLOPs")
    lines.append(f"    ├─ Attention Compute:   {format_number(summary['attention_compute_per_layer'])} FLOPs")
    lines.append(f"    └─ FFN Compute:         {format_number(summary['ffn_compute_per_layer'])} FLOPs")
    
    # Memory metrics (activation tensors)
    lines.append("")
    lines.append("  Memory Requirements (activation tensors):")
    lines.append(f"  • Total Model Elements:   {format_number(summary['total_model_elements'])} elements")
    lines.append(f"  • Per Layer Elements:     {format_number(summary['total_elements_per_layer'])} elements")
    lines.append(f"    ├─ Attention Elements:  {format_number(summary['attention_elements_per_layer'])} elements")
    lines.append(f"    └─ FFN Elements:        {format_number(summary['ffn_elements_per_layer'])} elements")
    
    # Memory with assumptions about precision
    lines.append("")
    lines.append("  Estimated Memory (assuming FP16 precision):")
    elements_in_bytes = summary['total_model_elements'] * 2  # 2 bytes per element for FP16
    elements_in_gb = elements_in_bytes / (1024**3)
    lines.append(f"  • Activation Memory:  {elements_in_gb:.2f} GB")
    
    # --- DETAILED LAYER INFORMATION ---
    lines.append("")
    lines.append("╔═══════════════════════════╗")
    lines.append("║   DETAILED LAYER INFO     ║")
    lines.append("╚═══════════════════════════╝")
    
    # Attention Block
    lines.append("")
    lines.append("  Attention Block:")
    attention_details = format_tensor_sizes(data["detail"]["attention"], indent=1)
    lines.extend(["  " + line for line in attention_details])
    
    # FFN Block
    lines.append("")
    lines.append("  Feed-Forward Network:")
    ffn_details = format_tensor_sizes(data["detail"]["ffn"], indent=1)
    lines.extend(["  " + line for line in ffn_details])
    
    return "\n".join(lines)

def main():
    """
    Analyzes all predefined model configurations and saves their summaries to YAML files.
    """
    yaml_output_dir = "./output/yaml"
    txt_output_dir = "./output/txt"
    os.makedirs(yaml_output_dir, exist_ok=True)
    os.makedirs(txt_output_dir, exist_ok=True)

    print(f"Analyzing all model configurations from transformer_model_configs.py...")
    print(f"Outputting YAML summaries to {yaml_output_dir}")
    print(f"Outputting Text summaries to {txt_output_dir}")

    model_configurations = all_configs()

    for model_name, config in model_configurations.items():
        print(f"  Analyzing: {model_name}")

        # Set default values for optional parameters if not present
        config.setdefault("batch_size", 1)
        config.setdefault("moe_enabled", False)
        config.setdefault("num_experts", 8)
        config.setdefault("active_experts", 2)
        config.setdefault("num_shared_experts", 0)
        config.setdefault("num_routed_experts", 0)
        config.setdefault("group_size", 1)

        # Ensure group_size is correctly handled for non-GQA types if modified from defaults
        if config["attention_type"] != "GQA":
            config["group_size"] = 1 # For MHA, MQA, group_size is effectively 1

        try:
            # Create model instance
            model = TransformerScaleModel(
                vocab_size=config["vocab_size"],
                seq_length=config["seq_length"],
                embedding_dim=config["embedding_dim"],
                num_heads=config["num_heads"],
                ffn_dim=config["ffn_dim"],
                num_layers=config["num_layers"],
                batch_size=config["batch_size"],
                attention_type=config["attention_type"],
                group_size=config["group_size"],
                moe_enabled=config["moe_enabled"],
                num_experts=config["num_experts"],
                active_experts=config["active_experts"],
                num_shared_experts=config["num_shared_experts"],
                num_routed_experts=config["num_routed_experts"]
            )

            # Generate tensor data (includes summary)
            data = model.generate_tensor_data()

            # Prepare data for YAML output (config + summary)
            output_data = {
                "model_name": model_name,
                "configuration": data["config"],
                "summary_statistics": data["summary"],
                "layer_details": data["detail"]
            }

            # Define output filename (sanitize model name for filename)
            safe_model_name = "".join(c if c.isalnum() else "_" for c in model_name)
            yaml_filename = os.path.join(yaml_output_dir, f"{safe_model_name}.yaml")

            # Write to YAML file
            with open(yaml_filename, 'w') as yaml_file:
                yaml.dump(output_data, yaml_file, default_flow_style=False, sort_keys=False)

            # --- Text Output ---
            text_summary = format_model_summary_text(model_name, data)
            txt_filename = os.path.join(txt_output_dir, f"{safe_model_name}.txt")

            # Write to Text file
            with open(txt_filename, 'w') as txt_file:
                txt_file.write(text_summary)

        except Exception as e:
            print(f"    ERROR analyzing {model_name}: {e}")
            # Optionally continue to the next model or re-raise the exception

    print(f"\nAnalysis complete. Summary files saved in {yaml_output_dir} and {txt_output_dir}")

if __name__ == "__main__":
    # Ensure PyYAML is installed
    try:
        import yaml
    except ImportError:
        print("Error: PyYAML is not installed. Please install it using:")
        print("pip install PyYAML")
        exit(1)

    main()