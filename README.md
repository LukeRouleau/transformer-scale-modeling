# transformer-scale-modeling
A (✨vibe-coded✨) model the for estimating the computational scale and memory requirements of transformer-based language models.

## Overview

This project provides:
1.  A Python class (`TransformerScaleModel` in `transformer_scale_model.py`) to calculate tensor shapes, activation memory (element counts), and approximate computational cost (FLOPs) for different components of a transformer model (Attention, FFN).
2.  Support for various attention mechanisms (MHA, GQA, MQA) and Mixture of Experts (MoE) configurations.
3.  A collection of pre-defined configurations for popular transformer models (`transformer_model_configs.py`).
4.  A script (`transformer_scale_analysis.py`) to analyze these configurations and generate detailed summary reports in both YAML and formatted text files.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/transformer-scale-model.git
cd transformer-scale-model

# Install dependencies (requires PyYAML)
pip install -r requirements.txt
```

## Files

*   `transformer_scale_model.py`: Contains the `TransformerScaleModel` class for performing the calculations.
*   `transformer_model_configs.py`: Defines Python dictionaries for various pre-existing LLM architectures (GPT, LLaMA, Mixtral, Gemma, etc.).
*   `transformer_scale_analysis.py`: Script to run the analysis using the model and configs, generating output reports.
*   `requirements.txt`: Lists the necessary Python packages (currently just `PyYAML`).
*   `output/`: Directory created to store the generated analysis reports (subdirectories `yaml/` and `txt/`).

## Usage

To analyze all pre-defined model configurations and generate reports:

```bash
python transformer_scale_analysis.py
```

This will:
1.  Create the `./output/yaml` and `./output/txt` directories if they don't exist.
2.  Iterate through each model configuration defined in `transformer_model_configs.py`.
3.  Instantiate `TransformerScaleModel` for each configuration.
4.  Calculate tensor sizes and compute estimates.
5.  Save a detailed summary in YAML format to `./output/yaml/<model_name>.yaml`.
6.  Save a formatted, human-readable text summary to `./output/txt/<model_name>.txt`.

## Features

### Supported Attention Mechanisms

*   **Multi-Head Attention (MHA)**: Standard attention mechanism.
*   **Grouped-Query Attention (GQA)**: Shares Key/Value heads across groups of Query heads. Requires specifying `group_size`.
*   **Multi-Query Attention (MQA)**: A special case of GQA where there is only one Key/Value head pair.

### Mixture of Experts Support

The model supports calculating scale for transformers with Mixture of Experts (MoE) layers:
*   **Standard MoE**: Define `moe_enabled=True`, `num_experts`, and `active_experts`.
*   **MoE with Shared Experts**: Define `moe_enabled=True`, `num_shared_experts`, `num_routed_experts`, and `active_experts`.

The calculations account for router computation and the fact that only active experts process tokens.

### Pre-defined Model Configurations

Includes configurations for models like:
*   GPT-2 (various sizes)
*   GPT-3, GPT-4 (Estimate)
*   LLaMA 2 & 3 (various sizes)
*   TinyLLaMA
*   Mixtral (8x7B, 8x22B)
*   DeepSeek (7B, 67B, Coder V2)
*   Falcon (7B, 40B)
*   Phi-2, Phi-3 (Mini, Medium)
*   Gemma & Gemma 2 (various sizes)
*   _(Note: Configurations for newer/closed models are estimates)._

## Example Output

Running the analysis script produces two types of files per model in the `output/` directory:

*   **YAML (`.yaml`)**: Structured data containing the model configuration, summary statistics (total compute/elements per layer/model), and detailed tensor shapes/elements for attention and FFN blocks. Suitable for programmatic use.
*   **Text (`.txt`)**: A formatted report including:
    *   Model Architecture details (dims, layers, heads, attention type, MoE config).
    *   Model Scale Metrics (Compute FLOPs, Activation Elements/Memory).
    *   Detailed tensor shapes and element counts for each step within the Attention and FFN blocks.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.