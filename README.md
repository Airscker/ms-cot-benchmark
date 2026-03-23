# MS-CoT Benchmark: Zero-Shot Structure Elucidation from Mass Spectra

This repository contains the code and data to reproduce the zero-shot Chain-of-Thought (CoT) LLM benchmark presented in our paper.

## Repository Contents

* `prepare_dataset.py`: Script to generate the text-based CoT prompts from the MassSpecGym dataset.
* `benchmark_dataset_test.jsonl` (to be generated or downloaded): The generated benchmark dataset containing input prompts, metadata, and target SMILES.
* **Generation Scripts** (`claude.py`, `openai_4o.py`, `llama3.py`, `llama3-8b.py`): Scripts used to query the respective model APIs for structure prediction.
* **Evaluation Scripts** (`evaluate.py`, `analyze_cot.py`, `smiles_accuracy_analysis.py`): Scripts used to compute exact match, Tanimoto similarity, DBE accuracy, and CoT contradiction rates.
* **Plotting**: `plot_results.py` for regenerating paper figures.
* `examples/` (to be created): Sample LLM generation traces from the benchmark.

## Quick Start

### 1. Environment Setup

Create a new Python environment and install the required dependencies:

```bash
conda create -n ms-cot python=3.10
conda activate ms-cot
pip install -r requirements.txt
```

### 2. Dataset Preparation

To generate the evaluation dataset from the Hugging Face MassSpecGym benchmark, navigate to the root directory and run:

```bash
mkdir -p data
python prepare_dataset.py --split full
```
This will download the requested splits and generate `benchmark_dataset_<split>.jsonl` files in the `data/` directory.

### 3. Running Model Inference

Run the generation script for the desired model. For example, to evaluate GPT-4o-mini:

```bash
export OPENAI_API_KEY="your-api-key"
python openai_4o.py --dataset data/benchmark_dataset_test.jsonl
```
Outputs will be saved as JSON lines containing the model's `<think>` trajectories and `<answer>` fields.

### 4. Running the Evaluation

To compute the benchmark quantitative metrics (Top-k Exact Match, Tanimoto Similarity, Formula Consistency, DBE Accuracy) and analyze the reasoning logic, run:

```bash
python evaluate.py --predictions path/to/generated_results.jsonl
python analyze_cot.py --predictions path/to/generated_results.jsonl
python smiles_accuracy_analysis.py --predictions path/to/generated_results.jsonl
```
