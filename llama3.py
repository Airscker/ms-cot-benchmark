import os
import json
import requests
from datasets import load_dataset
from tqdm import tqdm


def query_llama3(
    prompt: str,
    model: str = "meta-llama/Llama-3.3-70B-Instruct",
    url: str = "http://localhost:8000/v1/chat/completions",
    temperature: float = 0.7,
    max_tokens: int = 1000,
) -> dict:
    """Send one prompt to the local LLaMA3 API."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error for prompt: {prompt[:80]}...\n{e}")
        return {"error": str(e)}


def run_benchmark(
    dataset_path: str, output_dir: str, model: str = "meta-llama/Llama-3.3-70B-Instruct"
):
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset
    dataset = load_dataset("json", data_files=dataset_path)["train"]
    prompts = [ex["prompt"] for ex in dataset]

    for idx, prompt in enumerate(tqdm(prompts, desc="Running prompts")):
        if idx <= 8740:
            continue
        result = query_llama3(prompt, model=model)
        # Save to JSON
        output_path = os.path.join(output_dir, f"response_{idx:05d}.json")
        with open(output_path, "w") as f:
            json.dump({"index": idx, "prompt": prompt, "response": result}, f, indent=4)


if __name__ == "__main__":
    run_benchmark(
        dataset_path="./data/sft_dataset_full.jsonl",
        output_dir="/data/yufeng/benchmark/llama3-70B-instruct",
        model="meta-llama/Llama-3.3-70B-Instruct",
    )
