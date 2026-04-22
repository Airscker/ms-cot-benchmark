import os
import json
import argparse
import requests
from tqdm import tqdm


def query_llama3(
    prompt: str,
    model: str,
    url: str = "http://localhost:8000/v1/chat/completions",
    temperature: float = 0,
    max_tokens: int = 4096,
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
    dataset_path: str,
    output_dir: str,
    model: str,
    url: str = "http://localhost:8000/v1/chat/completions",
    max_tokens: int = 4096,
    temperature: float = 0,
):
    os.makedirs(output_dir, exist_ok=True)

    with open(dataset_path) as f:
        records = [json.loads(line) for line in f]

    prompts = [ex["instruction"] + "\n" + ex["input"] for ex in records]

    for idx, prompt in enumerate(tqdm(prompts, desc="Running prompts")):
        output_path = os.path.join(output_dir, f"response_{idx:05d}.json")
        if os.path.exists(output_path):
            continue
        result = query_llama3(prompt, model=model, url=url,
                               temperature=temperature, max_tokens=max_tokens)
        with open(output_path, "w") as f:
            json.dump({"index": idx, "prompt": prompt, "response": result}, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="./data/benchmark_dataset_full.jsonl")
    parser.add_argument("--out_dir", default="./benchmark/llama3-70B-instruct")
    parser.add_argument("--model", default="meta-llama/Llama-3-70B-Instruct")
    parser.add_argument("--url", default="http://localhost:8000/v1/chat/completions")
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0)
    args = parser.parse_args()

    run_benchmark(
        dataset_path=args.dataset,
        output_dir=args.out_dir,
        model=args.model,
        url=args.url,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
