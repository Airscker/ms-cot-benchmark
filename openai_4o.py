"""
batch_ms_smiles.py
------------------
Run:   OPENAI_API_KEY=sk-... \
       python batch_ms_smiles.py \
              --dataset ./data/sft_dataset_full.jsonl \
              --batch_size 500 \
              --model "gpt-4o-mini" \
              --out_dir outputs
"""

import os, json, time, argparse, tempfile, shutil, pathlib
from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # needs OPENAI_API_KEY env var

# ---------- CLI ----------
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="./data/benchmark_dataset_full.jsonl",
                    help="Path to a .jsonl or .json dataset file")
parser.add_argument("--batch_size", type=int, default=500)
parser.add_argument("--model", default="gpt-4o-mini")
parser.add_argument("--temperature", type=float, default=0)
parser.add_argument("--max_tokens", type=int, default=4096)
parser.add_argument("--completion_window", default="24h")
parser.add_argument("--out_dir", default="./outputs")
args = parser.parse_args()

pathlib.Path(args.out_dir).mkdir(exist_ok=True)

# ---------- Load prompts ----------
if args.dataset.endswith(".jsonl"):
    with open(args.dataset) as f:
        prompts = [json.loads(line) for line in f]
else:
    with open(args.dataset) as f:
        prompts = json.load(f)
print(f"Total prompts: {len(prompts)}")



# ---------- Helper: make one batch ----------
def run_one_batch(prompts_slice, batch_idx):
    import tempfile, shutil

    if os.path.exists(
        os.path.join(args.out_dir, f"batch_{args.batch_size}_{batch_idx:04d}.jsonl")
    ):
        print(f"Skipping batch_{args.batch_size}_{batch_idx:04d} (already processed)")
        return  # Skip if batch already processed

    tmp_dir = tempfile.mkdtemp()
    tmp_path = os.path.join(tmp_dir, "batch_tmp.jsonl")

    # 1. Write correct Batch API format JSONL file
    with open(tmp_path, "w") as fout:
        for i, _prompt in enumerate(prompts_slice):
            entry = {
                "custom_id": f"batch{batch_idx:03d}_req{i:05d}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": args.model,
                    "messages": [
                        {
                            "role": "system",
                            "content": _prompt["instruction"],
                        },
                        {
                            "role": "user",
                            "content": _prompt["input"],
                        },
                    ],
                    "temperature": args.temperature,
                    "max_tokens": args.max_tokens,
                },
            }
            json.dump(entry, fout)
            fout.write("\n")

    # 2. Upload the input file to OpenAI
    file_obj = client.files.create(file=open(tmp_path, "rb"), purpose="batch")

    # 3. Create the batch job — now valid!
    batch = client.batches.create(
        input_file_id=file_obj.id,
        endpoint="/v1/chat/completions",
        completion_window=args.completion_window,
        metadata={"desc": f"MS->SMILES batch {batch_idx}"},
    )
    # print(f"Created batch {batch_idx} → id={batch.id}")

    # 4. Wait for batch to complete
    while batch.status in ("validating", "in_progress"):
        # print(f"Waiting for batch {batch_idx} (status: {batch.status}) ...")
        time.sleep(30)
        batch = client.batches.retrieve(batch.id)

    if batch.status in ["failed", "expired", "cancelling", "cancelled"]:
        print(f"Batch {batch_idx} failed with status: {batch.status}")
        return
    while batch.status == "finalizing":
        time.sleep(10)
        batch = client.batches.retrieve(batch.id)
    try:
        # 5. Download result
        time.sleep(3)
        out_path = os.path.join(
            args.out_dir, f"batch_{args.batch_size}_{batch_idx:04d}.jsonl"
        )
        output_file = client.files.content(batch.output_file_id)
        with open(out_path, "wb") as f:
            f.write(output_file.read())
    except Exception as e:
        print(f"Error downloading batch {batch_idx} output file: {e}")
        exit(0)
    # print(f"Saved batch output to {out_path}")

    shutil.rmtree(tmp_dir, ignore_errors=True)


# idx_proc=list(range(174*100,len(prompts)))
# ---------- Iterate over mini-batches ----------
bar = tqdm(range(0, len(prompts), args.batch_size))
for idx in bar:
    # if idx not in idx_proc:
    #     continue
    # if idx//10 <=355:
    #     print(f"Skipping batch_10_{idx // 10} (already processed)")
    #     continue
    bar.set_description(f"Processing batch {idx // args.batch_size}")
    batch_prompts = prompts[idx : idx + args.batch_size]
    run_one_batch(batch_prompts, idx // args.batch_size)
