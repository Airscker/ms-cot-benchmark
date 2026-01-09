import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
import json
import time
import os
from typing import List, Dict, Any
from datetime import datetime
from datasets import load_dataset
from tqdm import tqdm


class MSAnalysisBatchProcessor:
    def __init__(self, api_key: str):
        """
        Initialize the batch processor with Claude API key

        Args:
            api_key (str): Your Anthropic API key
        """
        self.client = anthropic.Anthropic(api_key=api_key)
        self.system_prompt = """You are an expert in analyzing Mass Spectra (MS) to deduce the corresponding SMILES. 
        You will analyze MS data and provide detailed structural analysis following a specific template format."""

    def create_batch_requests(
        self,
        prompts: List[str],
        model: str = "claude-3-5-sonnet-20241022",
        max_tokens: int = 1000,
    ) -> List[Request]:
        """
        Create batch requests from list of prompts

        Args:
            prompts (list): List of prepared prompts
            model (str): Claude model to use
            max_tokens (int): Maximum tokens per response

        Returns:
            list: List of Request objects for batch processing
        """
        requests = []

        for i, prompt in enumerate(prompts):
            request = Request(
                custom_id=f"ms-analysis-{i+1}",
                params=MessageCreateParamsNonStreaming(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=0.1,
                    system=self.system_prompt,
                    messages=[{"role": "user", "content": prompt}],
                ),
            )
            requests.append(request)

        return requests

    def submit_batch(self, requests: List[Request]) -> str:
        """
        Submit batch requests to Claude API

        Args:
            requests (list): List of Request objects

        Returns:
            str: Batch ID for tracking
        """
        # print(f"Submitting batch with {len(requests)} requests...")

        message_batch = self.client.messages.batches.create(requests=requests)

        # print(f"Batch submitted successfully!")
        # print(f"Batch ID: {message_batch.id}")
        # print(f"Status: {message_batch.processing_status}")
        # print(f"Request count: {message_batch.request_counts}")

        return message_batch.id

    def check_batch_status(self, batch_id: str) -> Dict[str, Any]:
        """
        Check the status of a submitted batch

        Args:
            batch_id (str): Batch ID to check

        Returns:
            dict: Batch status information
        """
        batch_info = self.client.messages.batches.retrieve(batch_id)

        status_info = {
            "id": batch_info.id,
            "processing_status": batch_info.processing_status,
            "request_counts": batch_info.request_counts,
            "created_at": batch_info.created_at,
            "expires_at": batch_info.expires_at,
        }

        if hasattr(batch_info, "ended_at") and batch_info.ended_at:
            status_info["ended_at"] = batch_info.ended_at

        return status_info

    def wait_for_batch_completion(
        self, batch_id: str, check_interval: int = 30, timeout_minutes: int = 60
    ) -> bool:
        """
        Wait for batch to complete with periodic status checks

        Args:
            batch_id (str): Batch ID to monitor
            check_interval (int): Seconds between status checks
            timeout_minutes (int): Maximum time to wait in minutes

        Returns:
            bool: True if completed successfully, False if timeout/error
        """
        start_time = time.time()
        timeout_seconds = timeout_minutes * 60

        # print(f"Waiting for batch {batch_id} to complete...")
        # print(
        #     f"Checking every {check_interval} seconds (timeout: {timeout_minutes} minutes)"
        # )

        while True:
            status_info = self.check_batch_status(batch_id)
            status = status_info["processing_status"]

            # print(f"Status: {status} | Requests: {status_info['request_counts']}")

            if status == "ended":
                # print("✓ Batch completed successfully!")
                return True
            elif status in ["failed", "canceled", "expired"]:
                # print(f"✗ Batch failed with status: {status}")
                return False

            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > timeout_seconds:
                # print(f"✗ Timeout reached ({timeout_minutes} minutes)")
                return False

            time.sleep(check_interval)

    def retrieve_batch_results(self, batch_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve results from completed batch

        Args:
            batch_id (str): Batch ID to retrieve results from

        Returns:
            list: List of result dictionaries
        """
        # print(f"Retrieving results for batch {batch_id}...")

        # Get batch results
        results = self.client.messages.batches.results(batch_id)

        processed_results = []

        for result in results:
            result_data = {
                "custom_id": result.custom_id,
                "success": (
                    result.result.type == "succeeded"
                    if hasattr(result.result, "type")
                    else True
                ),
                "timestamp": datetime.now().isoformat(),
            }

            if hasattr(result.result, "message") and result.result.message:
                result_data["analysis"] = result.result.message.content[0].text
                result_data["error"] = None
            elif hasattr(result.result, "error") and result.result.error:
                result_data["analysis"] = None
                result_data["error"] = str(result.result.error)
                result_data["success"] = False
            else:
                result_data["analysis"] = None
                result_data["error"] = "Unknown result format"
                result_data["success"] = False

            processed_results.append(result_data)

        # print(f"Retrieved {len(processed_results)} results")
        return processed_results

    def save_results(self, results: List[Dict[str, Any]], output_file: str):
        """
        Save results to JSON file

        Args:
            results (list): List of analysis results
            output_file (str): Output file path
        """
        # Add metadata
        output_data = {
            "metadata": {
                "total_results": len(results),
                "successful_results": sum(1 for r in results if r["success"]),
                "failed_results": sum(1 for r in results if not r["success"]),
                "generated_at": datetime.now().isoformat(),
            },
            "results": results,
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)

        # print(f"Results saved to {output_file}")

    def process_batches(
        self,
        prompts: List[str],
        batch_size: int = 50,
        output_prefix: str = "batch_results",
        model: str = "claude-3-5-sonnet-20241022",
        wait_for_completion: bool = True,
    ) -> List[str]:
        """
        Process prompts in batches and save results

        Args:
            prompts (list): List of prepared prompts
            batch_size (int): Number of prompts per batch
            output_prefix (str): Prefix for output filenames
            model (str): Claude model to use
            wait_for_completion (bool): Whether to wait for completion and retrieve results

        Returns:
            list: List of batch IDs
        """
        total_prompts = len(prompts)
        num_batches = (total_prompts + batch_size - 1) // batch_size

        print(
            f"Processing {total_prompts} prompts in {num_batches} batches of {batch_size}"
        )

        batch_ids = []
        bar=tqdm(range(num_batches), mininterval=1)
        for batch_num in bar:
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, total_prompts)
            batch_prompts = prompts[start_idx:end_idx]

            # print(f"\n--- Processing Batch {batch_num + 1}/{num_batches} ---")
            # print(f"Prompts {start_idx + 1}-{end_idx} of {total_prompts}")
            bar.set_description(f"Processing Batch {batch_num + 1}/{num_batches}")

            # Create and submit batch
            requests = self.create_batch_requests(batch_prompts, model=model)
            batch_id = self.submit_batch(requests)
            batch_ids.append(batch_id)

            if wait_for_completion:
                # Wait for completion
                success = self.wait_for_batch_completion(batch_id)

                if success:
                    # Retrieve and save results
                    results = self.retrieve_batch_results(batch_id)
                    output_file = (
                        f"{output_prefix}/batch_{batch_num:05d}_{batch_id[:8]}.json"
                    )
                    self.save_results(results, output_file)
                else:
                    print(f"Batch {batch_num + 1} failed or timed out")

            # Small delay between batch submissions
            if batch_num < num_batches - 1:
                time.sleep(2)

        return batch_ids

    def extract_smiles_from_results(self, results_file: str) -> Dict[str, Any]:
        """
        Extract SMILES from saved results file

        Args:
            results_file (str): Path to results JSON file

        Returns:
            dict: Extracted SMILES data
        """
        with open(results_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        extracted_smiles = []

        for result in data["results"]:
            if result["success"] and result["analysis"]:
                # Extract SMILES from <answer> block
                analysis_text = result["analysis"]
                smiles_list = []

                if "<answer>" in analysis_text and "</answer>" in analysis_text:
                    answer_block = analysis_text.split("<answer>")[1].split(
                        "</answer>"
                    )[0]
                    # Extract individual SMILES
                    if "SMILES Proposals:" in answer_block:
                        smiles_part = answer_block.split("SMILES Proposals:")[1].strip()
                        smiles_list = [
                            s.strip() for s in smiles_part.split(",") if s.strip()
                        ]

                extracted_smiles.append(
                    {
                        "custom_id": result["custom_id"],
                        "smiles_candidates": smiles_list,
                        "success": True,
                    }
                )
            else:
                extracted_smiles.append(
                    {
                        "custom_id": result["custom_id"],
                        "smiles_candidates": [],
                        "success": False,
                        "error": result.get("error"),
                    }
                )

        return {"metadata": data["metadata"], "extracted_smiles": extracted_smiles}


# Example usage
def main():
    # Initialize processor with your API key
    # api_key = os.getenv("ANTHROPIC_API_KEY")
    # if not api_key:
    # raise ValueError("Please set ANTHROPIC_API_KEY environment variable")
    api_key = os.getenv("ANTHROPIC_API_KEY")
    processor = MSAnalysisBatchProcessor(api_key)

    # Example: Use prepared prompts
    dataset = load_dataset("json", data_files="./data/sft_dataset_full.jsonl")["train"]
    prompts=[dataset[i]['prompt'] for i in range(len(dataset))]
    # Process in batches
    batch_ids = processor.process_batches(
        prompts=prompts,
        batch_size=10,  # Adjust batch size as needed
        output_prefix="./benchmark/claude-3.5-sonnet-20241022",
        model="claude-3-5-sonnet-20241022",
        wait_for_completion=True,
    )

    print(f"\nAll batches processed. Batch IDs: {batch_ids}")

    # Example: Extract SMILES from results (if you have result files)
    # smiles_data = processor.extract_smiles_from_results('ms_analysis_results_batch_1_xxx.json')
    # print(f"Extracted SMILES: {smiles_data}")


if __name__ == "__main__":
    main()
