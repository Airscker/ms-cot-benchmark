"""
SMILES Accuracy Analysis - Extract SMILES from <answer></answer> blocks and analyze block presence.
"""

import json
import re
import os
import sys
from typing import List, Dict, Optional, Tuple
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import BulkTanimotoSimilarity
import rdkit
import numpy as np
rdkit.RDLogger.DisableLog('rdApp.*')

# Redirect stdout to capture printed output
class TeeOutput:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()

def extract_smiles_from_answer_block(content: str) -> str:
    """
    Extract SMILES from <answer></answer> block.
    
    Args:
        content: The prediction content
        
    Returns:
        Extracted SMILES string or empty string if not found
    """
    pattern = r'<answer>(.*?)</answer>'
    match = re.search(pattern, content, re.DOTALL)
    
    if match:
        smiles = match.group(1).strip()
        # Clean up any extra whitespace or newlines
        smiles = re.sub(r'\s+', '', smiles)
        return smiles
    
    return ""

def check_think_block_presence(content: str) -> bool:
    """
    Check if <think></think> block is present in the content.
    Also check for incomplete <think> blocks (starting with </think>).
    
    Args:
        content: The prediction content
        
    Returns:
        True if <think></think> block or incomplete think block is found
    """
    # Check for complete <think></think> blocks
    complete_pattern = r'<think>.*?</think>'
    complete_match = re.search(complete_pattern, content, re.DOTALL)
    
    # Check for incomplete think blocks (starting with </think>)
    # incomplete_pattern = r'</think>'
    # incomplete_match = re.search(incomplete_pattern, content, re.DOTALL)
    
    return complete_match is not None # or incomplete_match is not None

def check_answer_block_presence(content: str) -> bool:
    """
    Check if <answer></answer> block is present in the content.
    
    Args:
        content: The prediction content
        
    Returns:
        True if <answer></answer> block is found
    """
    pattern = r'<answer>.*?</answer>'
    match = re.search(pattern, content, re.DOTALL)
    return match is not None

def check_reasoning_content(content: str) -> bool:
    """
    Check if the content contains reasoning/analysis content.
    
    Args:
        content: The prediction content
        
    Returns:
        True if reasoning content is found
    """
    reasoning_patterns = [
        r'Step-by-Step',
        r'Reasoning:',
        r'Analysis:',
        r'Fragment Analysis',
        r'Functional Group',
        r'Molecular Formula',
        r'DBE',
        r'Double Bond Equivalent',
        r'### \d+\.',
        r'#### \d+\.',
        r'Let\'s',
        r'Given the',
        r'Based on',
        r'Therefore',
        r'Thus',
        r'Hence'
    ]
    
    for pattern in reasoning_patterns:
        if re.search(pattern, content, re.IGNORECASE):
            return True
    
    return False

def canonical_smiles(smiles: str) -> Optional[str]:
    """
    Convert SMILES to canonical form using RDKit.
    
    Args:
        smiles: SMILES string to canonicalize
        
    Returns:
        Canonical SMILES or None if invalid
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, isomericSmiles=True)
    except:
        return None

def calculate_tanimoto_similarity(smiles1: str, smiles2: str) -> float:
    """
    Calculate Tanimoto similarity between two SMILES strings using Morgan fingerprints.
    
    Args:
        smiles1: First SMILES string
        smiles2: Second SMILES string
        
    Returns:
        Tanimoto similarity score (0.0 to 1.0) or 0.0 if invalid
    """
    try:
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        
        if mol1 is None or mol2 is None:
            return 0.0
        
        # Generate Morgan fingerprints
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)
        
        # Calculate Tanimoto similarity
        similarity = BulkTanimotoSimilarity(fp1, [fp2])[0]
        return similarity
    except:
        return 0.0

def calculate_mecs_similarity(smiles1: str, smiles2: str) -> float:
    """
    Calculate MECS (Molecular Embedding Cosine Similarity) between two SMILES strings.
    This is a simplified version using Morgan fingerprint vectors.
    
    Args:
        smiles1: First SMILES string
        smiles2: Second SMILES string
        
    Returns:
        MECS similarity score (0.0 to 1.0) or 0.0 if invalid
    """
    try:
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        
        if mol1 is None or mol2 is None:
            return 0.0
        
        # Generate Morgan fingerprints as vectors
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)
        
        # Convert to numpy arrays
        vec1 = np.array(fp1)
        vec2 = np.array(fp2)
        
        # Calculate cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        cosine_similarity = dot_product / (norm1 * norm2)
        return float(cosine_similarity)
    except:
        return 0.0

def normalize_smiles(smiles: str) -> str:
    """
    Normalize SMILES string for comparison.
    
    Args:
        smiles: SMILES string to normalize
        
    Returns:
        Normalized SMILES string
    """
    # Remove stereochemistry indicators for comparison
    normalized = smiles.replace('@', '')
    # Convert to lowercase
    normalized = normalized.lower()
    return normalized

def calculate_exact_match(predicted_smiles: str, reference_smiles: str) -> bool:
    """
    Check if predicted SMILES exactly matches the reference.
    
    Args:
        predicted_smiles: Predicted SMILES string
        reference_smiles: Reference SMILES string
        
    Returns:
        True if there's an exact match
    """
    if not predicted_smiles or not reference_smiles:
        return False
    
    # Try canonical comparison first
    pred_canonical = canonical_smiles(predicted_smiles)
    ref_canonical = canonical_smiles(reference_smiles)
    
    if pred_canonical and ref_canonical:
        return pred_canonical == ref_canonical
    
    # Fall back to normalized comparison
    pred_normalized = normalize_smiles(predicted_smiles)
    ref_normalized = normalize_smiles(reference_smiles)
    
    return pred_normalized == ref_normalized

def process_and_analyze(input_file: str, extracted_output: str, accuracy_output: str):
    """
    Process the detailed predictions file, extract SMILES from <answer> blocks, and calculate accuracy.
    
    Args:
        input_file: Path to the detailed_predictions.json file
        extracted_output: Output file for extracted SMILES
        accuracy_output: Output file for accuracy results
    """
    print("Loading predictions file...")
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    print("Extracting SMILES from <answer> blocks and analyzing block presence...")
    results = {}
    
    think_block_count = 0
    answer_block_count = 0
    reasoning_content_count = 0
    valid_smiles_count = 0
    
    for item in data:
        item_id = item['id']
        prediction = item.get('prediction', '')
        reference = item.get('reference', '')
        
        # Extract SMILES from <answer> block
        extracted_smiles = extract_smiles_from_answer_block(prediction)
        
        # Check block presence
        has_think_block = check_think_block_presence(prediction)
        has_answer_block = check_answer_block_presence(prediction)
        has_reasoning_content = check_reasoning_content(prediction)
        
        # Count blocks
        if has_think_block:
            think_block_count += 1
        if has_answer_block:
            answer_block_count += 1
        if has_reasoning_content:
            reasoning_content_count += 1
        if extracted_smiles:
            valid_smiles_count += 1
        
        results[item_id] = {
            'prediction': prediction,
            'reference': reference,
            'extracted_smiles': extracted_smiles,
            'has_think_block': has_think_block,
            'has_answer_block': has_answer_block,
            'has_reasoning_content': has_reasoning_content,
            'has_valid_smiles': bool(extracted_smiles)
        }
    
    print("Saving extracted SMILES...")
    with open(extracted_output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("Calculating accuracy and similarity metrics...")
    total_items = len(results)
    exact_matches = 0
    tanimoto_scores = []
    mecs_scores = []
    
    accuracy_results = {}
    
    for item_id, item in results.items():
        predicted_smiles = item.get('extracted_smiles', '')
        reference_smiles = item.get('reference', '')
        
        if predicted_smiles and reference_smiles:
            is_exact_match = calculate_exact_match(predicted_smiles, reference_smiles)
            if is_exact_match:
                exact_matches += 1
            
            # Calculate similarity metrics
            tanimoto_score = calculate_tanimoto_similarity(predicted_smiles, reference_smiles)
            mecs_score = calculate_mecs_similarity(predicted_smiles, reference_smiles)
            
            tanimoto_scores.append(tanimoto_score)
            mecs_scores.append(mecs_score)
            
            accuracy_results[item_id] = {
                'exact_match': is_exact_match,
                'predicted_smiles': predicted_smiles,
                'reference_smiles': reference_smiles,
                'tanimoto_similarity': tanimoto_score,
                'mecs_similarity': mecs_score,
                'has_think_block': item['has_think_block'],
                'has_answer_block': item['has_answer_block']
            }
        else:
            accuracy_results[item_id] = {
                'exact_match': False,
                'predicted_smiles': predicted_smiles,
                'reference_smiles': reference_smiles,
                'tanimoto_similarity': 0.0,
                'mecs_similarity': 0.0,
                'has_think_block': item['has_think_block'],
                'has_answer_block': item['has_answer_block']
            }
    
    # Calculate metrics
    overall_accuracy = exact_matches / total_items if total_items > 0 else 0
    prediction_accuracy = exact_matches / valid_smiles_count if valid_smiles_count > 0 else 0
    think_block_ratio = think_block_count / total_items if total_items > 0 else 0
    answer_block_ratio = answer_block_count / total_items if total_items > 0 else 0
    reasoning_content_ratio = reasoning_content_count / total_items if total_items > 0 else 0
    valid_smiles_ratio = valid_smiles_count / total_items if total_items > 0 else 0
    
    # Calculate similarity statistics
    avg_tanimoto = np.mean(tanimoto_scores) if tanimoto_scores else 0.0
    avg_mecs = np.mean(mecs_scores) if mecs_scores else 0.0
    median_tanimoto = np.median(tanimoto_scores) if tanimoto_scores else 0.0
    median_mecs = np.median(mecs_scores) if mecs_scores else 0.0
    
    metrics = {
        'total_items': total_items,
        'items_with_valid_smiles': valid_smiles_count,
        'exact_matches': exact_matches,
        'overall_accuracy': overall_accuracy,
        'prediction_accuracy': prediction_accuracy,
        'coverage': valid_smiles_ratio,
        'think_block_count': think_block_count,
        'answer_block_count': answer_block_count,
        'reasoning_content_count': reasoning_content_count,
        'think_block_ratio': think_block_ratio,
        'answer_block_ratio': answer_block_ratio,
        'reasoning_content_ratio': reasoning_content_ratio,
        'avg_tanimoto_similarity': avg_tanimoto,
        'avg_mecs_similarity': avg_mecs,
        'median_tanimoto_similarity': median_tanimoto,
        'median_mecs_similarity': median_mecs,
        'tanimoto_scores': tanimoto_scores,
        'mecs_scores': mecs_scores
    }
    
    print("Saving accuracy results...")
    output_data = {
        'metrics': metrics,
        'results': accuracy_results
    }
    
    with open(accuracy_output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    return metrics, results

def print_summary(metrics: Dict, results: Dict):
    """
    Print a summary of the extraction and accuracy results.
    
    Args:
        metrics: The accuracy metrics
        results: The extracted results
    """
    total_items = len(results)
    
    print("\n" + "="*80)
    print("SMILES EXTRACTION AND ACCURACY ANALYSIS SUMMARY")
    print("="*80)
    
    print(f"\nBLOCK PRESENCE ANALYSIS:")
    print(f"  Total items processed: {total_items}")
    print(f"  Items with <think></think> blocks (or incomplete): {metrics['think_block_count']} ({metrics['think_block_ratio']:.3f} - {metrics['think_block_ratio']*100:.1f}%)")
    print(f"  Items with <answer></answer> blocks: {metrics['answer_block_count']} ({metrics['answer_block_ratio']:.3f} - {metrics['answer_block_ratio']*100:.1f}%)")
    print(f"  Items with reasoning content: {metrics['reasoning_content_count']} ({metrics['reasoning_content_ratio']:.3f} - {metrics['reasoning_content_ratio']*100:.1f}%)")
    print(f"  Items with valid SMILES: {metrics['items_with_valid_smiles']} ({metrics['coverage']:.3f} - {metrics['coverage']*100:.1f}%)")
    
    print(f"\nACCURACY RESULTS:")
    print(f"  Exact matches: {metrics['exact_matches']}")
    print(f"  Overall accuracy: {metrics['overall_accuracy']:.3f} ({metrics['overall_accuracy']*100:.1f}%)")
    print(f"  Prediction accuracy: {metrics['prediction_accuracy']:.3f} ({metrics['prediction_accuracy']*100:.1f}%)")
    
    print(f"\nSIMILARITY METRICS:")
    print(f"  Average Tanimoto similarity: {metrics['avg_tanimoto_similarity']:.3f}")
    print(f"  Median Tanimoto similarity: {metrics['median_tanimoto_similarity']:.3f}")
    print(f"  Average MECS similarity: {metrics['avg_mecs_similarity']:.3f}")
    print(f"  Median MECS similarity: {metrics['median_mecs_similarity']:.3f}")
    
    # Show some examples
    print(f"\nEXAMPLES OF EXTRACTED SMILES:")
    count = 0
    for item_id, item in results.items():
        if item['has_valid_smiles'] and count < 5:
            print(f"  ID {item_id}: {item['extracted_smiles'][:100]}...")  # Show first 100 chars
            count += 1

if __name__ == "__main__":
    _path = './msllm/sft/evaluation_results/'
    input_file = os.path.join(_path, 'detailed_predictions.json')
    extracted_output = os.path.join(_path, 'extracted_smiles.json')
    accuracy_output = os.path.join(_path, 'smiles_accuracy.json')
    log_output = os.path.join(_path, 'analysis_log.txt')
    
    # Redirect output to both terminal and file
    tee_output = TeeOutput(log_output)
    sys.stdout = tee_output
    
    try:
        print("Starting SMILES extraction and accuracy analysis...")
        print(f"Analysis started at: {__import__('datetime').datetime.now()}")
        print("="*80)
        
        metrics, results = process_and_analyze(input_file, extracted_output, accuracy_output)
        
        print_summary(metrics, results)
        
        print(f"\nFiles saved:")
        print(f"  - Extracted SMILES: {extracted_output}")
        print(f"  - Accuracy results: {accuracy_output}")
        print(f"  - Analysis log: {log_output}")
        
        print(f"\nAnalysis completed at: {__import__('datetime').datetime.now()}")
        
    finally:
        # Restore stdout and close log file
        sys.stdout = tee_output.terminal
        tee_output.close() 