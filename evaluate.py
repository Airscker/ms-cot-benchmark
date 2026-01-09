import json
import re
import logging
import rdkit
from typing import List, Dict, Tuple, Optional
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, rdFMCS, DataStructs, Descriptors
from rdkit.Chem import AllChem
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
from prettytable import PrettyTable

# Suppress RDKit warnings
logging.getLogger('rdkit').setLevel(logging.ERROR)
rdkit.RDLogger.DisableLog('rdApp.*')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())
logger.addHandler(logging.FileHandler('msllm/benchmark/evaluation.log'))

def load_answers(file_path: str) -> List[str]:
    """Load answers from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def extract_blocks(text: str) -> Tuple[bool, bool, str]:
    """
    Extract <think> and <answer> blocks from text.
    Returns: (has_think, has_answer, answer_content)
    """
    has_think = '<think>' in text and '</think>' in text
    has_answer = '<answer>' in text and '</answer>' in text
    
    answer_content = ""
    if has_answer:
        # Extract content between <answer> and </answer>
        match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
        if match:
            answer_content = match.group(1).strip()
    
    return has_think, has_answer, answer_content

def extract_dbe_value(text: str) -> Optional[float]:
    """
    Extract DBE value from text.
    Looks for patterns like "DBE: 9.0" or "Double Bond Equivalents (DBE): 8.0".
    """
    # Look for "Double Bond Equivalents (DBE):" followed by number
    dbe_match = re.search(r'Double Bond Equivalents \(DBE\):\s*([0-9]+\.?[0-9]*)', text, re.IGNORECASE)
    if dbe_match:
        return float(dbe_match.group(1))
    
    # Look for "DBE:" followed by number
    dbe_match = re.search(r'DBE:\s*([0-9]+\.?[0-9]*)', text, re.IGNORECASE)
    if dbe_match:
        return float(dbe_match.group(1))
    
    # Look for "Double Bond Equivalents:" followed by number
    dbe_match = re.search(r'Double Bond Equivalents:\s*([0-9]+\.?[0-9]*)', text, re.IGNORECASE)
    if dbe_match:
        return float(dbe_match.group(1))
    
    return None

def calculate_dbe_from_formula(formula: str) -> float:
    """
    Calculate DBE from molecular formula.
    DBE = C - H/2 + N/2 + 1
    """
    # Parse formula to get atom counts
    atoms = {}
    current_atom = ""
    current_count = ""
    
    for char in formula:
        if char.isupper():
            # Save previous atom if exists
            if current_atom:
                count = int(current_count) if current_count else 1
                atoms[current_atom] = atoms.get(current_atom, 0) + count
            # Start new atom
            current_atom = char
            current_count = ""
        elif char.islower():
            current_atom += char
        elif char.isdigit():
            current_count += char
    
    # Don't forget the last atom
    if current_atom:
        count = int(current_count) if current_count else 1
        atoms[current_atom] = atoms.get(current_atom, 0) + count
    
    # Calculate DBE
    C = atoms.get('C', 0)
    H = atoms.get('H', 0)
    N = atoms.get('N', 0)
    
    dbe = C - H/2 + N/2 + 1
    return dbe

def calculate_tanimoto_similarity(smiles1: str, smiles2: str) -> float:
    """
    Calculate Tanimoto similarity between two SMILES strings using Morgan fingerprints.
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
        similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
        return similarity
    except:
        return 0.0

def calculate_mces_similarity(smiles1: str, smiles2: str) -> float:
    """
    Calculate MCES (Maximum Common Edge Subgraph) similarity between two SMILES strings.
    Returns the ratio of common edges to total edges in the larger molecule.
    """
    try:
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        
        if mol1 is None or mol2 is None:
            return 0.0
        
        # Find maximum common substructure
        mcs = rdFMCS.FindMCS([mol1, mol2], 
                            atomCompare=rdFMCS.AtomCompare.CompareElements,
                            bondCompare=rdFMCS.BondCompare.CompareOrder,
                            ringMatchesRingOnly=True,
                            completeRingsOnly=True)
        
        if mcs.numBonds == 0:
            return 0.0
        
        # Calculate similarity as ratio of common edges to total edges in larger molecule
        edges1 = mol1.GetNumBonds()
        edges2 = mol2.GetNumBonds()
        max_edges = max(edges1, edges2)
        
        if max_edges == 0:
            return 0.0
        
        similarity = mcs.numBonds / max_edges
        return similarity
    except:
        return 0.0

def calculate_top_k_similarity_metrics(predicted_smiles_list: List[str], true_smiles: str, k: int) -> Tuple[float, float]:
    """
    Calculate Top-k Maximum Tanimoto Similarity and Top-k Minimum MCES for a list of predicted SMILES.
    Returns (max_tanimoto, min_mces) where min_mces is actually the maximum MCES (higher is better).
    """
    if not predicted_smiles_list:
        return 0.0, 0.0
    
    top_k_predictions = predicted_smiles_list[:k]
    
    # Calculate Tanimoto similarities
    tanimoto_similarities = [calculate_tanimoto_similarity(pred, true_smiles) for pred in top_k_predictions]
    max_tanimoto = max(tanimoto_similarities) if tanimoto_similarities else 0.0
    
    # Calculate MCES similarities
    mces_similarities = [calculate_mces_similarity(pred, true_smiles) for pred in top_k_predictions]
    min_mces = min(mces_similarities) if mces_similarities else 0.0
    
    return max_tanimoto, min_mces

def verify_formula_match(smiles: str, target_formula: str) -> bool:
    """
    Verify if the SMILES matches the target molecular formula.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        
        # Get molecular formula from SMILES
        predicted_formula = rdMolDescriptors.CalcMolFormula(mol)
        
        # Normalize formulas for comparison
        target_formula = re.sub(r'([A-Z][a-z]?)1', r'\1', target_formula)  # Remove 1s
        predicted_formula = re.sub(r'([A-Z][a-z]?)1', r'\1', predicted_formula)  # Remove 1s
        
        return target_formula == predicted_formula
    except Exception as e:
        return False

def extract_smiles_from_answer(answer_content: str) -> List[str]:
    """
    Extract SMILES strings from answer content.
    Looks for patterns like "SMILES: ..." or just SMILES strings.
    """
    smiles_list = []
    
    # Split by common delimiters
    parts = re.split(r'[,;\n]', answer_content)
    
    for part in parts:
        if 'Proposal' in part:
            continue
        part = part.strip()
        # remove prefixes like 1. 2. 3.
        part = re.sub(r'^\d+\.\s*', '', part)
        if not part:
            continue
        
        # Check if it looks like a SMILES string
        if part and re.match(r'^[A-Za-z0-9@+\-\[\]\(\)=#%\.]+$', part):
            try:
                # Try to parse as SMILES
                mol = Chem.MolFromSmiles(part)
                if mol is not None:
                    # Convert to canonical SMILES
                    canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
                    smiles_list.append(canonical_smiles)
            except:
                continue
    
    return smiles_list

def calculate_accuracy(predicted_smiles_list: List[str], true_smiles: str, top_k: int = 1) -> bool:
    """
    Calculate if the true SMILES is in the top-k predicted SMILES.
    """
    if not predicted_smiles_list:
        return False
    
    # Convert true SMILES to canonical form
    try:
        true_mol = Chem.MolFromSmiles(true_smiles)
        if true_mol is None:
            return False
        canonical_true = Chem.MolToSmiles(true_mol, canonical=True)
    except:
        return False
    
    # Check if canonical true SMILES is in top-k predictions
    top_k_predictions = predicted_smiles_list[:top_k]
    return canonical_true in top_k_predictions

def compute_metrics_from_results(results: List[Dict]) -> Dict:
    """
    Calculate aggregated metrics from a list of results.
    """
    total_samples = len(results)
    if total_samples == 0:
        return {
            'total_samples': 0,
            'has_think_rate': 0.0,
            'has_answer_rate': 0.0,
            'valid_smiles_rate': 0.0,
            'top1_accuracy': 0.0,
            'top10_accuracy': 0.0,
            'top1_accuracy_valid': 0.0,
            'top10_accuracy_valid': 0.0,
            'dbe_accuracy': 0.0,
            'dbe_evaluated_rate': 0.0,
            'formula_match_rate': 0.0,
            'formula_evaluated_rate': 0.0,
            'top1_tanimoto_avg': 0.0,
            'top10_tanimoto_avg': 0.0,
            'top1_mces_avg': 0.0,
            'top10_mces_avg': 0.0,
            'similarity_evaluated_rate': 0.0
        }

    has_think_count = sum(1 for r in results if r.get('has_think'))
    has_answer_count = sum(1 for r in results if r.get('has_answer'))
    
    valid_smiles_list = [r for r in results if r.get('valid_predicted_smiles')]
    valid_smiles_count = len(valid_smiles_list)
    
    top1_correct = sum(1 for r in valid_smiles_list if r.get('top1_correct'))
    top10_correct = sum(1 for r in valid_smiles_list if r.get('top10_correct'))
    
    # DBE
    dbe_evaluated = [r for r in results if r.get('extracted_dbe') is not None]
    dbe_evaluated_count = len(dbe_evaluated)
    dbe_correct_count = sum(1 for r in dbe_evaluated if r.get('dbe_correct'))
    
    # Formula
    formula_evaluated = [r for r in results if r.get('formula_match_score') is not None]
    formula_evaluated_count = len(formula_evaluated)
    formula_match_sum = sum(r['formula_match_score'] for r in formula_evaluated)
    
    # Similarity
    similarity_evaluated = valid_smiles_list
    similarity_evaluated_count = len(similarity_evaluated)
    
    top1_tanimoto_sum = sum(r.get('top1_tanimoto', 0.0) for r in similarity_evaluated)
    top10_tanimoto_sum = sum(r.get('top10_tanimoto', 0.0) for r in similarity_evaluated)
    top1_mces_sum = sum(r.get('top1_mces', 0.0) for r in similarity_evaluated)
    top10_mces_sum = sum(r.get('top10_mces', 0.0) for r in similarity_evaluated)

    return {
        'total_samples': total_samples,
        'has_think_rate': has_think_count / total_samples,
        'has_answer_rate': has_answer_count / total_samples,
        'valid_smiles_rate': valid_smiles_count / total_samples,
        'top1_accuracy': top1_correct / total_samples,
        'top10_accuracy': top10_correct / total_samples,
        'top1_accuracy_valid': top1_correct / valid_smiles_count if valid_smiles_count > 0 else 0,
        'top10_accuracy_valid': top10_correct / valid_smiles_count if valid_smiles_count > 0 else 0,
        'dbe_accuracy': dbe_correct_count / dbe_evaluated_count if dbe_evaluated_count > 0 else 0,
        'dbe_evaluated_rate': dbe_evaluated_count / total_samples,
        'formula_match_rate': formula_match_sum / formula_evaluated_count if formula_evaluated_count > 0 else 0,
        'formula_evaluated_rate': formula_evaluated_count / total_samples,
        'top1_tanimoto_avg': top1_tanimoto_sum / similarity_evaluated_count if similarity_evaluated_count > 0 else 0,
        'top10_tanimoto_avg': top10_tanimoto_sum / similarity_evaluated_count if similarity_evaluated_count > 0 else 0,
        'top1_mces_avg': top1_mces_sum / similarity_evaluated_count if similarity_evaluated_count > 0 else 0,
        'top10_mces_avg': top10_mces_sum / similarity_evaluated_count if similarity_evaluated_count > 0 else 0,
        'similarity_evaluated_rate': similarity_evaluated_count / total_samples
    }

def evaluate_model_answers(answers: List[str], true_smiles_list: List[str], true_formulas: List[str]) -> Dict:
    """
    Evaluate model answers for a given model.
    """
    results = []
    
    for i, (answer, true_smiles, true_formula) in enumerate(zip(answers, true_smiles_list, true_formulas)):
        has_think, has_answer, answer_content = extract_blocks(answer)
        
        # Extract SMILES from answer
        predicted_smiles = extract_smiles_from_answer(answer_content)
        
        # Extract molecular formula and DBE from think block
        think_content = ""
        if has_think:
            think_match = re.search(r'<think>(.*?)</think>', answer, re.DOTALL)
            if think_match:
                think_content = think_match.group(1)
        
        extracted_dbe = extract_dbe_value(think_content)
        
        # Calculate true DBE
        true_dbe = calculate_dbe_from_formula(true_formula)
        
        # Check DBE consistency
        dbe_correct = None
        if extracted_dbe is not None:
            dbe_correct = abs(extracted_dbe - true_dbe) < 0.1  # Allow small floating point differences
        
        # Check formula matching
        formula_match_score = None
        formula_match_bool = None
        if len(predicted_smiles) > 0:
            valid = []
            for _smiles in predicted_smiles[:10]:  # Check top 10 predictions
                if verify_formula_match(_smiles, true_formula):
                    valid.append(1.0)
                else:
                    valid.append(0.0)
            formula_match_score = np.mean(valid)
            formula_match_bool = any(v > 0 for v in valid)
        
        # Calculate similarity metrics
        top1_tanimoto, top1_mces = 0.0, 0.0
        top10_tanimoto, top10_mces = 0.0, 0.0
        
        if predicted_smiles:
            top1_tanimoto, top1_mces = calculate_top_k_similarity_metrics(predicted_smiles, true_smiles, 1)
            top10_tanimoto, top10_mces = calculate_top_k_similarity_metrics(predicted_smiles, true_smiles, 10)
        
        # Calculate accuracy
        top1_correct = False
        top10_correct = False
        if predicted_smiles:
            top1_correct = calculate_accuracy(predicted_smiles, true_smiles, top_k=1)
            top10_correct = calculate_accuracy(predicted_smiles, true_smiles, top_k=10)
        
        # Store individual results
        results.append({
            'index': i,
            'has_think': has_think,
            'has_answer': has_answer,
            'valid_predicted_smiles': predicted_smiles,
            'true_smiles': true_smiles,
            'extracted_dbe': extracted_dbe,
            'true_formula': true_formula,
            'true_dbe': true_dbe,
            'dbe_correct': dbe_correct,
            'formula_match_score': formula_match_score,
            'formula_match': formula_match_bool,
            'top1_correct': top1_correct,
            'top10_correct': top10_correct,
            'top1_tanimoto': top1_tanimoto,
            'top10_tanimoto': top10_tanimoto,
            'top1_mces': top1_mces,
            'top10_mces': top10_mces
        })
    
    # Calculate overall metrics
    metrics = compute_metrics_from_results(results)
    
    # Calculate metrics per molecular weight bin
    bins = {
        "0-200": [],
        "200-400": [],
        "400-600": [],
        "600-800": [],
        "800+": []
    }
    
    for res in results:
        try:
            mol = Chem.MolFromSmiles(res['true_smiles'])
            if mol:
                mw = Descriptors.ExactMolWt(mol)
                if mw < 200: bins["0-200"].append(res)
                elif mw < 400: bins["200-400"].append(res)
                elif mw < 600: bins["400-600"].append(res)
                elif mw < 800: bins["600-800"].append(res)
                else: bins["800+"].append(res)
        except:
            continue
    
    binned_metrics = {}
    for bin_name, bin_results in bins.items():
        binned_metrics[bin_name] = compute_metrics_from_results(bin_results)
    
    metrics['binned_metrics'] = binned_metrics
    metrics['detailed_results'] = results
    
    return metrics

def main():
    # Load dataset to get true SMILES and formulas
    logger.info("Loading dataset...")
    dataset = load_dataset(
        "roman-bushuiev/MassSpecGym",
        cache_dir="/Users/airskcer/Library/CloudStorage/OneDrive-Personal/StonyBrook/AI4SCI/MS2Mol/MSMOL/data/MSGymDataset_HF",
    )
    true_smiles_list = dataset['val']['smiles']
    true_formulas = dataset['val']['formula']  # Assuming formula field exists
    
    # Load model answers
    logger.info("Loading model answers...")
    models = {
        'claude-3.5-sonnet': 'msllm/benchmark/claude_answers.json',
        'gpt-4o-mini': 'msllm/benchmark/gpt4o_answers.json',
        'llama_3_8b': 'msllm/benchmark/llama_3_8b_answers.json',
        'llama_70b': 'msllm/benchmark/llama_70b_answers.json'
    }
    
    all_results = {}
    
    bar=tqdm(models.items(), desc="Evaluating models")
    for model_name, file_path in bar:
        logger.info(f"\nEvaluating {model_name}...")
        try:
            answers = load_answers(file_path)
            # Ensure we have the same number of answers as true labels
            min_len = min(len(answers), len(true_smiles_list), len(true_formulas))
            answers = answers[:min_len]
            true_smiles_subset = true_smiles_list[:min_len]
            true_formulas_subset = true_formulas[:min_len]
            
            metrics = evaluate_model_answers(answers, true_smiles_subset, true_formulas_subset)
            all_results[model_name] = metrics
            
            # Print summary
            logger.info(model_name)
            logger.info(f"  Total samples: {metrics['total_samples']}")
            logger.info(f"  Has <think> block: {metrics['has_think_rate']:.3f}")
            logger.info(f"  Has <answer> block: {metrics['has_answer_rate']:.3f}")
            logger.info(f"  Valid SMILES rate: {metrics['valid_smiles_rate']:.3f}")
            logger.info(f"  Top-1 Accuracy: {metrics['top1_accuracy']:.3e}")
            logger.info(f"  Top-10 Accuracy: {metrics['top10_accuracy']:.3e}")
            logger.info(f"  Top-1 Accuracy (valid only): {metrics['top1_accuracy_valid']:.3e}")
            logger.info(f"  Top-10 Accuracy (valid only): {metrics['top10_accuracy_valid']:.3e}")
            logger.info(f"  DBE Accuracy: {metrics['dbe_accuracy']:.3f}")
            logger.info(f"  DBE Evaluated Rate: {metrics['dbe_evaluated_rate']:.3f}")
            logger.info(f"  Formula Match Rate: {metrics['formula_match_rate']:.3f}")
            logger.info(f"  Formula Evaluated Rate: {metrics['formula_evaluated_rate']:.3f}")
            logger.info(f"  Top-1 Tanimoto Similarity: {metrics['top1_tanimoto_avg']:.3f}")
            logger.info(f"  Top-10 Tanimoto Similarity: {metrics['top10_tanimoto_avg']:.3f}")
            logger.info(f"  Top-1 MCES Similarity: {metrics['top1_mces_avg']:.3f}")
            logger.info(f"  Top-10 MCES Similarity: {metrics['top10_mces_avg']:.3f}")
            
            # Print binned metrics
            logger.info("  Binned Metrics (Molecular Weight):")
            for bin_name, bin_metrics in metrics['binned_metrics'].items():
                logger.info(f"    Bin {bin_name} (n={bin_metrics['total_samples']}):")
                if bin_metrics['total_samples'] > 0:
                    logger.info(f"      Top-1 Accuracy: {bin_metrics['top1_accuracy']:.3e}")
                    logger.info(f"      Top-10 Accuracy: {bin_metrics['top10_accuracy']:.3e}")
                    logger.info(f"      Top-1 Tanimoto: {bin_metrics['top1_tanimoto_avg']:.3f}")
                    logger.info(f"      Top-10 Tanimoto: {bin_metrics['top10_tanimoto_avg']:.3f}")
                    logger.info(f"      Top-1 MCES: {bin_metrics['top1_mces_avg']:.3f}")
                    logger.info(f"      Top-10 MCES: {bin_metrics['top10_mces_avg']:.3f}")
                else:
                    logger.info("      No samples in this bin.")

        except FileNotFoundError:
            logger.info(f"  File {file_path} not found, skipping...")
        except Exception as e:
            logger.info(f"  Error evaluating {model_name}: {e}")
    
    # Save detailed results
    logger.info("\nSaving detailed results...")
    with open('msllm/benchmark/evaluation_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Saved detailed results to msllm/benchmark/evaluation_results.json")
    logger.info(f"Total results: {len(all_results)}")
    
    # Print comparison table

    table = PrettyTable()
    table.title = "COMPARISON TABLE"
    table.field_names = ["Model","Think %", "Answer %", "SMILES Validity %",
    "Top-1 ACC %", "Top-10 ACC %", "DBE ACC %", "Formula Consistency %",
    "Top-1 Tanimoto", "Top-10 Tanimoto", "Top-1 MCES", "Top-10 MCES"]
    table.align = "l"

    for model_name, metrics in all_results.items():
        table.add_row([model_name,
        metrics['has_think_rate']*100,
        metrics['has_answer_rate']*100,
        metrics['valid_smiles_rate']*100,
        metrics['top1_accuracy']*100,
        metrics['top10_accuracy']*100,
        metrics['dbe_accuracy']*100,
        metrics['formula_match_rate']*100,
        metrics['top1_tanimoto_avg'],
        metrics['top10_tanimoto_avg'],
        metrics['top1_mces_avg'],
        metrics['top10_mces_avg']])
    logger.info(table)
    # Save table to csv
    with open('msllm/benchmark/comparison_table.csv', 'w') as f:
        f.write(table.get_csv_string())

if __name__ == "__main__":
    main() 