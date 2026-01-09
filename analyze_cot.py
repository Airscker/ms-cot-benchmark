import json
import re
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import Counter
from typing import List, Dict, Tuple, Optional
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit import RDLogger

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('msllm/benchmark/cot_analysis.log')
    ]
)

logger = logging.getLogger(__name__)

def load_results(file_path: str = 'msllm/benchmark/evaluation_results.json') -> Dict:
    """Load evaluation results from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def extract_blocks(text: str) -> Tuple[bool, bool, str, str]:
    """
    Extract <think> and <answer> blocks from text.
    Returns: (has_think, has_answer, think_content, answer_content)
    """
    has_think = '<think>' in text and '</think>' in text
    has_answer = '<answer>' in text and '</answer>' in text
    
    think_content = ""
    if has_think:
        match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
        if match:
            think_content = match.group(1).strip()
            
    answer_content = ""
    if has_answer:
        match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
        if match:
            answer_content = match.group(1).strip()
    
    return has_think, has_answer, think_content, answer_content

def extract_smiles_from_text(text: str) -> List[str]:
    """Extract SMILES strings from any text."""
    smiles_list = []
    # Use a regex to find potential SMILES strings
    # Refined regex to avoid matching simple text, requiring valid SMILES chars
    # Matches continuous strings of SMILES characters
    raw_candidates = re.findall(r'[A-Za-z0-9@+\-\[\]\(\)=#%\./\\]+', text)
    
    # Common non-SMILES words in this domain to skip immediately
    skip_words = {
        'the', 'and', 'for', 'with', 'from', 'loss', 'peak', 'ion', 'm/z', 
        'intensity', 'potential', 'subformula', 'consistent', 'formula', 
        'observed', 'calculated', 'theoretical', 'corresponding', 'fragment',
        'neutral', 'radical', 'cation', 'anion', 'adduct', 'spectrum',
        'analysis', 'significant', 'likely', 'arises', 'suggests', 'presence',
        'group', 'structure', 'identified', 'proposed', 'contain', 'contains'
    }

    for part in raw_candidates:
        # Cleanup trailing punctuation often matched
        part = part.rstrip('.,;:!)')
        
        # Quick filtering heuristics
        if len(part) < 2: continue 
        if part.lower() in skip_words: continue
        
        # Heuristic: If purely alphabetic and length > 3, it's likely a word, not a SMILES
        # (SMILES usually have numbers, brackets, or are short like "CCO")
        if part.isalpha() and len(part) > 3:
            continue
            
        # Heuristic: Must contain at least one SMILES-specific char if long
        # (digits, brackets, bond markers, etc)
        if len(part) > 3 and not re.search(r'[0-9\[\]\(\)=#@+\-]', part):
            # Exception for some common purely alpha SMILES? e.g. "CCN"
            # But we handled isalpha check above. 
            # This catches things like "MassBank" if not caught by isalpha (e.g. mixed case?)
            continue

        try:
            mol = Chem.MolFromSmiles(part)
            if mol is not None:
                canonical = Chem.MolToSmiles(mol, canonical=True)
                smiles_list.append(canonical)
        except:
            continue
    return list(set(smiles_list)) # Unique SMILES

def calculate_dbe(formula: str) -> float:
    """Calculate DBE from molecular formula."""
    try:
        atoms = {}
        current_atom = ""
        current_count = ""
        
        for char in formula:
            if char.isupper():
                if current_atom:
                    count = int(current_count) if current_count else 1
                    atoms[current_atom] = atoms.get(current_atom, 0) + count
                current_atom = char
                current_count = ""
            elif char.islower():
                current_atom += char
            elif char.isdigit():
                current_count += char
        
        if current_atom:
            count = int(current_count) if current_count else 1
            atoms[current_atom] = atoms.get(current_atom, 0) + count
            
        C = atoms.get('C', 0)
        H = atoms.get('H', 0)
        N = atoms.get('N', 0)
        # Halogens treated as H
        H += atoms.get('F', 0) + atoms.get('Cl', 0) + atoms.get('Br', 0) + atoms.get('I', 0)
        
        return C - H/2 + N/2 + 1
    except:
        return 0.0

def normalize_formula(formula: str) -> str:
    """Normalize molecular formula."""
    try:
        # Simple normalization: remove 1s
        return re.sub(r'([A-Z][a-z]?)1(?![0-9])', r'\1', formula)
    except:
        return formula

def analyze_cot_metrics(results: Dict):
    """Analyze Chain-of-Thought metrics."""
    
    model_metrics = {}
    
    for model_name, model_data in results.items():
        logger.info(f"Analyzing {model_name}...")
        
        detailed_results = model_data.get('detailed_results', [])
        
        # Metrics to track
        cot_lengths = []
        dbe_correctness = [] # (predicted_in_cot == true_dbe)
        formula_correctness = [] # (predicted_in_cot == true_formula)
        contradiction_rate = [] # (smiles_in_cot != smiles_in_answer)
        hallucination_patterns = Counter()
        chemical_terms = Counter()
        
        # Load raw answers if available to get full text
        # Assuming detailed_results contains necessary info or we need to re-load raw files
        # Since detailed_results might not have the full text, we might need to load from original files
        # BUT detailed_results has 'has_think' etc. 
        # Wait, detailed_results in evaluate.py doesn't store the full 'answer' string!
        # We need to reload the original JSON files.
        
        # Get original file path from model name map (hardcoded for now based on evaluate.py)
        file_path_map = {
            'claude-3.5-sonnet': 'msllm/benchmark/claude_answers.json',
            'gpt-4o-mini': 'msllm/benchmark/gpt4o_answers.json',
            'llama_3_8b': 'msllm/benchmark/llama_3_8b_answers.json',
            'llama_70b': 'msllm/benchmark/llama_70b_answers.json'
        }
        
        file_path = file_path_map.get(model_name)
        if not file_path:
            logger.warning(f"No file path found for {model_name}, skipping detailed text analysis.")
            continue
            
        try:
            with open(file_path, 'r') as f:
                answers = json.load(f)
        except FileNotFoundError:
            logger.warning(f"File {file_path} not found.")
            continue

        # Align answers with detailed results (assuming same order and length)
        # detailed_results might be shorter due to min_len in evaluate.py
        limit = len(detailed_results)
        answers = answers[:limit]
        
        for i, (res, raw_answer) in enumerate(zip(detailed_results, answers)):
            has_think, has_answer, think_content, answer_content = extract_blocks(raw_answer)
            
            if not has_think:
                continue
                
            # 1. Length Distribution
            cot_lengths.append(len(think_content.split()))
            
            # 2. Chemical Terms Frequency
            # Simple extraction of chemical-like terms (capitalized words, numbers, brackets)
            terms = re.findall(r'\b[A-Z][a-z]*[0-9]*\b', think_content)
            chemical_terms.update([t for t in terms if len(t) > 1])
            
            # 3. DBE Correctness in CoT
            # Use pre-calculated value from evaluation_results.json if available
            if 'dbe_correct' in res and res['dbe_correct'] is not None:
                dbe_correctness.append(1 if res['dbe_correct'] else 0)
            
            # 4. Formula Correctness in CoT
            # Look for pattern like "Formula: C10H12O2" inside the text
            # This is distinct from 'formula_match' in json which checks the final SMILES structure
            formula_match = re.search(r'Formula:\s*([A-Za-z0-9]+)', think_content)
            if formula_match:
                pred_formula = normalize_formula(formula_match.group(1))
                true_formula = normalize_formula(res.get('true_formula', ''))
                formula_correctness.append(1 if pred_formula == true_formula else 0)
            
            # 5. CoT-to-SMILES Contradiction
            # Simplified: Check if the DBE stated in CoT matches the DBE of the final predicted SMILES
            # This avoids re-parsing all SMILES from text.
            # If model says "DBE is 5" but generates a molecule with DBE 4, that's a contradiction.
            extracted_dbe = res.get('extracted_dbe')
            predicted_smiles = res.get('valid_predicted_smiles', [])
            
            if extracted_dbe is not None and predicted_smiles:
                # Calculate DBE of the first predicted SMILES
                try:
                    mol = Chem.MolFromSmiles(predicted_smiles[0])
                    if mol:
                        calc_formula = rdMolDescriptors.CalcMolFormula(mol)
                        final_dbe = calculate_dbe(calc_formula)
                        # Check contradiction (allow small float diff)
                        if abs(extracted_dbe - final_dbe) > 0.1:
                            contradiction_rate.append(1) # Contradiction
                        else:
                            contradiction_rate.append(0) # Consistent
                except:
                    pass
            
            # 6. Hallucination Patterns
            # Check for common hallucination phrases or structures
            phrases = [
                "search the database", "spectral library", "massbank", "nist", "hmdb",
                "similarity search", "retention time", "collision energy"
            ]
            think_lower = think_content.lower()
            for phrase in phrases:
                if phrase in think_lower:
                    hallucination_patterns[phrase] += 1

        model_metrics[model_name] = {
            'cot_length_mean': np.mean(cot_lengths) if cot_lengths else 0,
            'cot_length_std': np.std(cot_lengths) if cot_lengths else 0,
            'dbe_correctness_cot': np.mean(dbe_correctness) if dbe_correctness else 0,
            'formula_correctness_cot': np.mean(formula_correctness) if formula_correctness else 0,
            'contradiction_rate': np.mean(contradiction_rate) if contradiction_rate else 0,
            'hallucinations': dict(hallucination_patterns.most_common(10)),
            'top_chemical_terms': dict(chemical_terms.most_common(20))
        }
        
    return model_metrics

def plot_cot_metrics(metrics: Dict, save_dir: str = 'msllm/benchmark/cot_plots/'):
    """Plot the analyzed CoT metrics."""
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    models = list(metrics.keys())
    if not models: return

    # 1. Bar plot for Correctness & Contradiction
    plot_vars = ['dbe_correctness_cot', 'formula_correctness_cot', 'contradiction_rate']
    plot_names = ['DBE Correctness (CoT)', 'Formula Correctness (CoT)', 'Contradiction Rate']
    
    data = []
    for m in models:
        for var, name in zip(plot_vars, plot_names):
            data.append({
                'Model': m,
                'Metric': name,
                'Score': metrics[m][var]
            })
            
    df = pd.DataFrame(data)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x='Metric', y='Score', hue='Model', palette='husl')
    plt.title('CoT Reasoning Quality Metrics', fontsize=16)
    plt.ylabel('Rate / Score', fontsize=12)
    plt.ylim(0, 1.0)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'cot_quality_metrics.png'), dpi=300)
    plt.close()
    
    # 2. CoT Length Distribution (Mean + Std)
    plt.figure(figsize=(8, 6))
    means = [metrics[m]['cot_length_mean'] for m in models]
    stds = [metrics[m]['cot_length_std'] for m in models]
    
    plt.bar(models, means, yerr=stds, capsize=5, color=sns.color_palette('husl', len(models)))
    plt.title('Average CoT Length (Words)', fontsize=16)
    plt.ylabel('Word Count', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'cot_length_distribution.png'), dpi=300)
    plt.close()

    # 3. Hallucination Patterns Heatmap (if enough data)
    # Aggregate all hallucination keys
    all_keys = set()
    for m in models:
        all_keys.update(metrics[m]['hallucinations'].keys())
    
    if all_keys:
        halluc_data = []
        for k in all_keys:
            row = {'Phrase': k}
            for m in models:
                row[m] = metrics[m]['hallucinations'].get(k, 0)
            halluc_data.append(row)
        
        df_h = pd.DataFrame(halluc_data).set_index('Phrase')
        # Filter to top N most common overall
        df_h['total'] = df_h.sum(axis=1)
        df_h = df_h.sort_values('total', ascending=False).drop(columns=['total']).head(10)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(df_h, annot=True, fmt='d', cmap='Reds')
        plt.title('Common Hallucination/Heuristic Phrases in CoT', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'hallucination_heatmap.png'), dpi=300)
        plt.close()

    logger.info(f"Plots saved to {save_dir}")

def main():
    logger.info("Loading evaluation results...")
    results = load_results()
    
    logger.info("Analyzing CoT metrics...")
    metrics = analyze_cot_metrics(results)
    
    # Save metrics to JSON
    with open('msllm/benchmark/cot_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved CoT metrics to msllm/benchmark/cot_metrics.json")
    
    logger.info("Plotting results...")
    plot_cot_metrics(metrics)
    
    # Print summary
    print("\n=== CoT Analysis Summary ===")
    for model, data in metrics.items():
        print(f"\nModel: {model}")
        print(f"  Avg CoT Length: {data['cot_length_mean']:.1f} words")
        print(f"  DBE Correctness: {data['dbe_correctness_cot']:.3f}")
        print(f"  Formula Correctness: {data['formula_correctness_cot']:.3f}")
        print(f"  Contradiction Rate: {data['contradiction_rate']:.3f}")
        # print(f"  Top Hallucinations: {list(data['hallucinations'].keys())}")

if __name__ == "__main__":
    main()
