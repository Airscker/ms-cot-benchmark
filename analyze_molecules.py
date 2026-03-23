import json
import re
from collections import defaultdict

# Read the dataset
with open('/Users/airskcer/Library/CloudStorage/OneDrive-Personal/StonyBrook/AI4SCI/MS2Mol/msllm/data/sft_dataset_full.json', 'r') as f:
    data = json.load(f)

# Function to parse molecular formula and count heavy atoms
def count_heavy_atoms(formula):
    # Extract all elements and their counts (excluding H)
    pattern = r'([A-Z][a-z]?)(\d*)'
    matches = re.findall(pattern, formula)
    
    total = 0
    for element, count in matches:
        if element != 'H':
            count = int(count) if count else 1
            total += count
    return total

# Function to calculate SMILES complexity (simple metric: length without stereochemistry markers)
def smiles_complexity(smiles):
    # Remove stereochemistry markers for a simpler complexity measure
    simple_smiles = smiles.replace('@', '').replace('/', '').replace('\\', '')
    return len(simple_smiles)

# Categorize molecules
molecules = []
for entry in data:
    input_text = entry['input']
    smiles = entry['output']
    
    # Extract molecular formula
    formula_match = re.search(r'molecular formula: ([^\s]+)', input_text)
    if formula_match:
        formula = formula_match.group(1).split('(')[0].strip()
        heavy_atoms = count_heavy_atoms(formula)
        complexity = smiles_complexity(smiles)
        
        molecules.append({
            'smiles': smiles,
            'formula': formula,
            'heavy_atoms': heavy_atoms,
            'complexity': complexity
        })

# Sort by heavy atoms
molecules.sort(key=lambda x: x['heavy_atoms'])

# Categorize into small, medium, large
# Small: 5-12 heavy atoms
# Medium: 13-20 heavy atoms
# Large: 21+ heavy atoms

small = [m for m in molecules if 5 <= m['heavy_atoms'] <= 12]
medium = [m for m in molecules if 13 <= m['heavy_atoms'] <= 20]
large = [m for m in molecules if 21 <= m['heavy_atoms'] <= 30]

print("=" * 80)
print("SMALL MOLECULES (5-12 heavy atoms)")
print("=" * 80)
# Show first 20 to pick from
for i, m in enumerate(small[:30]):
    print(f"\n{i+1}. Heavy atoms: {m['heavy_atoms']}, Formula: {m['formula']}")
    print(f"   SMILES: {m['smiles']}")

print("\n" + "=" * 80)
print("MEDIUM MOLECULES (13-20 heavy atoms)")
print("=" * 80)
for i, m in enumerate(medium[:30]):
    print(f"\n{i+1}. Heavy atoms: {m['heavy_atoms']}, Formula: {m['formula']}")
    print(f"   SMILES: {m['smiles']}")

print("\n" + "=" * 80)
print("LARGE MOLECULES (21-30 heavy atoms)")
print("=" * 80)
for i, m in enumerate(large[:30]):
    print(f"\n{i+1}. Heavy atoms: {m['heavy_atoms']}, Formula: {m['formula']}")
    print(f"   SMILES: {m['smiles']}")

print("\n" + "=" * 80)
print(f"Total small molecules: {len(small)}")
print(f"Total medium molecules: {len(medium)}")
print(f"Total large molecules: {len(large)}")
