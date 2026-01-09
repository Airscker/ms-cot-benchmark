import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_results(file_path: str = 'msllm/benchmark/evaluation_results.json') -> Dict:
    """Load evaluation results from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def prepare_radar_data(results: Dict) -> Tuple[List[str], List[str], Dict]:
    """
    Prepare data for radar plots.
    Returns: (model_names, metric_names, data_dict)
    """
    model_names = list(results.keys())
    
    # Define metrics for radar plot (normalized to 0-1 scale)
    metrics = {
        'Think Block Rate': 'has_think_rate',
        'Answer Block Rate': 'has_answer_rate', 
        'SMILES Validity': 'valid_smiles_rate',
        'DBE Accuracy': 'dbe_accuracy',
        'Formula Consistency': 'formula_match_rate',
        'Top-1 Tanimoto': 'top1_tanimoto_avg',
        'Top-10 Tanimoto': 'top10_tanimoto_avg',
        'Top-1 MCES': 'top1_mces_avg',
        'Top-10 MCES': 'top10_mces_avg'
    }
    
    metric_names = list(metrics.keys())
    data_dict = {}
    
    for model in model_names:
        data_dict[model] = []
        for metric_key in metrics.values():
            value = results[model][metric_key]
            data_dict[model].append(value)
    
    return model_names, metric_names, data_dict

def create_final_radar_plot(results: Dict, save_path: str = 'msllm/benchmark/radar_plot_final.png'):
    """
    Creates a publication-quality radar plot with a visual floor for enhanced clarity.
    
    This method normalizes all data to a [0, 1] scale but sets a minimum display
    value (a 'floor') to prevent visually jarring drops to the center for low scores.

    Args:
        results (Dict): A dictionary containing model performance data.
        save_path (str): The path to save the generated plot.
    """
    model_names, metric_names, data_dict = prepare_radar_data(results)

    # --- Data Normalization ---
    higher_is_better = [
        "SMILES Validity", "DBE Accuracy", "Formula Consistency",
        "Top-1 Tanimoto", "Top-10 Tanimoto", "Think Block Rate", "Answer Block Rate"
    ]
    df = pd.DataFrame(data_dict, index=metric_names)
    normalized_df = df.copy()
    
    # Invert metric names *before* normalization loop for cleaner logic
    inverted_metrics = [m for m in metric_names if m not in higher_is_better]
    updated_metric_names = [m + ' (Inverted)' if m in inverted_metrics else m for m in metric_names]
    
    for metric in metric_names:
        min_val, max_val = df.loc[metric].min(), df.loc[metric].max()
        if max_val == min_val:
            normalized_df.loc[metric] = 0.5
            continue
        
        if metric in higher_is_better:
            normalized_df.loc[metric] = (df.loc[metric] - min_val) / (max_val - min_val)
        else:
            normalized_df.loc[metric] = (max_val - df.loc[metric]) / (max_val - min_val)

    # --- Plotting ---
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams['font.family'] = 'serif'

    N = len(updated_metric_names)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
    
    # Key change: Define the visual floor
    floor_level = 0.1
    
    # Use a clear, professional color palette
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']

    for i, model in enumerate(model_names):
        values = normalized_df[model].values.flatten().tolist()
        values += values[:1]
        
        # Key change: Apply the floor before plotting
        plot_values = np.maximum(values, floor_level)
        
        ax.plot(angles, plot_values, color=colors[i % len(colors)], linewidth=2.5, label=model, marker='o', markersize=6)
        ax.fill(angles, plot_values, color=colors[i % len(colors)], alpha=0.2)

    # --- Aesthetics and Labels ---
    # Adjust axis limits and labels for the new floor
    ax.set_ylim(floor_level, 1.0)
    ax.set_yticks(np.linspace(floor_level, 1.0, 5))
    ax.set_yticklabels([f"{tick:.1f}" for tick in np.linspace(floor_level, 1.0, 5)], color="grey", size=10)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(updated_metric_names, size=12)
    ax.tick_params(axis='x', pad=20)

    ax.set_title("Normalized Model Performance Comparison", size=16, weight='bold', pad=35)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=len(model_names), frameon=False, fontsize=11)

    ax.grid(color='grey', linestyle='--', linewidth=0.5)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    
    # Add a note about the floor level for transparency
    fig.text(0.99, 0.01, f'*Note: Normalized scores below {floor_level} are plotted at the floor for visual clarity.', 
             ha='right', va='bottom', fontsize=8, color='grey', style='italic')
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    # plt.show()
    print(f"Final radar plot saved to {save_path}")

def create_bar_comparison(results: Dict, save_path: str = 'msllm/benchmark/bar_comparison.png'):
    """Create bar charts comparing specific metrics."""
    model_names = list(results.keys())
    
    # Select key metrics for comparison
    key_metrics = {
        'SMILES Validity Rate': 'valid_smiles_rate',
        'DBE Accuracy': 'dbe_accuracy', 
        'Top-1 Tanimoto': 'top1_tanimoto_avg',
        'Top-10 Tanimoto': 'top10_tanimoto_avg',
        'Top-1 MCES': 'top1_mces_avg',
        'Top-10 MCES': 'top10_mces_avg'
    }
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    # Use sns barplot
    for i, (metric_name, metric_key) in enumerate(key_metrics.items()):
        values = [results[model][metric_key] for model in model_names]
        sns.barplot(x=model_names, y=values, ax=axes[i], palette=colors[:len(model_names)])
        axes[i].set_title(metric_name, fontsize=14, fontweight='bold')
        axes[i].set_ylabel('Score')
        axes[i].tick_params(axis='x', rotation=45)
        # Add value labels on bars
        bars = axes[i].patches
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width()/2., height,
                        f'{value:.3f}', ha='center', va='bottom', fontsize=10)


    # for i, (metric_name, metric_key) in enumerate(key_metrics.items()):
    #     values = [results[model][metric_key] for model in model_names]
        
    #     bars = axes[i].bar(model_names, values, color=colors[:len(model_names)], alpha=0.8)
    #     axes[i].set_title(metric_name, fontsize=14, fontweight='bold')
    #     axes[i].set_ylabel('Score')
    #     axes[i].tick_params(axis='x', rotation=45)
        
    #     # Add value labels on bars
    #     for bar, value in zip(bars, values):
    #         height = bar.get_height()
    #         axes[i].text(bar.get_x() + bar.get_width()/2., height + 0.01,
    #                     f'{value:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    # plt.show()
    print(f"Bar comparison saved to {save_path}")

def create_heatmap(results: Dict, save_path: str = 'msllm/benchmark/heatmap.png'):
    """Create a heatmap of all metrics."""
    model_names = list(results.keys())
    
    # Select metrics for heatmap
    metrics = {
        'Think Rate': 'has_think_rate',
        'Answer Rate': 'has_answer_rate',
        'SMILES Validity': 'valid_smiles_rate',
        'Top-1 Accuracy': 'top1_accuracy',
        'Top-10 Accuracy': 'top10_accuracy',
        'DBE Accuracy': 'dbe_accuracy',
        'Formula Match': 'formula_match_rate',
        'Top-1 Tanimoto': 'top1_tanimoto_avg',
        'Top-10 Tanimoto': 'top10_tanimoto_avg',
        'Top-1 MCES': 'top1_mces_avg',
        'Top-10 MCES': 'top10_mces_avg'
    }

    axis_label_map={
        'claude-3.5-sonnet': 'Claude-3.5-Sonnet',
        'gpt-4o': 'GPT-4o',
        'gpt-4o-mini': 'GPT-4o-mini',
        'llama_3_8b': 'Llama-3-8B',
        'llama_70b': 'Llama-3-70B',
    }
    
    # Create data matrix
    data_matrix = []
    for model in model_names:
        row = [results[model][metric_key] for metric_key in metrics.values()]
        data_matrix.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(data_matrix, index=model_names, columns=list(metrics.keys()))
    # replace index with axis_label_map
    df.index = [axis_label_map[key] for key in df.index]
    # invert 'Top-1 MCES', 'Top-10 MCES' cause lower is better
    df[['Top-1 MCES', 'Top-10 MCES']] = 1 - df[['Top-1 MCES', 'Top-10 MCES']]
    # change names for 'Top-1 MCES', 'Top-10 MCES' to 'Top-1 MCES (Inverted)', 'Top-10 MCES (Inverted)'
    df.rename(columns={'Top-1 MCES': 'Top-1 MCES (Inverted)', 'Top-10 MCES': 'Top-10 MCES (Inverted)'}, inplace=True)
    # sort the order of columns
    df = df[['Think Rate', 'Answer Rate', 'Top-1 MCES (Inverted)', 'Top-10 MCES (Inverted)',
             'SMILES Validity', 'Top-1 Tanimoto', 'Top-10 Tanimoto', 'DBE Accuracy', 'Formula Match',
             'Top-1 Accuracy', 'Top-10 Accuracy']]
    
    # Create heatmap
    plt.figure(figsize=(12, 5))
    sns.heatmap(df, annot=True, cmap='RdYlBu_r', fmt='.3f', 
                cbar_kws={'label': 'Score'}, square=False)
    # rotate x/y axis labels
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.xlabel('Metrics')
    plt.ylabel('Models')
    # plt.title('Model Performance Heatmap', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    # plt.show()
    print(f"Heatmap saved to {save_path}")

def create_individual_radar_plots(results: Dict, save_dir: str = 'msllm/benchmark/individual_radars/'):
    """Create individual radar plots for each model."""
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    model_names, metric_names, data_dict = prepare_radar_data(results)
    
    # Number of variables
    N = len(metric_names)
    
    # Create angles for each metric
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Complete the circle
    
    for model in model_names:
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
        
        values = data_dict[model]
        values += values[:1]  # Complete the circle
        
        ax.plot(angles, values, 
                color='#FF6B6B', linewidth=3, marker='o', markersize=8)
        ax.fill(angles, values, 
                color='#FF6B6B', alpha=0.2)
        
        ax.set_ylim(0, 1)
        ax.set_title(f'{model} Performance Profile', fontsize=16, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3)
        
        # Set the labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_names)
        
        # Add value annotations
        for i, (metric, value) in enumerate(zip(metric_names, data_dict[model])):
            angle = i * 2 * np.pi / len(metric_names)
            x = 1.1 * np.cos(angle - np.pi/2)
            y = 1.1 * np.sin(angle - np.pi/2)
            ax.text(x, y, f'{value:.3f}', ha='center', va='center', 
                   fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/{model.replace("-", "_")}_radar.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Individual radar plots saved to {save_dir}")

def create_binned_metrics_plots(results: Dict, save_dir: str = 'msllm/benchmark/binned_plots/'):
    """Create bar plots for binned metrics comparison."""
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    model_names = list(results.keys())
    
    # Check if binned_metrics exist
    if 'binned_metrics' not in results[model_names[0]]:
        print("No binned metrics found in results. Skipping binned plots.")
        return

    bins = ["0-200", "200-400", "400-600", "600-800", "800+"]
    
    metrics_to_plot = {
        'Top-1 Accuracy': 'top1_accuracy',
        'Top-10 Accuracy': 'top10_accuracy',
        'Top-1 Tanimoto': 'top1_tanimoto_avg',
        'Top-10 Tanimoto': 'top10_tanimoto_avg',
        'Top-1 MCES': 'top1_mces_avg',
        'Top-10 MCES': 'top10_mces_avg'
    }
    
    # Prepare data for plotting
    plot_data = []
    
    model_label_map = {
        'claude-3.5-sonnet': 'Claude-3.5-Sonnet',
        'gpt-4o': 'GPT-4o',
        'gpt-4o-mini': 'GPT-4o-mini',
        'llama_3_8b': 'Llama-3-8B',
        'llama_70b': 'Llama-3-70B',
    }

    for model in model_names:
        binned = results[model].get('binned_metrics', {})
        display_name = model_label_map.get(model, model)
        for bin_name in bins:
            if bin_name in binned:
                bin_data = binned[bin_name]
                # If total_samples is 0, metrics might be 0.0, which is fine
                item = {'Model': display_name, 'Bin': bin_name}
                for metric_name, metric_key in metrics_to_plot.items():
                    item[metric_name] = bin_data.get(metric_key, 0.0)
                plot_data.append(item)
    
    df = pd.DataFrame(plot_data)
    
    # Define colors
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
    
    sns.set_style("whitegrid")
    
    for metric_name in metrics_to_plot.keys():
        plt.figure(figsize=(12, 6))
        
        # Create grouped bar plot
        ax = sns.barplot(data=df, x='Bin', y=metric_name, hue='Model', palette=colors[:len(model_names)])
        
        # plt.title(f'{metric_name} by Molecular Weight Bin', fontsize=16, fontweight='bold', pad=15)
        plt.xlabel('Molecular Weight Range', fontsize=18)
        plt.ylabel('Score', fontsize=18)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.legend(title='Models', loc='best', fontsize=14, title_fontsize=16)
        
        # Add value labels
        # This can be crowded, so maybe skip for now or make it optional/selective
        # for container in ax.containers:
        #     ax.bar_label(container, fmt='%.2f', padding=3, fontsize=8)

        plt.tight_layout()
        save_path = os.path.join(save_dir, f'{metric_name.lower().replace(" ", "_").replace("-", "")}_binned.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved {save_path}")

def main():
    """Main function to generate all visualizations."""
    print("Loading evaluation results...")
    results = load_results()
    
    print("Creating visualizations...")
    
    # Create all plots
    create_final_radar_plot(results)
    create_bar_comparison(results)
    create_heatmap(results)
    create_individual_radar_plots(results)
    create_binned_metrics_plots(results)
    
    print("\nAll visualizations completed!")
    print("Generated files:")
    print("- msllm/benchmark/radar_plot_final.png")
    print("- msllm/benchmark/bar_comparison.png") 
    print("- msllm/benchmark/heatmap.png")
    print("- msllm/benchmark/individual_radars/ (directory with individual radar plots)")
    print("- msllm/benchmark/binned_plots/ (directory with binned metrics plots)")

if __name__ == "__main__":
    main() 