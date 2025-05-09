import pickle
import matplotlib.pyplot as plt
import sys
import os
from typing import Dict, List

# Append the project root to sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(project_root, '..')
sys.path.append(project_root)
from definitions.constants import SENSITIVITY_DIR, SENS_PARAMETER_CONFIG

def load_pickle(file_path: str) -> Dict:
    with open(file_path, 'rb') as file:
        return pickle.load(file)

def plot_group(parameter: str, metrics: Dict[str, tuple], output_path: str) -> None:
    fig, axs = plt.subplots(len(metrics), 2, figsize=(20, 6 * len(metrics)))
    if len(metrics) == 1:
        axs = [axs]

    for ax_row, (metric_key, (title, y_label, color)) in zip(axs, metrics.items()):
        full_metric_name_tillGS = f"{metric_key}_{parameter}_tillGS"
        full_metric_name_full = f"{metric_key}_{parameter}_full"
        
        data_tillGS = load_pickle(os.path.join(SENSITIVITY_DIR, f"{full_metric_name_tillGS}.pkl"))
        data_full = load_pickle(os.path.join(SENSITIVITY_DIR, f"{full_metric_name_full}.pkl"))
        
        # Special handling for list-type parameters
        if parameter in ['mom_window', 'half_life']:
            # Convert string keys like '[252]' to integers and sort
            processed_tillGS = []
            for k, v in data_tillGS.items():
                try:
                    # Remove brackets and convert to integer
                    num_val = int(k.strip('[]'))
                    processed_tillGS.append((num_val, k, v))
                except:
                    processed_tillGS.append((float('inf'), k, v))
            
            processed_full = []
            for k, v in data_full.items():
                try:
                    # Remove brackets and convert to integer
                    num_val = int(k.strip('[]'))
                    processed_full.append((num_val, k, v))
                except:
                    processed_full.append((float('inf'), k, v))
            
            # Sort by numeric value while keeping original string key
            processed_tillGS.sort(key=lambda x: x[0])
            processed_full.sort(key=lambda x: x[0])
            
            sorted_keys_tillGS = [str(p[0]) for p in processed_tillGS]  # Use numeric value as label
            sorted_values_tillGS = [p[2] for p in processed_tillGS]
            sorted_keys_full = [str(p[0]) for p in processed_full]  # Use numeric value as label
            sorted_values_full = [p[2] for p in processed_full]
        else:
            # Standard numeric sorting for other parameters
            try:
                keys_tillGS = [float(k) if '.' in k else int(k) for k in data_tillGS.keys()]
            except:
                keys_tillGS = list(data_tillGS.keys())
            
            try:
                keys_full = [float(k) if '.' in k else int(k) for k in data_full.keys()]
            except:
                keys_full = list(data_full.keys())
            
            values_tillGS = list(data_tillGS.values())
            values_full = list(data_full.values())
            
            sorted_pairs_tillGS = sorted(zip(keys_tillGS, values_tillGS), key=lambda x: x[0])
            sorted_pairs_full = sorted(zip(keys_full, values_full), key=lambda x: x[0])
            
            sorted_keys_tillGS, sorted_values_tillGS = zip(*sorted_pairs_tillGS) if sorted_pairs_tillGS else ([], [])
            sorted_keys_full, sorted_values_full = zip(*sorted_pairs_full) if sorted_pairs_full else ([], [])

        ax_row[0].plot(sorted_keys_tillGS, sorted_values_tillGS, marker='o', color=color, label='Till GS')
        ax_row[0].set_title(title + " (Till GS)")
        ax_row[0].set_xlabel(parameter.replace('_', ' ').title())
        ax_row[0].set_ylabel(y_label)
        
        ax_row[1].plot(sorted_keys_full, sorted_values_full, marker='o', color=color, label='Full')
        ax_row[1].set_title(title + " (Full)")
        ax_row[1].set_xlabel(parameter.replace('_', ' ').title())
        ax_row[1].set_ylabel(y_label)
        
        # Improve x-axis labels for list-type parameters
        if parameter in ['mom_window', 'half_life']:
            ax_row[0].set_xticks(sorted_keys_tillGS)
            ax_row[0].set_xticklabels([str(p[0]) for p in processed_tillGS])  # Show numeric values
            ax_row[1].set_xticks(sorted_keys_full)
            ax_row[1].set_xticklabels([str(p[0]) for p in processed_full])  # Show numeric values

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Plot group saved to {output_path}")

# Generate plots
for parameter, metrics in SENS_PARAMETER_CONFIG.items():
    output_path = f"{parameter}_sensitivity_analysis.pdf"
    plot_group(parameter, metrics, output_path)