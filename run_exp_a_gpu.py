"""Run Experiment A on the WEXAC GPU cluster."""
import os, sys
os.chdir('/home/projects/galvardi/yoado')
sys.path.insert(0, '/home/projects/galvardi/yoado/dataset_reconstruction')
sys.path.insert(0, '/home/projects/galvardi/yoado')

import torch
torch.set_default_dtype(torch.float64)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')

from experiments.run_experiment_a import run_single_config

results = run_single_config(
    rank=8, n_per_class=1, seed=42,
    run_baseline=True, device=device, verbose=True,
)

print('\n=== EXPERIMENT A RESULTS ===')
print(f"LoRA (rank=8): {results['lora_metrics']}")
if 'full_ft_metrics' in results:
    print(f"Full FT baseline: {results['full_ft_metrics']}")
print(f"Control comparison: {results['control_metrics']}")

# Save results
import json
save_path = '/home/projects/galvardi/yoado/results/experiment_a_r8_n1.json'
os.makedirs(os.path.dirname(save_path), exist_ok=True)
save_data = {}
for k in ['lora_metrics', 'full_ft_metrics', 'control_metrics', 'lora_train']:
    if k in results:
        save_data[k] = results[k]
with open(save_path, 'w') as f:
    json.dump(save_data, f, indent=2)
print(f"\nResults saved to {save_path}")
