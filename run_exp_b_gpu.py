"""Run Experiment B on the WEXAC GPU cluster."""
import os, sys
os.chdir('/home/projects/galvardi/yoado')
sys.path.insert(0, '/home/projects/galvardi/yoado/dataset_reconstruction')
sys.path.insert(0, '/home/projects/galvardi/yoado')

import torch
torch.set_default_dtype(torch.float64)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')

from experiments.run_experiment_b import run_single_config

# T=1, full model + LoRA rank=8
results = run_single_config(
    n_steps=1, rank=8, n_per_class=1, seed=42,
    extraction_epochs=20000,
    device=device, verbose=True,
)

print('\n=== EXPERIMENT B RESULTS (T=1) ===')
if 'full_metrics' in results:
    print(f"Full model (T=1): {results['full_metrics']}")
if 'lora_metrics' in results:
    print(f"LoRA rank=8 (T=1): {results['lora_metrics']}")
if 'full_diagnostics' in results:
    print(f"NTK diagnostics: {results['full_diagnostics']}")
if 'control_metrics' in results:
    print(f"Control: {results['control_metrics']}")

# Save results
import json
save_path = '/home/projects/galvardi/yoado/results/experiment_b_t1.json'
os.makedirs(os.path.dirname(save_path), exist_ok=True)
save_data = {}
for k in ['full_metrics', 'lora_metrics', 'full_diagnostics', 'control_metrics']:
    if k in results:
        save_data[k] = results[k]
with open(save_path, 'w') as f:
    json.dump(save_data, f, indent=2)
print(f"\nResults saved to {save_path}")
