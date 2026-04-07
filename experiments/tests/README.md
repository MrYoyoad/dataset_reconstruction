# experiments/tests/

pytest test suite for the experiment infrastructure — 6 test files.

---

## Test Index

| File | Tests | Coverage |
|------|-------|----------|
| `test_data_utils.py` | MNIST loading, binary labeling, shape validation, mean subtraction | `data_utils.py` |
| `test_lora_wrapper.py` | LoRALinear correctness, apply_lora, compose_state_dict, param counts | `lora_wrapper.py` |
| `test_train.py` | Training loop convergence, LoRA vs full fine-tuning | `train_lora.py` |
| `test_ntk.py` | NTK coefficient extraction, reconstruction loss, multi-step gradients | `ntk_extraction.py`, `ntk_steps.py` |
| `test_integration.py` | End-to-end Experiment B on tiny config (smoke test) | Full pipeline |
| `test_sprint2c.py` | Sprint 2c-specific ablation tests | `run_sprint2c_sweep.py` |

---

## Running Tests

```bash
# From repo root:
python -m pytest experiments/tests/ -v

# Single file:
python -m pytest experiments/tests/test_ntk.py -v

# With output:
python -m pytest experiments/tests/ -v -s
```

Tests use CPU to avoid GPU dependencies.

---

## Coverage Gaps

- `phase0_vit_inversion.py` — no tests (requires timm + peft + GPU)
- `plotting.py` — no tests (visual output)
- `run_sprint2b_sweep.py` — no dedicated tests (covered by integration)
