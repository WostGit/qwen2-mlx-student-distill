# qwen2-mlx-student-distill

Research repo for **black-box interface leakage** experiments on macOS GitHub Actions using **Python + MLX + mlx-lm**.

This repository includes exactly two experiment families:

1. **Toy baseline**: tiny MLX student distills from a toy victim, reporting agreement + KL.
2. **Qwen2-0.5B distillation**: one-step next-token distillation from `Qwen/Qwen2-0.5B-Instruct` under multiple victim interfaces (`argmax`, `topk`, `probs`).

## Why this repo exists

The goal is to measure how much information is leaked through different black-box interfaces:

- `argmax`: only the top token.
- `topk`: only top-k tokens with renormalized probabilities.
- `probs`: full next-token probabilities.

The student is trained explicitly with **MLX training primitives** (`mlx.nn.Module`, `mlx.nn.value_and_grad`, and Adam optimizer updates).

---

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Hugging Face token setup

Model download can use authentication.

- Local: export `HF_TOKEN=...`
- GitHub Actions: add repository secret named `HF_TOKEN`

The code prints whether authenticated download is enabled without printing the secret.

---

## Quick start

### 1) Startup self-check (required in CI)

```bash
python -m experiments.run_experiments --startup-selfcheck
```

Writes: `results/qwen_startup_selfcheck.txt`.

The report includes:

- Python version
- macOS version
- MLX version
- mlx-lm version
- model id
- cache dir
- tokenizer class
- vocab size
- one-prompt forward-pass summary

### 2) Sanity check LLM path (required flag)

```bash
python -m experiments.run_experiments --sanity-check-llm
```

This mode:

- Tokenizes 3 short prompts.
- Prints Python token types before MLX conversion.
- Converts only at final step using `mx.array(..., dtype=mx.int32)`.
- Prints MLX dtypes.
- Runs forward pass and verifies logits shape + vocab dimension.
- Computes next-token softmax and verifies probability sum is 1 within tolerance.
- Exits.

### 3) Run experiments

```bash
python -m experiments.run_experiments --run-all
```

### 4) Optional deep debugging

```bash
python -m experiments.run_experiments --run-all --debug-llm
```

Debug mode writes per-example artifacts for first 10 eval prompts per interface/seed.

---

## MLX student training loop

Student training follows the official MLX-style loop:

1. Build an `mlx.nn.Module` student.
2. Define loss function.
3. Create gradient function with `nn.value_and_grad`.
4. Compute gradients each step.
5. Apply updates with Adam optimizer.

No non-MLX workaround is used for the student optimization loop.

---

## Token-ID conversion safety rule

To avoid token-conversion bugs:

- Keep token ids as plain Python `list[int]` as long as possible.
- Only convert at final step with explicit dtype:

```python
mx.array(token_ids, dtype=mx.int32)
```

- Never build prompt tensors from bytes/bytearray/uint8/memoryview/implicit binary buffers.

---

## Interface definitions

Given victim full next-token softmax `p_victim` over full vocabulary:

- **argmax**: one-hot at `argmax(p_victim)`
- **topk**: keep top-k entries, set others to zero, renormalize
- **probs**: unchanged `p_victim`

All interface targets are derived from the **same dense victim softmax**.

---

## KL computation + invalid-row rules

- KL is computed only where student and victim distributions are aligned on the same support.
- For restricted-vocab CI speed mode, explicit mapping artifacts are written and KL is computed only on shared support after transparent renormalization.
- KL is never computed from labels only.
- NaN/Inf rows are never silently zeroed.
- Each skipped KL row receives a human-readable reason.
- `results_summary.csv` reports `valid_kl_count` and `invalid_kl_count`.

---

## Outputs

Key outputs in `results/`:

- `results_summary.csv`
- `multi_seed_summary.csv`
- `agreement_vs_budget.png`
- `kl_vs_budget.png`
- `qwen_topk_kl.png`
- `qwen_startup_selfcheck.txt`
- `qwen_nan_report.csv`
- `qwen_vocab_alignment_check.csv`
- `debug_qwen_*.jsonl` and `debug_qwen_*.csv` (when `--debug-llm`)

### If something fails, inspect first

1. `results/qwen_startup_selfcheck.txt`
2. `results/qwen_vocab_alignment_check.csv`
3. `results/qwen_nan_report.csv`
4. Relevant `debug_qwen_*.jsonl`

---

## CI

GitHub Actions is macOS-only and does:

1. Install dependencies
2. Run startup self-check
3. Run `--sanity-check-llm`
4. Run full experiment set
5. Upload artifacts even on failure

It also caches pip wheels and Hugging Face model cache.
