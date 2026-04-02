# qwen2-mlx-student-distill

Research repository for black-box interface-leakage experiments on macOS GitHub Actions with **Python + MLX + mlx-lm**.

## What this repo runs

Exactly two experiment families are implemented:

1. **Toy baseline**
   - Tiny MLX student model (`mlx.nn.Module`) trained against a synthetic toy victim.
   - Reports valid top-1 agreement and KL divergence.

2. **Qwen2-0.5B one-step next-token distillation**
   - Victim model: `Qwen/Qwen2-0.5B-Instruct` loaded with `mlx_lm.load`.
   - Interfaces compared: `argmax`, `top2`, `top3`, `top5`, `probs`.
   - Budget sweep: `64, 128, 256, 512`.

## Dependency policy

Only these dependencies are used:

- numpy
- pandas
- matplotlib
- pyyaml
- mlx
- mlx-lm
- huggingface_hub

`transformers` is intentionally not used because `mlx-lm` already provides loading and tokenizer access needed here.

## Hugging Face token setup (GitHub Actions)

1. Add repository secret named `HF_TOKEN`.
2. Workflow exports `HF_TOKEN` and `HF_HOME=~/.cache/huggingface`.
3. Startup logs print whether authenticated download is enabled (`True/False`) without printing the secret itself.
4. Model cache is reused via `actions/cache`.

## Startup self-check artifact

Run:

```bash
python experiments/run_experiments.py --startup-selfcheck
```

This writes:

- `results/qwen_startup_selfcheck.txt`

Containing:
- Python version
- macOS version
- MLX version
- mlx-lm version
- model id
- cache dir
- tokenizer class
- vocab size
- one-prompt forward-pass shape

## Required LLM sanity check mode

Run:

```bash
python experiments/run_experiments.py --sanity-check-llm
```

It:
- tokenizes 3 short prompts,
- prints **Python token types**,
- converts only at final step with `mx.array(..., dtype=mx.int32)` and prints MLX dtypes,
- runs one forward pass per prompt,
- verifies logits shape + vocabulary dimension,
- computes next-token softmax and checks sum≈1.

## MLX training loop (official pattern)

Training uses:
- `mlx.nn.Module` (`TinyStudent`)
- `mlx.nn.value_and_grad` for gradients
- `mlx.optimizers.Adam` for updates

The loop updates model parameters via MLX-native optimizer state and calls `mx.eval(...)` each step.

## Token-ID conversion safety rule

To avoid token conversion issues:
- token ids stay as plain Python `int` lists through preprocessing,
- no bytes/bytearray/uint8/memoryview/binary buffer prompt tensors are used,
- conversion happens only in one function:

```python
mx.array([token_ids], dtype=mx.int32)
```

## Victim interface definitions

The victim pipeline always starts from **full next-token logits**, then full-vocab dense softmax `p_victim`.

Targets derived from that same dense vector:
- `argmax`: one-hot at top token
- `topk`: keep top-k entries and renormalize
- `probs`: unchanged full softmax

## KL and invalid-row rules

Evaluation compares student and victim distributions on identical support (same vocab shape).

- KL is computed from distributions directly (never labels-only KL).
- If shapes mismatch, non-finite values, invalid probability sums, or negative probs are found, KL is skipped and a reason is recorded.
- NaNs are **not** silently replaced.
- `results_summary.csv` includes `valid_kl_count` and `invalid_kl_count`.

## Debug artifacts (`--debug-llm`)

For first 10 eval prompts per interface/seed:
- `results/debug/*.jsonl`
- `results/debug/*.csv`

Includes:
- prompt id / text
- token ids
- Python token types before MLX conversion
- MLX dtype after conversion
- victim/student top-10 ids/probs
- probability sums
- NaN/inf flags
- vocab size
- support alignment
- exact KL
- skip reason (if any)

Also writes:
- `results/qwen_nan_report.csv`
- `results/qwen_vocab_alignment_check.csv`

## Running all experiments

```bash
python experiments/run_experiments.py --run-all --debug-llm
```

Outputs:
- `results/results_summary.csv`
- `results/multi_seed_summary.csv`
- `results/agreement_vs_budget.png`
- `results/kl_vs_budget.png`
- `results/qwen_topk_kl.png`

## If something fails, inspect these first

1. `results/qwen_startup_selfcheck.txt`
2. sanity-check stdout
3. `results/qwen_vocab_alignment_check.csv`
4. `results/qwen_nan_report.csv`
5. `results/debug/*.jsonl`
