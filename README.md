# qwen2-mlx-student-distill

Self-contained research repo for black-box interface-leakage experiments on **GitHub Actions macOS runners** using **Python + MLX + mlx-lm**.

## What this repo runs
Exactly **two experiment families**:

1. **Toy baseline**
   - Tiny MLX student trained on a toy victim.
   - Reports valid top-1 agreement and KL divergence.
2. **Qwen2-0.5B one-step distillation**
   - Victim model: `Qwen/Qwen2-0.5B` loaded via `mlx-lm`.
   - Interfaces: `argmax`, `topk` (`top2`, `top3`, `top5`), and `probs`.
   - Uses one-step next-token distributions and compares student/victim on identical support.

## Dependencies
Only these are used:
- numpy
- pandas
- matplotlib
- pyyaml
- mlx
- mlx-lm
- huggingface_hub

`transformers` is intentionally not added: tokenizer/model loading is done via `mlx-lm` first.

## HF token setup (GitHub Actions)
Add repository secret `HF_TOKEN` (optional but recommended for gated/authenticated downloads):
- Settings → Secrets and variables → Actions → New repository secret
- Name: `HF_TOKEN`

The script prints whether authenticated mode is enabled, **without printing the secret**.

## MLX student training loop
The student model is a small MLX network:
- `nn.Embedding`
- `nn.Linear` + ReLU
- `nn.Linear` to vocabulary logits

Training uses official MLX pattern:
- `mlx.nn.Module`
- `mlx.nn.value_and_grad`
- `mlx.optimizers.Adam`
- optimizer updates with explicit `mx.eval(...)`

## Token-ID conversion safety rule
To avoid MLX token conversion issues:
- Keep token ids as plain Python `list[int]`.
- Convert only at final step with explicit `mx.array(..., dtype=mx.int32)`.
- Never build prompt tensors from bytes, byte arrays, uint8 arrays, memoryviews, or implicit binary buffers.

## Victim interfaces and target construction
For each prompt, victim full next-token logits are converted to dense full-vocabulary softmax.
Interface targets are derived from that **same dense vector**:
- `argmax`: one-hot at top token
- `topk`: keep top-k probs and renormalize
- `probs`: unchanged full softmax

## KL computation and invalid-row rules
Evaluation compares student distribution against victim distribution on the same token support.
- KL is never computed from labels only.
- If shapes mismatch / non-finite / invalid normalization, KL is skipped with reason.
- NaNs are **not** silently replaced.
- `results_summary.csv` includes `kl_valid` and `kl_invalid` counts.

## Required checks and debug mode
### `--sanity-check-llm` (required mode)
Runs before full jobs and:
- tokenizes 3 prompts
- prints Python token types
- prints MLX dtype after conversion
- runs one forward pass
- checks logits shape + vocab dimension
- computes next-token softmax and verifies sum≈1

### `--debug-llm`
Writes per-example JSONL/CSV (first 10 eval prompts per interface/seed), including:
- prompt id/text context and token ids
- Python token types and MLX dtypes
- victim/student top-10 ids + probs
- probability sums, NaN/inf flags
- vocab size + support alignment
- exact KL or skip reason

Also writes:
- `results/qwen_nan_report.csv`
- `results/qwen_vocab_alignment_check.csv`

## Runtime budget (Day 1)
Small CI-friendly budgets:
- Budget sweep: `64, 128, 256, 512` (toy and Qwen)
- Fixed-budget Qwen interface sweep at `256`: `argmax`, `top2`, `top3`, `top5`, `probs`
- Multi-seed summary: small seeds `[0,1,2]`
- Prompt caps are explicit in code (`train <= 20`, eval cap configurable)

## Key artifacts
- `results/qwen_startup_selfcheck.txt`
- `results/results_summary.csv`
- `results/multi_seed_summary.csv`
- `results/agreement_vs_budget.png`
- `results/kl_vs_budget.png`
- `results/qwen_topk_kl.png`
- `results/qwen_nan_report.csv`
- `results/qwen_vocab_alignment_check.csv`
- `results/debug/*.jsonl` and `results/debug/*.csv` (with `--debug-llm`)

## What to inspect first when failures happen
1. Initialization/tokenizer/model issues:
   - `results/qwen_startup_selfcheck.txt`
2. Sanity-check shape/probability failures:
   - CI logs from `--sanity-check-llm`
3. KL validity failures:
   - `results/qwen_nan_report.csv`
   - `results/qwen_vocab_alignment_check.csv`
   - debug JSONL/CSV rows for exact skip reasons

## Run locally
```bash
python -m pip install -r requirements.txt
python run_experiments.py --results-dir results --sanity-check-llm
python run_experiments.py --results-dir results --debug-llm --qwen-eval-cap 10
```
