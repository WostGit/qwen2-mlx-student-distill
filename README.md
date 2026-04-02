# qwen2-mlx-student-distill

Self-contained research repo for black-box interface-leakage experiments on macOS GitHub Actions using **Python + MLX + mlx-lm**.

## What this runs

Exactly two experiment families:

1. **Toy baseline**
   - Tiny MLX student is trained against a deterministic toy victim.
   - Reports valid **agreement** and **KL divergence**.

2. **Qwen2-0.5B one-step next-token distillation**
   - Victim: `Qwen/Qwen2-0.5B-Instruct` loaded through `mlx-lm`.
   - Student: small MLX model (`mlx.nn.Module`) trained via `mlx.nn.value_and_grad` + `mlx.optimizers.Adam`.
   - Compares victim interfaces: `argmax`, `top2`, `top3`, `top5`, and `probs`.

## Dependency policy

Dependencies used (only):

- numpy
- pandas
- matplotlib
- pyyaml
- mlx
- mlx-lm
- huggingface_hub

No `transformers` dependency is added: tokenizer/model loading is handled by `mlx-lm` first.

## HF token setup

Set GitHub Actions secret:

- `HF_TOKEN` (optional but recommended for authenticated downloads / gated models)

The pipeline prints whether authenticated download is enabled, **never the token value**.

## Token-id conversion rule (MLX bug avoidance)

To avoid token conversion bugs:

- Keep token IDs as plain Python `list[int]` until final conversion.
- Convert explicitly with `mx.array(py_int_list, dtype=mx.int32)`.
- Never construct prompt tensors from bytes, byte arrays, uint8 arrays, memoryviews, or implicit binary buffers.

## Required startup and sanity checks

Startup always writes:

- `results/qwen_startup_selfcheck.txt`

Includes:

- Python version
- macOS version
- MLX version
- mlx-lm version
- model id
- cache dir
- tokenizer class
- vocab size
- one-prompt forward-pass shape

Required sanity mode:

```bash
python src/experiments.py --sanity-check-llm
```

It:
- tokenizes 3 prompts
- prints Python token types
- prints MLX dtype after explicit conversion
- runs forward pass
- verifies logits shape and vocab size
- computes softmax for next token
- verifies probability sums to 1 (tolerance check)
- exits

## Victim interface definitions

For each prompt, victim pipeline is:

1. Get full next-token logits.
2. Convert to dense full-vocab softmax.
3. Derive interface target from the same dense vector:
   - `argmax`: one-hot top token
   - `topk`: renormalized top-k probabilities
   - `probs`: unchanged full softmax

## Student model and training loop

Student is a small MLX model for one-step prediction:

- embedding(last token)
- linear + ReLU
- linear output over a restricted shared support vocabulary

The restricted support is used for CI speed and is explicitly documented with:

- `results/qwen_support_mapping.json`

KL is computed only on shared support with transparent alignment artifacts.

## KL and invalid-row rules

- KL is computed as `KL(victim || student)` on aligned support.
- Never computed from labels only.
- No silent NaN replacement.
- Invalid KL rows are counted and written with reasons.

Key artifacts:

- `results/results_summary.csv` (includes valid/invalid KL counts)
- `results/qwen_nan_report.csv`
- `results/qwen_vocab_alignment_check.csv`

## Debug mode

```bash
python src/experiments.py --debug-llm
```

Writes per-example artifacts for at least first 10 eval prompts per interface and seed:

- `results/qwen_debug_examples.jsonl`
- `results/qwen_debug_examples.csv`

Includes prompt id/text, token IDs, token types, MLX dtypes, top-10 victim/student token IDs+probs, prob sums, NaN/inf flags, vocab size, support alignment, exact KL, and skip reason (if any).

## Budgets / sweeps (Day-1 CI sized)

- Budget sweep: `64, 128, 256, 512` (toy + Qwen)
- Qwen interface sweep at fixed budget includes: `argmax`, `top2`, `top3`, `top5`, `probs`
- Multi-seed summary with mean/std

Generated plots:

- `results/agreement_vs_budget.png`
- `results/kl_vs_budget.png`
- `results/qwen_topk_kl.png`

## Running locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/experiments.py --sanity-check-llm
python src/experiments.py --debug-llm
```

## If something fails, inspect first

1. `results/qwen_startup_selfcheck.txt` (init/tokenizer/forward basics)
2. `results/qwen_vocab_alignment_check.csv` (support mass/alignment)
3. `results/qwen_nan_report.csv` (invalid KL reasons)
4. `results/qwen_debug_examples.jsonl` (fine-grained per-example diagnostics)
