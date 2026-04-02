import argparse
import csv
import json
import math
import os
import platform
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import mlx
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import pandas as pd
from huggingface_hub import login
from mlx_lm import load as mlx_lm_load


RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


TOY_PROMPTS = [
    "red fox",
    "blue moon",
    "small cat",
    "green leaf",
    "quiet river",
    "bright star",
    "empty road",
    "warm tea",
]

QWEN_PROMPTS_TRAIN = [
    "The capital of France is",
    "2 + 2 equals",
    "A healthy breakfast often includes",
    "The opposite of hot is",
    "In Python, a list can",
    "The largest planet in our solar system is",
    "The color of a ripe banana is",
    "Water freezes at",
    "A triangle has",
    "The past tense of run is",
]

QWEN_PROMPTS_EVAL = [
    "The fastest land animal is",
    "Earth revolves around",
    "A synonym for quick is",
    "The chemical symbol for water is",
    "Winter is usually",
    "Birds can usually",
    "The square root of 9 is",
    "An antonym of happy is",
    "Trees absorb",
    "The ocean is made of",
    "The first day of the week is",
    "Cooking pasta requires boiling",
]

BUDGETS = [64, 128, 256, 512]
SEEDS = [0, 1, 2]
TOPK_SWEEP = [1, 2, 3, 5]


@dataclass
class KLResult:
    value: Optional[float]
    valid: bool
    reason: str


class TinyStudent(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = 32, hidden: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.fc1 = nn.Linear(emb_dim, hidden)
        self.fc2 = nn.Linear(hidden, vocab_size)

    def __call__(self, token_ids: mx.array) -> mx.array:
        # token_ids shape: [batch, seq], using only last token for 1-step prediction
        last_tokens = token_ids[:, -1]
        emb = self.embedding(last_tokens)
        h = nn.relu(self.fc1(emb))
        return self.fc2(h)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits.astype(np.float64)
    logits = logits - np.max(logits)
    exps = np.exp(logits)
    denom = np.sum(exps)
    return exps / denom


def interface_from_dense(dense_probs: np.ndarray, interface: str) -> np.ndarray:
    if interface == "probs":
        return dense_probs.copy()
    if interface == "argmax":
        out = np.zeros_like(dense_probs)
        out[int(np.argmax(dense_probs))] = 1.0
        return out
    if interface.startswith("top"):
        k = int(interface.replace("top", ""))
        idx = np.argpartition(-dense_probs, k - 1)[:k]
        out = np.zeros_like(dense_probs)
        out[idx] = dense_probs[idx]
        s = out.sum()
        if s <= 0:
            raise ValueError(f"top-k interface produced zero mass for {interface}")
        return out / s
    raise ValueError(f"Unknown interface: {interface}")


def kl_divergence(victim_p: np.ndarray, student_q: np.ndarray, eps: float = 1e-12) -> KLResult:
    if victim_p.shape != student_q.shape:
        return KLResult(None, False, "shape_mismatch")
    if not np.all(np.isfinite(victim_p)) or not np.all(np.isfinite(student_q)):
        return KLResult(None, False, "nan_or_inf")
    s1 = victim_p.sum()
    s2 = student_q.sum()
    if abs(s1 - 1.0) > 1e-4 or abs(s2 - 1.0) > 1e-4:
        return KLResult(None, False, "probability_sum_invalid")

    support = victim_p > eps
    if np.any(student_q[support] <= 0):
        return KLResult(None, False, "zero_student_prob_on_victim_support")

    val = float(np.sum(victim_p[support] * (np.log(victim_p[support]) - np.log(student_q[support]))))
    if not math.isfinite(val):
        return KLResult(None, False, "kl_not_finite")
    return KLResult(val, True, "ok")


def top1_agreement(victim_p: np.ndarray, student_q: np.ndarray) -> float:
    return float(int(np.argmax(victim_p) == np.argmax(student_q)))


def ensure_hf_login() -> bool:
    token = os.environ.get("HF_TOKEN", "").strip()
    enabled = bool(token)
    print(f"[startup] HF authenticated download enabled: {enabled}")
    if enabled:
        login(token=token, add_to_git_credential=False)
    return enabled


def to_mx_int32(tokens: Sequence[int]) -> mx.array:
    py_ints = [int(x) for x in tokens]
    return mx.array(py_ints, dtype=mx.int32)


def forward_logits(model: Any, input_ids: mx.array) -> mx.array:
    out = model(input_ids[None, :])
    if isinstance(out, tuple):
        out = out[0]
    if isinstance(out, dict):
        if "logits" in out:
            out = out["logits"]
        else:
            raise ValueError("Model output dict has no logits key")
    if hasattr(out, "logits"):
        out = out.logits
    return out


def write_startup_selfcheck(model: Any, tokenizer: Any, model_id: str, cache_dir: str) -> None:
    prompt = "Hello from MLX"
    toks = tokenizer.encode(prompt)
    ids = to_mx_int32(toks)
    logits = forward_logits(model, ids)
    logits_np = np.array(mx.array(logits))

    report = {
        "python_version": sys.version,
        "macos_version": platform.platform(),
        "mlx_version": getattr(mlx, "__version__", "unknown"),
        "mlx_lm_version": __import__("mlx_lm").__version__ if hasattr(__import__("mlx_lm"), "__version__") else "unknown",
        "model_id": model_id,
        "cache_dir": cache_dir,
        "tokenizer_class": tokenizer.__class__.__name__,
        "vocab_size": int(getattr(tokenizer, "vocab_size", -1)),
        "one_prompt_forward_shape": list(logits_np.shape),
    }

    out_path = RESULTS_DIR / "qwen_startup_selfcheck.txt"
    with out_path.open("w", encoding="utf-8") as f:
        for k, v in report.items():
            f.write(f"{k}: {v}\n")
    print(f"[startup] wrote startup self-check: {out_path}")


def sanity_check_llm(model: Any, tokenizer: Any, model_id: str) -> None:
    print(f"[sanity] Running sanity check for {model_id}")
    prompts = ["cat", "blue sky", "simple test"]
    for p in prompts:
        tokens = tokenizer.encode(p)
        py_types = [type(x).__name__ for x in tokens]
        print(f"[sanity] prompt={p!r} tokens={tokens} py_types={py_types}")

        ids = to_mx_int32(tokens)
        print(f"[sanity] mlx dtype after conversion: {ids.dtype}")

        logits = forward_logits(model, ids)
        logits_np = np.array(mx.array(logits))
        if logits_np.ndim != 3:
            raise RuntimeError(f"Unexpected logits rank: {logits_np.shape}")

        vocab_size = int(getattr(tokenizer, "vocab_size", logits_np.shape[-1]))
        if logits_np.shape[-1] != vocab_size:
            raise RuntimeError(f"Vocab mismatch: logits={logits_np.shape[-1]} tokenizer={vocab_size}")

        next_logits = logits_np[0, -1, :]
        probs = stable_softmax(next_logits)
        psum = float(probs.sum())
        print(f"[sanity] logits_shape={logits_np.shape} next_prob_sum={psum:.8f}")
        if abs(psum - 1.0) > 1e-6:
            raise RuntimeError(f"Softmax sum invalid: {psum}")

    print("[sanity] SUCCESS")


def toy_victim_distribution(prompt: str, vocab_size: int = 32) -> np.ndarray:
    # Deterministic toy victim: hash prompt -> smooth categorical
    rng = np.random.default_rng(abs(hash(prompt)) % (2**32))
    logits = rng.normal(size=vocab_size)
    return stable_softmax(logits)


def train_student_mlx(
    model: TinyStudent,
    optimizer: optim.Adam,
    x_tokens: List[List[int]],
    y_targets: List[np.ndarray],
    steps: int,
) -> None:
    y_stack = np.stack(y_targets, axis=0).astype(np.float32)

    def loss_fn(m: TinyStudent, xb: mx.array, yb: mx.array) -> mx.array:
        logits = m(xb)
        log_probs = nn.log_softmax(logits, axis=-1)
        return -(yb * log_probs).sum(axis=-1).mean()

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    batch_size = min(8, len(x_tokens))
    for step in range(steps):
        idx = np.random.choice(len(x_tokens), size=batch_size, replace=True)
        xb_py = [x_tokens[i] for i in idx]
        max_len = max(len(x) for x in xb_py)
        xb_pad = [x + [x[-1]] * (max_len - len(x)) for x in xb_py]

        xb = mx.array(np.array(xb_pad, dtype=np.int32), dtype=mx.int32)
        yb = mx.array(y_stack[idx], dtype=mx.float32)

        loss, grads = loss_and_grad(model, xb, yb)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        if step % max(1, steps // 5) == 0:
            print(f"[train] step={step}/{steps} loss={float(loss):.6f}")


def predict_student_probs(model: TinyStudent, token_ids: List[int]) -> np.ndarray:
    x = mx.array(np.array([token_ids], dtype=np.int32), dtype=mx.int32)
    logits = model(x)
    probs = nn.softmax(logits, axis=-1)
    out = np.array(mx.array(probs))[0]
    return out / out.sum()


def build_support_from_victim(victim_dense: Dict[str, np.ndarray], max_support: int = 512) -> List[int]:
    counts: Dict[int, float] = {}
    for vec in victim_dense.values():
        top = np.argpartition(-vec, min(63, len(vec) - 1))[: min(64, len(vec))]
        for i in top:
            counts[int(i)] = counts.get(int(i), 0.0) + float(vec[int(i)])
    sorted_ids = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    return [tid for tid, _ in sorted_ids[:max_support]]


def project_to_support(dense: np.ndarray, support: List[int]) -> np.ndarray:
    sub = dense[np.array(support, dtype=np.int64)]
    s = sub.sum()
    if s <= 0:
        return np.zeros_like(sub)
    return sub / s


def collect_qwen_victim_distributions(model: Any, tokenizer: Any, prompts: List[str]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for i, p in enumerate(prompts):
        toks = tokenizer.encode(p)
        ids = to_mx_int32(toks)
        logits = forward_logits(model, ids)
        logits_np = np.array(mx.array(logits))
        next_logits = logits_np[0, -1, :]
        dense = stable_softmax(next_logits)
        out[str(i)] = {
            "prompt": p,
            "token_ids": [int(x) for x in toks],
            "python_token_types": [type(x).__name__ for x in toks],
            "mlx_dtype": str(ids.dtype),
            "dense_probs": dense,
        }
    return out


def run_toy_family(results_rows: List[Dict[str, Any]]) -> None:
    print("[toy] Running toy baseline family")
    vocab_size = 32
    train_x = [[(abs(hash(p)) + i) % vocab_size for i in range(3)] for p in TOY_PROMPTS]
    victim_dense = {p: toy_victim_distribution(p, vocab_size=vocab_size) for p in TOY_PROMPTS}

    for budget in BUDGETS:
        for seed in SEEDS:
            set_seed(seed)
            model = TinyStudent(vocab_size=vocab_size, emb_dim=16, hidden=32)
            optimizer = optim.Adam(learning_rate=1e-2)
            y = [victim_dense[p] for p in TOY_PROMPTS]
            train_student_mlx(model, optimizer, train_x, y, steps=budget)

            agreements, kls = [], []
            valid_kl, invalid_kl = 0, 0
            for p, x in zip(TOY_PROMPTS, train_x):
                v = victim_dense[p]
                s = predict_student_probs(model, x)
                agreements.append(top1_agreement(v, s))
                kl = kl_divergence(v, s)
                if kl.valid:
                    valid_kl += 1
                    kls.append(kl.value)
                else:
                    invalid_kl += 1

            results_rows.append({
                "family": "toy",
                "interface": "probs",
                "budget": budget,
                "seed": seed,
                "agreement": float(np.mean(agreements)),
                "kl_divergence": float(np.mean(kls)) if kls else np.nan,
                "valid_kl_count": valid_kl,
                "invalid_kl_count": invalid_kl,
            })


def run_qwen_family(
    model: Any,
    tokenizer: Any,
    debug_llm: bool,
    results_rows: List[Dict[str, Any]],
) -> None:
    print("[qwen] Running Qwen2-0.5B one-step distillation family")

    train_data = collect_qwen_victim_distributions(model, tokenizer, QWEN_PROMPTS_TRAIN)
    eval_data = collect_qwen_victim_distributions(model, tokenizer, QWEN_PROMPTS_EVAL)

    support = build_support_from_victim({k: v["dense_probs"] for k, v in train_data.items()}, max_support=512)
    support_map = {int(tok): i for i, tok in enumerate(support)}
    with (RESULTS_DIR / "qwen_support_mapping.json").open("w", encoding="utf-8") as f:
        json.dump({"support_token_ids": support, "size": len(support)}, f, indent=2)

    with (RESULTS_DIR / "qwen_vocab_alignment_check.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["prompt_id", "prompt", "support_size", "support_mass", "aligned"])
        w.writeheader()
        for pid, row in eval_data.items():
            mass = float(row["dense_probs"][np.array(support, dtype=np.int64)].sum())
            w.writerow({
                "prompt_id": pid,
                "prompt": row["prompt"],
                "support_size": len(support),
                "support_mass": mass,
                "aligned": mass > 0,
            })

    interfaces = ["argmax", "top2", "top3", "top5", "probs"]

    nan_rows = []
    debug_rows = []

    for interface in interfaces:
        for budget in BUDGETS:
            for seed in SEEDS:
                set_seed(seed)
                model_s = TinyStudent(vocab_size=len(support), emb_dim=32, hidden=64)
                optimizer = optim.Adam(learning_rate=3e-3)

                x_tokens, y_targets = [], []
                for row in train_data.values():
                    x_tokens.append(row["token_ids"])
                    dense = row["dense_probs"]
                    iface_dense = interface_from_dense(dense, interface)
                    y_targets.append(project_to_support(iface_dense, support))

                train_student_mlx(model_s, optimizer, x_tokens, y_targets, steps=budget)

                agreements, kls = [], []
                valid_kl, invalid_kl = 0, 0

                for idx, (pid, row) in enumerate(eval_data.items()):
                    victim_dense = row["dense_probs"]
                    victim_iface = interface_from_dense(victim_dense, interface)
                    victim_proj = project_to_support(victim_iface, support)
                    student_proj = predict_student_probs(model_s, row["token_ids"])

                    agree = top1_agreement(victim_proj, student_proj)
                    agreements.append(agree)

                    kl = kl_divergence(victim_proj, student_proj)
                    if kl.valid:
                        valid_kl += 1
                        kls.append(kl.value)
                    else:
                        invalid_kl += 1
                        nan_rows.append({
                            "family": "qwen",
                            "interface": interface,
                            "budget": budget,
                            "seed": seed,
                            "prompt_id": pid,
                            "reason": kl.reason,
                        })

                    if debug_llm and idx < 10:
                        v_top = np.argpartition(-victim_proj, min(9, len(victim_proj)-1))[: min(10, len(victim_proj))]
                        s_top = np.argpartition(-student_proj, min(9, len(student_proj)-1))[: min(10, len(student_proj))]
                        debug_rows.append({
                            "prompt_id": pid,
                            "prompt_text": row["prompt"],
                            "token_ids": row["token_ids"],
                            "python_token_types": row["python_token_types"],
                            "mlx_dtypes": row["mlx_dtype"],
                            "victim_top10_token_ids": [int(support[i]) for i in v_top.tolist()],
                            "victim_top10_probs": [float(victim_proj[i]) for i in v_top.tolist()],
                            "student_top10_token_ids": [int(support[i]) for i in s_top.tolist()],
                            "student_top10_probs": [float(student_proj[i]) for i in s_top.tolist()],
                            "victim_prob_sum": float(victim_proj.sum()),
                            "student_prob_sum": float(student_proj.sum()),
                            "victim_has_nan_inf": bool(np.any(~np.isfinite(victim_proj))),
                            "student_has_nan_inf": bool(np.any(~np.isfinite(student_proj))),
                            "vocab_size": len(support),
                            "support_alignment": True,
                            "exact_kl": kl.value,
                            "kl_skip_reason": "" if kl.valid else kl.reason,
                            "interface": interface,
                            "seed": seed,
                            "budget": budget,
                        })

                results_rows.append({
                    "family": "qwen",
                    "interface": interface,
                    "budget": budget,
                    "seed": seed,
                    "agreement": float(np.mean(agreements)),
                    "kl_divergence": float(np.mean(kls)) if kls else np.nan,
                    "valid_kl_count": valid_kl,
                    "invalid_kl_count": invalid_kl,
                })

    pd.DataFrame(nan_rows).to_csv(RESULTS_DIR / "qwen_nan_report.csv", index=False)

    if debug_llm:
        with (RESULTS_DIR / "qwen_debug_examples.jsonl").open("w", encoding="utf-8") as f:
            for row in debug_rows:
                f.write(json.dumps(row) + "\n")
        pd.DataFrame(debug_rows).to_csv(RESULTS_DIR / "qwen_debug_examples.csv", index=False)


def make_plots(df: pd.DataFrame) -> None:
    toy_qwen = df[df["family"].isin(["toy", "qwen"])]

    fig, ax = plt.subplots(figsize=(8, 5))
    plot_df = toy_qwen.groupby(["family", "budget"], as_index=False)["agreement"].mean()
    for fam in plot_df["family"].unique():
        s = plot_df[plot_df["family"] == fam]
        ax.plot(s["budget"], s["agreement"], marker="o", label=fam)
    ax.set_title("Agreement vs Budget")
    ax.set_xlabel("Training steps budget")
    ax.set_ylabel("Top-1 agreement")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "agreement_vs_budget.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    plot_df = toy_qwen.groupby(["family", "budget"], as_index=False)["kl_divergence"].mean(numeric_only=True)
    for fam in plot_df["family"].unique():
        s = plot_df[plot_df["family"] == fam]
        ax.plot(s["budget"], s["kl_divergence"], marker="o", label=fam)
    ax.set_title("KL vs Budget")
    ax.set_xlabel("Training steps budget")
    ax.set_ylabel("KL divergence")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "kl_vs_budget.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    q = df[(df["family"] == "qwen") & (df["budget"] == 256)]
    qg = q.groupby("interface", as_index=False)["kl_divergence"].mean(numeric_only=True)
    ax.bar(qg["interface"], qg["kl_divergence"])
    ax.set_title("Qwen top-k KL at budget=256")
    ax.set_xlabel("Victim interface")
    ax.set_ylabel("KL divergence")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "qwen_topk_kl.png")


def write_seed_summary(df: pd.DataFrame) -> None:
    grouped = df.groupby(["family", "interface", "budget"], as_index=False).agg(
        agreement_mean=("agreement", "mean"),
        agreement_std=("agreement", "std"),
        kl_mean=("kl_divergence", "mean"),
        kl_std=("kl_divergence", "std"),
        valid_kl_count=("valid_kl_count", "sum"),
        invalid_kl_count=("invalid_kl_count", "sum"),
    )
    grouped.to_csv(RESULTS_DIR / "multi_seed_summary.csv", index=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Black-box interface leakage experiments with MLX")
    p.add_argument("--model-id", default="Qwen/Qwen2-0.5B-Instruct")
    p.add_argument("--cache-dir", default=os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")))
    p.add_argument("--sanity-check-llm", action="store_true", required=False)
    p.add_argument("--debug-llm", action="store_true")
    p.add_argument("--skip-qwen", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(exist_ok=True)

    ensure_hf_login()
    print(f"[startup] cache dir: {args.cache_dir}")

    model, tokenizer = mlx_lm_load(args.model_id)
    write_startup_selfcheck(model, tokenizer, args.model_id, args.cache_dir)

    if args.sanity_check_llm:
        sanity_check_llm(model, tokenizer, args.model_id)
        return

    results_rows: List[Dict[str, Any]] = []
    run_toy_family(results_rows)

    if not args.skip_qwen:
        run_qwen_family(model, tokenizer, args.debug_llm, results_rows)

    df = pd.DataFrame(results_rows)
    df.to_csv(RESULTS_DIR / "results_summary.csv", index=False)
    write_seed_summary(df)
    make_plots(df)

    print("[done] wrote results in ./results")


if __name__ == "__main__":
    main()
