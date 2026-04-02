#!/usr/bin/env python3
"""Black-box interface leakage experiments using MLX and mlx-lm."""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mlx
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import pandas as pd
import yaml
from huggingface_hub import login
from mlx_lm import load

MODEL_ID = "Qwen/Qwen2-0.5B"
DEFAULT_RESULTS = Path("results")
PROMPT_FILE = Path("prompts.yaml")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    mx.random.seed(seed)


def ensure_dirs(results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "debug").mkdir(exist_ok=True)


def softmax_np(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / np.sum(ex)


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> tuple[float | None, str | None]:
    if p.shape != q.shape:
        return None, "shape_mismatch"
    if np.any(~np.isfinite(p)) or np.any(~np.isfinite(q)):
        return None, "non_finite_values"
    if np.any(p < 0) or np.any(q < 0):
        return None, "negative_probability"
    p_sum, q_sum = float(p.sum()), float(q.sum())
    if abs(p_sum - 1.0) > 1e-5 or abs(q_sum - 1.0) > 1e-5:
        return None, f"bad_norm_p={p_sum:.6f}_q={q_sum:.6f}"
    pp = np.clip(p, eps, 1.0)
    qq = np.clip(q, eps, 1.0)
    val = float(np.sum(pp * (np.log(pp) - np.log(qq))))
    if not math.isfinite(val):
        return None, "non_finite_kl"
    return val, None


def load_prompts() -> list[str]:
    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["qwen_prompts"]


def tokenize_prompt(tokenizer: Any, prompt: str) -> list[int]:
    ids = tokenizer.encode(prompt)
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    return [int(x) for x in ids]


def ids_to_mx(ids: list[int]) -> mx.array:
    # Keep Python ints until this explicit conversion.
    return mx.array(ids, dtype=mx.int32)


def get_last_token_distribution(model: Any, token_ids: list[int]) -> np.ndarray:
    x = ids_to_mx(token_ids)[None, :]
    logits = model(x)
    if isinstance(logits, tuple):
        logits = logits[0]
    last_logits = np.array(logits[0, -1, :])
    return softmax_np(last_logits)


def build_interface_target(full_probs: np.ndarray, interface: str) -> np.ndarray:
    out = np.zeros_like(full_probs)
    if interface == "probs":
        return full_probs
    if interface == "argmax":
        out[int(np.argmax(full_probs))] = 1.0
        return out
    if interface.startswith("top"):
        k = int(interface.replace("top", ""))
        idx = np.argpartition(-full_probs, k - 1)[:k]
        vals = full_probs[idx]
        vals = vals / vals.sum()
        out[idx] = vals
        return out
    raise ValueError(f"Unknown interface: {interface}")


class TinyStudent(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.fc1 = nn.Linear(emb_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, vocab_size)

    def __call__(self, tokens: mx.array) -> mx.array:
        x = self.embedding(tokens)
        x = mx.mean(x, axis=1)
        x = nn.relu(self.fc1(x))
        return self.fc2(x)


@dataclass
class EvalResult:
    agreement: float
    kl_mean: float
    kl_valid: int
    kl_invalid: int


def train_student(
    student: TinyStudent,
    train_ids: list[list[int]],
    train_targets: np.ndarray,
    steps: int,
    lr: float,
    batch_size: int,
) -> list[float]:
    opt = optim.Adam(learning_rate=lr)

    def loss_fn(model: TinyStudent, x: mx.array, y: mx.array) -> mx.array:
        logits = model(x)
        log_probs = nn.log_softmax(logits, axis=-1)
        return -mx.mean(mx.sum(y * log_probs, axis=-1))

    loss_and_grad = nn.value_and_grad(student, loss_fn)
    losses: list[float] = []
    n = len(train_ids)
    max_len = max(len(x) for x in train_ids)

    for step in range(steps):
        idx = np.random.choice(n, size=min(batch_size, n), replace=False)
        batch_x = np.zeros((len(idx), max_len), dtype=np.int32)
        for bi, i in enumerate(idx):
            seq = train_ids[i]
            batch_x[bi, : len(seq)] = np.array(seq, dtype=np.int32)
        batch_y = train_targets[idx]

        x_mx = mx.array(batch_x, dtype=mx.int32)
        y_mx = mx.array(batch_y, dtype=mx.float32)

        loss, grads = loss_and_grad(student, x_mx, y_mx)
        opt.update(student, grads)
        mx.eval(student.parameters(), opt.state)

        lf = float(loss.item())
        losses.append(lf)
        if step % max(1, steps // 10) == 0:
            print(f"[TRAIN] step={step}/{steps} loss={lf:.6f}")

    return losses


def evaluate_student(
    student: TinyStudent,
    eval_ids: list[list[int]],
    victim_probs: np.ndarray,
    interface: str,
    seed: int,
    debug_llm: bool,
    results_dir: Path,
) -> EvalResult:
    agreements = []
    kls: list[float] = []
    invalid = 0
    debug_rows = []
    nan_rows = []
    align_rows = []

    max_len = max(len(x) for x in eval_ids)

    for i, ids in enumerate(eval_ids):
        x = np.zeros((1, max_len), dtype=np.int32)
        x[0, : len(ids)] = np.array(ids, dtype=np.int32)
        logits = student(mx.array(x, dtype=mx.int32))
        probs_s = np.array(mx.softmax(logits, axis=-1)[0])
        probs_v = victim_probs[i]

        support_aligned = probs_s.shape == probs_v.shape
        align_rows.append({"idx": i, "aligned": support_aligned, "student_dim": probs_s.shape[0], "victim_dim": probs_v.shape[0]})

        agreements.append(int(np.argmax(probs_s) == np.argmax(probs_v)))
        kl, reason = kl_divergence(probs_v, probs_s)
        if kl is None:
            invalid += 1
            nan_rows.append({"idx": i, "interface": interface, "seed": seed, "reason": reason})
        else:
            kls.append(kl)

        if debug_llm and i < 10:
            top_v = np.argsort(-probs_v)[:10]
            top_s = np.argsort(-probs_s)[:10]
            debug_rows.append(
                {
                    "prompt_id": i,
                    "token_ids": ids,
                    "py_token_types": [type(x).__name__ for x in ids],
                    "mlx_dtype_after_conversion": str(ids_to_mx(ids).dtype),
                    "victim_top10_ids": top_v.tolist(),
                    "victim_top10_probs": probs_v[top_v].tolist(),
                    "student_top10_ids": top_s.tolist(),
                    "student_top10_probs": probs_s[top_s].tolist(),
                    "victim_prob_sum": float(np.sum(probs_v)),
                    "student_prob_sum": float(np.sum(probs_s)),
                    "nan_or_inf": bool(np.any(~np.isfinite(probs_s)) or np.any(~np.isfinite(probs_v))),
                    "vocab_size": int(probs_v.shape[0]),
                    "support_alignment": support_aligned,
                    "exact_kl": kl,
                    "kl_skip_reason": reason,
                    "interface": interface,
                    "seed": seed,
                }
            )

    if debug_rows:
        jpath = results_dir / "debug" / f"qwen_debug_{interface}_seed{seed}.jsonl"
        cpath = results_dir / "debug" / f"qwen_debug_{interface}_seed{seed}.csv"
        with open(jpath, "w", encoding="utf-8") as f:
            for row in debug_rows:
                f.write(json.dumps(row) + "\n")
        pd.DataFrame(debug_rows).to_csv(cpath, index=False)

    if nan_rows:
        pd.DataFrame(nan_rows).to_csv(results_dir / "qwen_nan_report.csv", index=False)
    pd.DataFrame(align_rows).to_csv(results_dir / "qwen_vocab_alignment_check.csv", index=False)

    return EvalResult(
        agreement=float(np.mean(agreements) if agreements else 0.0),
        kl_mean=float(np.mean(kls) if kls else float("nan")),
        kl_valid=len(kls),
        kl_invalid=invalid,
    )


def startup_selfcheck(results_dir: Path, model_id: str, cache_dir: str | None) -> tuple[Any, Any]:
    token = os.getenv("HF_TOKEN")
    print(f"[STARTUP] HF authenticated download enabled: {'yes' if bool(token) else 'no'}")
    if token:
        login(token=token, add_to_git_credential=False)

    model, tokenizer = load(model_id, tokenizer_config={"use_fast": True})
    prompt = "Self-check prompt for one-step next-token inference."
    ids = tokenize_prompt(tokenizer, prompt)
    x = ids_to_mx(ids)[None, :]
    logits = model(x)
    if isinstance(logits, tuple):
        logits = logits[0]
    lines = [
        f"python_version={sys.version}",
        f"macos_version={platform.mac_ver()[0]}",
        f"mlx_version={mlx.__version__}",
        f"mlx_lm_version={__import__('mlx_lm').__version__}",
        f"model_id={model_id}",
        f"cache_dir={cache_dir or os.getenv('HF_HOME', '~/.cache/huggingface')}",
        f"tokenizer_class={tokenizer.__class__.__name__}",
        f"vocab_size={getattr(tokenizer, 'vocab_size', 'unknown')}",
        f"forward_shape={tuple(logits.shape)}",
    ]
    out = results_dir / "qwen_startup_selfcheck.txt"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[STARTUP] wrote {out}")
    return model, tokenizer


def sanity_check_llm(model: Any, tokenizer: Any) -> None:
    prompts = ["Hello world", "What is 2+2?", "A cat sits"]
    print("[SANITY] running tokenizer/type/dtype checks")
    for p in prompts:
        ids = tokenize_prompt(tokenizer, p)
        print(f"[SANITY] prompt={p!r} token_types={[type(x).__name__ for x in ids[:5]]}")
        x = ids_to_mx(ids)
        print(f"[SANITY] mlx_dtype={x.dtype} len={len(ids)}")
        logits = model(x[None, :])
        if isinstance(logits, tuple):
            logits = logits[0]
        assert len(logits.shape) == 3, f"Expected 3D logits, got {logits.shape}"
        next_logits = np.array(logits[0, -1, :])
        probs = softmax_np(next_logits)
        assert probs.shape[0] == logits.shape[-1], "Vocab dimension mismatch"
        ps = probs.sum()
        assert abs(ps - 1.0) < 1e-5, f"softmax sum invalid: {ps}"
        print(f"[SANITY] logits_shape={tuple(logits.shape)} probs_sum={ps:.8f}")
    print("[SANITY] completed successfully; exiting per --sanity-check-llm")


def run_toy_family(results: list[dict[str, Any]], budgets: list[int], seeds: list[int]) -> None:
    print("[TOY] starting toy baseline family")
    vocab = 128
    train_n, eval_n = 128, 32
    for budget in budgets:
        for seed in seeds:
            set_seed(seed)
            toy_ids_train = [np.random.randint(0, vocab, size=np.random.randint(4, 10)).tolist() for _ in range(train_n)]
            toy_ids_eval = [np.random.randint(0, vocab, size=np.random.randint(4, 10)).tolist() for _ in range(eval_n)]

            def victim(ids: list[int]) -> np.ndarray:
                s = sum(ids) % vocab
                logits = np.full(vocab, -4.0)
                logits[s] = 4.0
                logits[(s + 1) % vocab] = 2.0
                return softmax_np(logits)

            v_train = np.stack([victim(x) for x in toy_ids_train])
            v_eval = np.stack([victim(x) for x in toy_ids_eval])
            student = TinyStudent(vocab_size=vocab, emb_dim=32, hidden_dim=64)
            train_student(student, toy_ids_train, v_train, steps=budget, lr=1e-2, batch_size=16)
            ev = evaluate_student(student, toy_ids_eval, v_eval, interface="probs", seed=seed, debug_llm=False, results_dir=DEFAULT_RESULTS)
            results.append({"family": "toy", "interface": "probs", "budget": budget, "seed": seed, "agreement": ev.agreement, "kl_divergence": ev.kl_mean, "kl_valid": ev.kl_valid, "kl_invalid": ev.kl_invalid})
            print(f"[TOY] budget={budget} seed={seed} agreement={ev.agreement:.4f} kl={ev.kl_mean:.6f}")


def run_qwen_family(
    model: Any,
    tokenizer: Any,
    results: list[dict[str, Any]],
    budgets: list[int],
    seeds: list[int],
    debug_llm: bool,
    qwen_eval_cap: int,
) -> None:
    prompts = load_prompts()
    train_prompts = prompts[: min(20, len(prompts))]
    eval_prompts = prompts[min(20, len(prompts)) : min(20 + qwen_eval_cap, len(prompts))]
    print(f"[QWEN] train_prompts={len(train_prompts)} eval_prompts={len(eval_prompts)}")

    train_ids = [tokenize_prompt(tokenizer, p) for p in train_prompts]
    eval_ids = [tokenize_prompt(tokenizer, p) for p in eval_prompts]
    victim_train = np.stack([get_last_token_distribution(model, ids) for ids in train_ids])
    victim_eval = np.stack([get_last_token_distribution(model, ids) for ids in eval_ids])
    vocab = victim_train.shape[1]

    for budget in budgets:
        for seed in seeds:
            for interface in ["argmax", "top2", "top3", "top5", "probs"]:
                set_seed(seed)
                y_train = np.stack([build_interface_target(v, interface) for v in victim_train])
                student = TinyStudent(vocab_size=vocab)
                train_student(student, train_ids, y_train, steps=budget, lr=1e-3, batch_size=8)
                ev = evaluate_student(student, eval_ids, victim_eval, interface=interface, seed=seed, debug_llm=debug_llm, results_dir=DEFAULT_RESULTS)
                results.append({"family": "qwen", "interface": interface, "budget": budget, "seed": seed, "agreement": ev.agreement, "kl_divergence": ev.kl_mean, "kl_valid": ev.kl_valid, "kl_invalid": ev.kl_invalid})
                print(f"[QWEN] interface={interface} budget={budget} seed={seed} agreement={ev.agreement:.4f} kl={ev.kl_mean:.6f} valid={ev.kl_valid} invalid={ev.kl_invalid}")


def generate_plots(df: pd.DataFrame, results_dir: Path) -> None:
    toy_qwen = df[df["family"].isin(["toy", "qwen"])].copy()
    agg = toy_qwen.groupby(["family", "budget"], as_index=False).agg(agreement=("agreement", "mean"), kl=("kl_divergence", "mean"))

    plt.figure(figsize=(8, 5))
    for fam in ["toy", "qwen"]:
        part = agg[agg["family"] == fam]
        plt.plot(part["budget"], part["agreement"], marker="o", label=fam)
    plt.xlabel("Budget (training steps)")
    plt.ylabel("Top-1 agreement")
    plt.title("Agreement vs budget")
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "agreement_vs_budget.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    for fam in ["toy", "qwen"]:
        part = agg[agg["family"] == fam]
        plt.plot(part["budget"], part["kl"], marker="o", label=fam)
    plt.xlabel("Budget (training steps)")
    plt.ylabel("KL divergence")
    plt.title("KL vs budget")
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "kl_vs_budget.png", dpi=150)
    plt.close()

    q = df[(df["family"] == "qwen") & (df["budget"] == 256)]
    qagg = q.groupby("interface", as_index=False).agg(kl=("kl_divergence", "mean"))
    plt.figure(figsize=(8, 5))
    plt.bar(qagg["interface"], qagg["kl"])
    plt.title("Qwen top-k interface vs KL (budget=256)")
    plt.ylabel("KL divergence")
    plt.tight_layout()
    plt.savefig(results_dir / "qwen_topk_kl.png", dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--model-id", type=str, default=MODEL_ID)
    parser.add_argument("--sanity-check-llm", action="store_true")
    parser.add_argument("--debug-llm", action="store_true")
    parser.add_argument("--qwen-eval-cap", type=int, default=10)
    args = parser.parse_args()

    ensure_dirs(args.results_dir)

    budgets = [64, 128, 256, 512]
    seeds = [0, 1, 2]

    model, tokenizer = startup_selfcheck(args.results_dir, args.model_id, os.getenv("HF_HOME"))

    if args.sanity_check_llm:
        sanity_check_llm(model, tokenizer)
        return

    all_results: list[dict[str, Any]] = []
    run_toy_family(all_results, budgets, seeds)
    run_qwen_family(model, tokenizer, all_results, budgets, seeds, args.debug_llm, args.qwen_eval_cap)

    df = pd.DataFrame(all_results)
    df.to_csv(args.results_dir / "results_summary.csv", index=False)

    summary = (
        df.groupby(["family", "interface", "budget"])  # multi-seed mean/std summary
        .agg(agreement_mean=("agreement", "mean"), agreement_std=("agreement", "std"), kl_mean=("kl_divergence", "mean"), kl_std=("kl_divergence", "std"), kl_valid=("kl_valid", "sum"), kl_invalid=("kl_invalid", "sum"))
        .reset_index()
    )
    summary.to_csv(args.results_dir / "multi_seed_summary.csv", index=False)
    generate_plots(df, args.results_dir)
    print("[DONE] Wrote results and plots to", args.results_dir)


if __name__ == "__main__":
    main()
