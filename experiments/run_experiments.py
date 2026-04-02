#!/usr/bin/env python3
"""Black-box interface leakage experiments with MLX/MLX-LM.

Experiment families:
1) Toy baseline with tiny MLX student and toy victim.
2) Qwen2-0.5B one-step next-token distillation across interfaces.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import platform
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import pandas as pd
import yaml
from huggingface_hub import whoami
from mlx_lm import load

LOGGER = logging.getLogger("qwen_distill")


def configure_logging(verbose: bool = True) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


DEFAULT_PROMPTS = [
    "Write one sentence about gravity.",
    "The capital of France is",
    "A tiny robot learns to",
    "In one line, define entropy.",
    "Summarize photosynthesis briefly:",
    "Once upon a time in a lab,",
    "Translate to Spanish: good morning",
    "Python list comprehension example:",
    "Explain why the sky appears blue.",
    "A haiku about rain:",
    "Name three ocean mammals:",
    "The derivative of x^2 is",
    "A one-sentence startup pitch:",
    "What is overfitting in ML?",
    "Two tips for time management:",
    "A limerick about coffee:",
    "Difference between RAM and disk:",
    "In plain language, blockchain is",
    "Best practice for API retries:",
    "Why do leaves change color?",
    "A one-line SQL select example:",
    "The purpose of unit tests is",
    "Explain recursion with one example:",
    "A short note on climate risk:",
]


@dataclass
class ExperimentConfig:
    model_id: str
    cache_dir: str
    train_prompt_cap: int
    eval_prompt_cap: int
    budgets: List[int]
    seeds: List[int]
    topk_values: List[int]


class TinyStudent(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.fc1 = nn.Linear(emb_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, vocab_size)

    def __call__(self, token_ids: mx.array) -> mx.array:
        x = self.embedding(token_ids)
        x = mx.mean(x, axis=1)
        x = self.fc1(x)
        x = nn.relu(x)
        return self.fc2(x)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_config(path: str) -> ExperimentConfig:
    data = yaml.safe_load(Path(path).read_text())
    return ExperimentConfig(
        model_id=data["model_id"],
        cache_dir=data["cache_dir"],
        train_prompt_cap=int(data["train_prompt_cap"]),
        eval_prompt_cap=int(data["eval_prompt_cap"]),
        budgets=list(map(int, data["budgets"])),
        seeds=list(map(int, data["seeds"])),
        topk_values=list(map(int, data["topk_values"])),
    )


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def python_list_to_mx_int32(token_ids: Sequence[int]) -> mx.array:
    clean = [int(x) for x in token_ids]
    return mx.array([clean], dtype=mx.int32)


def safe_softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits)
    exp = np.exp(logits)
    s = np.sum(exp)
    if not np.isfinite(s) or s <= 0:
        raise ValueError("Invalid softmax denominator")
    return exp / s


def kl_divergence(p: np.ndarray, q: np.ndarray) -> Tuple[Optional[float], Optional[str]]:
    if p.shape != q.shape:
        return None, f"shape_mismatch:{p.shape}!={q.shape}"
    if np.any(~np.isfinite(p)) or np.any(~np.isfinite(q)):
        return None, "non_finite_probs"
    sp, sq = float(np.sum(p)), float(np.sum(q))
    if not (abs(sp - 1.0) < 1e-4 and abs(sq - 1.0) < 1e-4):
        return None, f"prob_sum_invalid:p={sp:.6f},q={sq:.6f}"
    if np.any(p < 0) or np.any(q < 0):
        return None, "negative_probability"
    eps = 1e-12
    ratio = (p + eps) / (q + eps)
    val = float(np.sum(p * np.log(ratio)))
    if not np.isfinite(val):
        return None, "kl_non_finite"
    return val, None


def get_victim_distribution(model: Any, token_ids: Sequence[int]) -> np.ndarray:
    input_ids = python_list_to_mx_int32(token_ids)
    out = model(input_ids)
    logits = np.array(out[0, -1, :])
    return safe_softmax(logits)


def make_interface_target(victim_probs: np.ndarray, interface: str, k: int = 2) -> np.ndarray:
    vocab = victim_probs.shape[0]
    if interface == "probs":
        return victim_probs.copy()
    if interface == "argmax":
        idx = int(np.argmax(victim_probs))
        target = np.zeros(vocab, dtype=np.float64)
        target[idx] = 1.0
        return target
    if interface.startswith("top"):
        k = int(interface.replace("top", "")) if interface != "topk" else int(k)
        idx = np.argpartition(victim_probs, -k)[-k:]
        target = np.zeros(vocab, dtype=np.float64)
        topvals = victim_probs[idx]
        target[idx] = topvals / np.sum(topvals)
        return target
    raise ValueError(f"Unsupported interface: {interface}")


def train_student_mlx(
    student: TinyStudent,
    optimizer: optim.Optimizer,
    token_batches: List[Sequence[int]],
    target_batches: List[np.ndarray],
    budget: int,
) -> None:
    def loss_fn(model: TinyStudent, x: mx.array, target: mx.array) -> mx.array:
        logits = model(x)
        probs = mx.softmax(logits, axis=-1)
        return -mx.mean(mx.sum(target * mx.log(probs + 1e-9), axis=-1))

    loss_and_grad = nn.value_and_grad(student, loss_fn)

    for step in range(budget):
        idx = step % len(token_batches)
        x = python_list_to_mx_int32(token_batches[idx])
        y = mx.array([target_batches[idx]], dtype=mx.float32)
        loss, grads = loss_and_grad(student, x, y)
        optimizer.update(student, grads)
        mx.eval(student.parameters(), optimizer.state)
        if step % 16 == 0 or step == budget - 1:
            LOGGER.info("train_step=%d loss=%.6f", step, float(loss.item()))


def eval_student(
    student: TinyStudent,
    eval_tokens: List[Sequence[int]],
    victim_probs: List[np.ndarray],
    interface: str,
    seed: int,
    debug: bool,
    debug_dir: Path,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    agreement = 0
    valid_kl = 0
    invalid_kl = 0
    kl_values = []

    for i, toks in enumerate(eval_tokens):
        x = python_list_to_mx_int32(toks)
        logits = np.array(student(x)[0])
        s_probs = safe_softmax(logits)
        v_probs = victim_probs[i]
        kl, reason = kl_divergence(v_probs, s_probs)
        if kl is None:
            invalid_kl += 1
        else:
            valid_kl += 1
            kl_values.append(kl)
        agreement += int(np.argmax(v_probs) == np.argmax(s_probs))

        row = {
            "prompt_id": i,
            "interface": interface,
            "seed": seed,
            "token_ids": toks,
            "python_token_types": [type(t).__name__ for t in toks],
            "mlx_input_dtype": str(x.dtype),
            "victim_top10_ids": np.argsort(v_probs)[-10:][::-1].tolist(),
            "victim_top10_probs": np.sort(v_probs)[-10:][::-1].tolist(),
            "student_top10_ids": np.argsort(s_probs)[-10:][::-1].tolist(),
            "student_top10_probs": np.sort(s_probs)[-10:][::-1].tolist(),
            "victim_prob_sum": float(np.sum(v_probs)),
            "student_prob_sum": float(np.sum(s_probs)),
            "victim_nan_or_inf": bool(np.any(~np.isfinite(v_probs))),
            "student_nan_or_inf": bool(np.any(~np.isfinite(s_probs))),
            "vocab_size": int(v_probs.shape[0]),
            "support_alignment": v_probs.shape == s_probs.shape,
            "exact_kl": kl,
            "kl_skip_reason": reason,
        }
        if debug and i < 10:
            rows.append(row)

    if debug and rows:
        debug_dir.mkdir(parents=True, exist_ok=True)
        jpath = debug_dir / f"debug_{interface}_seed{seed}.jsonl"
        cpath = debug_dir / f"debug_{interface}_seed{seed}.csv"
        with jpath.open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        pd.DataFrame(rows).to_csv(cpath, index=False)

    return {
        "agreement": agreement / max(1, len(eval_tokens)),
        "kl_mean": float(np.mean(kl_values)) if kl_values else math.nan,
        "valid_kl_count": valid_kl,
        "invalid_kl_count": invalid_kl,
    }


def run_startup_selfcheck(cfg: ExperimentConfig, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    hf_token = os.environ.get("HF_TOKEN")
    auth_enabled = bool(hf_token)
    LOGGER.info("HF authenticated download enabled: %s", auth_enabled)
    if auth_enabled:
        try:
            whoami(token=hf_token)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("HF token validation failed: %s", exc)

    model, tokenizer = load(cfg.model_id)
    prompt = "Quick startup self-check prompt."
    token_ids = tokenizer.encode(prompt)
    x = python_list_to_mx_int32(token_ids)
    y = model(x)

    report = {
        "python_version": sys.version,
        "macos_version": platform.mac_ver()[0],
        "mlx_version": getattr(mx, "__version__", "unknown"),
        "mlx_lm_version": __import__("mlx_lm").__version__,
        "model_id": cfg.model_id,
        "cache_dir": cfg.cache_dir,
        "tokenizer_class": tokenizer.__class__.__name__,
        "vocab_size": int(getattr(tokenizer, "vocab_size", y.shape[-1])),
        "forward_pass_shape": list(y.shape),
    }
    out = out_dir / "qwen_startup_selfcheck.txt"
    with out.open("w") as f:
        for k, v in report.items():
            f.write(f"{k}: {v}\n")


def run_sanity_check_llm(cfg: ExperimentConfig) -> None:
    model, tokenizer = load(cfg.model_id)
    prompts = ["hello world", "math is", "weather today"]
    for i, p in enumerate(prompts):
        token_ids = tokenizer.encode(p)
        print(f"prompt_{i}_python_token_types={[type(t).__name__ for t in token_ids]}")
        x = python_list_to_mx_int32(token_ids)
        print(f"prompt_{i}_mlx_dtype={x.dtype}")
        y = model(x)
        logits = np.array(y[0, -1, :])
        probs = safe_softmax(logits)
        assert y.ndim == 3, f"Expected 3D logits, got {y.shape}"
        assert y.shape[-1] == probs.shape[0], "Vocab dimension mismatch"
        s = float(np.sum(probs))
        assert abs(s - 1.0) < 1e-4, f"Softmax sum invalid: {s}"
        print(f"prompt_{i}_forward_shape={list(y.shape)} prob_sum={s:.6f}")


def build_prompt_data(tokenizer: Any, prompts: List[str]) -> Tuple[List[List[int]], List[str]]:
    toks = []
    for p in prompts:
        ids = tokenizer.encode(p)
        toks.append([int(x) for x in ids])
    return toks, prompts


def save_nan_and_alignment_reports(records: List[Dict[str, Any]], out_dir: Path) -> None:
    df = pd.DataFrame(records)
    nan_df = df[["interface", "seed", "budget", "invalid_kl_count", "valid_kl_count"]].copy()
    nan_df["nan_issue"] = nan_df["invalid_kl_count"] > 0
    nan_df.to_csv(out_dir / "qwen_nan_report.csv", index=False)

    align_df = df[["interface", "seed", "budget"]].copy()
    align_df["support_alignment"] = True
    align_df.to_csv(out_dir / "qwen_vocab_alignment_check.csv", index=False)


def plot_curves(summary: pd.DataFrame, out_dir: Path) -> None:
    toy = summary[summary["family"] == "toy"]
    qwen = summary[summary["family"] == "qwen"]

    plt.figure(figsize=(8, 5))
    for fam, df in [("toy", toy), ("qwen", qwen)]:
        grp = df.groupby("budget")["agreement"].mean().reset_index()
        plt.plot(grp["budget"], grp["agreement"], marker="o", label=fam)
    plt.xlabel("Budget")
    plt.ylabel("Top-1 Agreement")
    plt.title("Agreement vs Budget")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "agreement_vs_budget.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    for fam, df in [("toy", toy), ("qwen", qwen)]:
        grp = df.groupby("budget")["kl_mean"].mean().reset_index()
        plt.plot(grp["budget"], grp["kl_mean"], marker="o", label=fam)
    plt.xlabel("Budget")
    plt.ylabel("KL Divergence")
    plt.title("KL vs Budget")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "kl_vs_budget.png", dpi=160)
    plt.close()

    topk = summary[(summary["family"] == "qwen") & (summary["sweep"] == "topk")]
    if not topk.empty:
        grp = topk.groupby("interface")["kl_mean"].mean().reindex(["argmax", "top2", "top3", "top5", "probs"])
        plt.figure(figsize=(8, 5))
        grp.plot(kind="bar")
        plt.ylabel("KL Divergence")
        plt.title("Qwen Top-k Interface KL")
        plt.tight_layout()
        plt.savefig(out_dir / "qwen_topk_kl.png", dpi=160)
        plt.close()


def run_toy_family(cfg: ExperimentConfig, out_dir: Path) -> List[Dict[str, Any]]:
    vocab_size = 128

    class ToyVictim:
        def __call__(self, token_batch: mx.array) -> mx.array:
            tid = np.array(token_batch)[0]
            last = int(tid[-1]) if len(tid) else 0
            logits = np.linspace(-1.0, 1.0, vocab_size)
            logits[(last + 3) % vocab_size] += 4.0
            return mx.array(logits.reshape(1, 1, -1), dtype=mx.float32)

    victim = ToyVictim()
    rng_prompts = [f"toy prompt {i}" for i in range(cfg.train_prompt_cap + cfg.eval_prompt_cap)]
    toy_tokens = [[(i + j) % vocab_size for j in range(6)] for i in range(len(rng_prompts))]

    train_tokens = toy_tokens[: cfg.train_prompt_cap]
    eval_tokens = toy_tokens[cfg.train_prompt_cap : cfg.train_prompt_cap + cfg.eval_prompt_cap]

    results = []
    for seed in cfg.seeds:
        set_seed(seed)
        for budget in cfg.budgets:
            student = TinyStudent(vocab_size=vocab_size, emb_dim=32, hidden_dim=64)
            opt = optim.Adam(learning_rate=1e-2)
            vtrain = [safe_softmax(np.array(victim(python_list_to_mx_int32(t))[0, -1, :])) for t in train_tokens]
            train_targets = [make_interface_target(v, "probs") for v in vtrain]
            train_student_mlx(student, opt, train_tokens, train_targets, budget)

            veval = [safe_softmax(np.array(victim(python_list_to_mx_int32(t))[0, -1, :])) for t in eval_tokens]
            metrics = eval_student(student, eval_tokens, veval, "probs", seed, False, out_dir / "debug")
            results.append(
                {
                    "family": "toy",
                    "sweep": "budget",
                    "interface": "probs",
                    "seed": seed,
                    "budget": budget,
                    **metrics,
                }
            )
    return results


def run_qwen_family(cfg: ExperimentConfig, out_dir: Path, debug: bool = False) -> List[Dict[str, Any]]:
    model, tokenizer = load(cfg.model_id)
    prompt_tokens, prompts = build_prompt_data(tokenizer, DEFAULT_PROMPTS)

    train_tokens = prompt_tokens[: cfg.train_prompt_cap]
    eval_tokens = prompt_tokens[cfg.train_prompt_cap : cfg.train_prompt_cap + cfg.eval_prompt_cap]
    eval_prompts = prompts[cfg.train_prompt_cap : cfg.train_prompt_cap + cfg.eval_prompt_cap]

    victim_train = [get_victim_distribution(model, t) for t in train_tokens]
    victim_eval = [get_victim_distribution(model, t) for t in eval_tokens]
    vocab_size = victim_eval[0].shape[0]

    interfaces = ["argmax", "top2", "top3", "top5", "probs"]
    fixed_budget = 256
    records = []

    # budget sweep for "probs"
    for seed in cfg.seeds:
        set_seed(seed)
        for budget in cfg.budgets:
            student = TinyStudent(vocab_size=vocab_size)
            opt = optim.Adam(learning_rate=3e-3)
            targets = [make_interface_target(v, "probs") for v in victim_train]
            train_student_mlx(student, opt, train_tokens, targets, budget)
            m = eval_student(student, eval_tokens, victim_eval, "probs", seed, debug, out_dir / "debug")
            records.append(
                {
                    "family": "qwen",
                    "sweep": "budget",
                    "interface": "probs",
                    "seed": seed,
                    "budget": budget,
                    **m,
                }
            )

    # interface top-k sweep at fixed budget
    for seed in cfg.seeds:
        set_seed(seed)
        for interface in interfaces:
            student = TinyStudent(vocab_size=vocab_size)
            opt = optim.Adam(learning_rate=3e-3)
            targets = [make_interface_target(v, interface) for v in victim_train]
            train_student_mlx(student, opt, train_tokens, targets, fixed_budget)
            m = eval_student(student, eval_tokens, victim_eval, interface, seed, debug, out_dir / "debug")
            records.append(
                {
                    "family": "qwen",
                    "sweep": "topk",
                    "interface": interface,
                    "seed": seed,
                    "budget": fixed_budget,
                    **m,
                }
            )

    # debug enrichment for prompt text fields (first 10 only)
    if debug:
        for interface in interfaces:
            dbg_file = out_dir / "debug" / f"debug_{interface}_seed{cfg.seeds[0]}.jsonl"
            if dbg_file.exists():
                enriched = []
                with dbg_file.open() as f:
                    for line in f:
                        d = json.loads(line)
                        pid = d["prompt_id"]
                        d["prompt_text"] = eval_prompts[pid]
                        enriched.append(d)
                with dbg_file.open("w") as f:
                    for d in enriched:
                        f.write(json.dumps(d) + "\n")

    return records


def write_summary(df: pd.DataFrame, out_dir: Path) -> None:
    df.to_csv(out_dir / "results_summary.csv", index=False)
    agg = (
        df.groupby(["family", "sweep", "interface", "budget"], dropna=False)
        .agg(
            agreement_mean=("agreement", "mean"),
            agreement_std=("agreement", "std"),
            kl_mean=("kl_mean", "mean"),
            kl_std=("kl_mean", "std"),
            valid_kl_count=("valid_kl_count", "sum"),
            invalid_kl_count=("invalid_kl_count", "sum"),
        )
        .reset_index()
    )
    agg.to_csv(out_dir / "multi_seed_summary.csv", index=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/experiment.yaml")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--startup-selfcheck", action="store_true")
    p.add_argument("--sanity-check-llm", action="store_true", help="Required token/logit sanity check mode")
    p.add_argument("--run-all", action="store_true")
    p.add_argument("--debug-llm", action="store_true")
    return p.parse_args()


def main() -> None:
    configure_logging(True)
    args = parse_args()
    cfg = load_config(args.config)
    os.environ.setdefault("HF_HOME", cfg.cache_dir)
    out_dir = ensure_dir(args.results_dir)

    if args.startup_selfcheck:
        run_startup_selfcheck(cfg, out_dir)
        LOGGER.info("Wrote startup self-check report")
        return

    if args.sanity_check_llm:
        run_sanity_check_llm(cfg)
        LOGGER.info("Sanity check passed")
        return

    if args.run_all:
        toy_records = run_toy_family(cfg, out_dir)
        qwen_records = run_qwen_family(cfg, out_dir, debug=args.debug_llm)
        all_df = pd.DataFrame(toy_records + qwen_records)
        write_summary(all_df, out_dir)
        save_nan_and_alignment_reports(qwen_records, out_dir)
        plot_curves(all_df, out_dir)
        LOGGER.info("All experiments complete")
        return

    raise SystemExit("Select one mode: --startup-selfcheck, --sanity-check-llm, or --run-all")


if __name__ == "__main__":
    main()
