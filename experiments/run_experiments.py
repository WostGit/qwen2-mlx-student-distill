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
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import pandas as pd
from huggingface_hub import login
from mlx_lm import load


RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

QWEN_MODEL_ID = "Qwen/Qwen2-0.5B-Instruct"
DEFAULT_BUDGETS = [64, 128, 256, 512]
DEFAULT_SEEDS = [0, 1]
PROMPTS_QWEN = [
    "The capital of France is",
    "A quick summary of photosynthesis:",
    "In one sentence, define entropy.",
    "Python list comprehensions are useful because",
    "Name a prime number between 50 and 60:",
    "The moon landing happened in",
    "One health tip for better sleep is",
    "Write one line about reinforcement learning:",
    "Water boils at sea level at",
    "A haiku about coding:",
    "The opposite of scarcity is",
    "Translate 'good morning' to Spanish:",
    "2 + 2 * 3 equals",
    "A safe SQL query should",
    "In networking, TCP stands for",
    "A simple definition of GDP:",
    "The first step in debugging is",
    "Name one greenhouse gas:",
    "A polite email opening:",
    "One sentence about gravity:",
]


@dataclass
class RunConfig:
    budgets: List[int]
    seeds: List[int]
    qwen_subset_vocab_size: int = 2048
    qwen_train_prompts: int = 16
    qwen_eval_prompts: int = 12
    toy_vocab: int = 64
    toy_train_samples: int = 160
    toy_eval_samples: int = 64
    qwen_lr: float = 1e-2
    toy_lr: float = 5e-2


class TinyStudent(nn.Module):
    def __init__(self, vocab_size: int, hidden: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden)
        self.fc1 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, vocab_size)

    def __call__(self, token_ids: mx.array) -> mx.array:
        x = self.embedding(token_ids)
        x = x[-1]  # one-step next-token from final context token
        x = nn.relu(self.fc1(x))
        return self.fc2(x)


def log(msg: str) -> None:
    print(msg, flush=True)


def safe_token_ids_to_mx(token_ids: Sequence[int]) -> mx.array:
    if not isinstance(token_ids, (list, tuple)):
        raise TypeError(f"token_ids must be list/tuple[int], got {type(token_ids)}")
    for idx, tok in enumerate(token_ids):
        if not isinstance(tok, int):
            raise TypeError(f"token at position {idx} is {type(tok)}, expected int")
    return mx.array(list(token_ids), dtype=mx.int32)


def softmax_np(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    expv = np.exp(shifted)
    denom = np.sum(expv)
    if denom <= 0 or not np.isfinite(denom):
        raise ValueError("Invalid softmax denominator")
    return expv / denom


def kl_divergence_np(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    if p.shape != q.shape:
        raise ValueError("KL shape mismatch")
    if np.any(~np.isfinite(p)) or np.any(~np.isfinite(q)):
        raise ValueError("KL received non-finite arrays")
    p = p.astype(np.float64)
    q = q.astype(np.float64)
    if abs(p.sum() - 1.0) > 1e-6 or abs(q.sum() - 1.0) > 1e-6:
        raise ValueError("KL received arrays that do not sum to 1")
    if np.any(p < 0) or np.any(q < 0):
        raise ValueError("Negative probabilities are invalid")
    q_safe = np.clip(q, eps, 1.0)
    mask = p > 0
    return float(np.sum(p[mask] * (np.log(p[mask]) - np.log(q_safe[mask]))))


def build_interface_target(full_probs: np.ndarray, interface: str, k: int = 3) -> np.ndarray:
    out = np.zeros_like(full_probs)
    if interface == "argmax":
        out[int(np.argmax(full_probs))] = 1.0
    elif interface.startswith("top"):
        if interface == "topk":
            use_k = k
        else:
            use_k = int(interface.replace("top", ""))
        top_idx = np.argpartition(full_probs, -use_k)[-use_k:]
        out[top_idx] = full_probs[top_idx]
        s = out.sum()
        if s <= 0:
            raise ValueError("topk interface sum is zero")
        out /= s
    elif interface == "probs":
        out = full_probs.copy()
    else:
        raise ValueError(f"Unknown interface: {interface}")
    return out


def setup_hf_auth() -> None:
    token = os.environ.get("HF_TOKEN", "").strip()
    enabled = bool(token)
    log(f"[startup] HF authenticated download enabled: {enabled}")
    if enabled:
        login(token=token, add_to_git_credential=False)


def load_qwen() -> Tuple[object, object]:
    setup_hf_auth()
    log(f"[startup] Loading model via mlx-lm: {QWEN_MODEL_ID}")
    model, tokenizer = load(QWEN_MODEL_ID)
    return model, tokenizer


def encode_prompt(tokenizer, prompt: str) -> List[int]:
    token_ids = tokenizer.encode(prompt)
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    token_ids = [int(x) for x in token_ids]
    return token_ids


def get_next_token_logits(model, token_ids: List[int]) -> np.ndarray:
    x = safe_token_ids_to_mx(token_ids)
    logits = model(x[None, :])
    last_logits = np.array(logits[0, -1, :])
    return last_logits


def startup_selfcheck() -> None:
    model, tokenizer = load_qwen()
    cache_dir = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))

    prompt = "Hello from qwen2-mlx-student-distill."
    token_ids = encode_prompt(tokenizer, prompt)
    logits = get_next_token_logits(model, token_ids)
    probs = softmax_np(logits)

    report_lines = [
        f"python_version={sys.version}",
        f"macos_version={platform.mac_ver()}",
        f"mlx_version={getattr(mx, '__version__', 'unknown')}",
        f"mlx_lm_version={__import__('mlx_lm').__version__ if hasattr(__import__('mlx_lm'), '__version__') else 'unknown'}",
        f"model_id={QWEN_MODEL_ID}",
        f"cache_dir={cache_dir}",
        f"tokenizer_class={tokenizer.__class__.__name__}",
        f"vocab_size={len(tokenizer)}",
        f"forward_pass_prompt={prompt}",
        f"forward_pass_input_len={len(token_ids)}",
        f"forward_pass_logits_shape={logits.shape}",
        f"forward_pass_prob_sum={float(probs.sum()):.8f}",
    ]
    out_path = RESULTS_DIR / "qwen_startup_selfcheck.txt"
    out_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    log(f"[startup] Wrote self-check report: {out_path}")


def sanity_check_llm() -> None:
    model, tokenizer = load_qwen()
    sanity_prompts = ["Hello", "MLX is", "One token please"]
    vocab_size = len(tokenizer)
    tol = 1e-5

    for i, prompt in enumerate(sanity_prompts):
        token_ids = encode_prompt(tokenizer, prompt)
        py_types = [type(x).__name__ for x in token_ids]
        log(f"[sanity] prompt[{i}]={prompt!r}")
        log(f"[sanity] python token types={py_types[:8]}")

        x = safe_token_ids_to_mx(token_ids)
        log(f"[sanity] mlx dtype={x.dtype}")
        logits = model(x[None, :])
        logits_np = np.array(logits[0, -1, :])
        if logits_np.shape[0] != vocab_size:
            raise RuntimeError(f"Vocab mismatch: logits {logits_np.shape[0]} vs tokenizer {vocab_size}")

        probs = softmax_np(logits_np)
        p_sum = float(probs.sum())
        log(f"[sanity] logits shape={logits_np.shape}, prob sum={p_sum:.8f}")
        if abs(p_sum - 1.0) > tol:
            raise RuntimeError(f"Probability sum check failed: {p_sum}")
    log("[sanity] all checks passed")


def train_student_mlx(
    model: TinyStudent,
    optimizer: optim.Adam,
    train_inputs: List[List[int]],
    train_targets: List[np.ndarray],
    budget: int,
) -> None:
    def loss_fn(m: TinyStudent, token_ids: mx.array, target: mx.array) -> mx.array:
        logits = m(token_ids)
        log_probs = nn.log_softmax(logits)
        return -mx.sum(target * log_probs)

    grad_fn = nn.value_and_grad(model, loss_fn)

    n = len(train_inputs)
    for step in range(budget):
        idx = step % n
        token_ids = safe_token_ids_to_mx(train_inputs[idx])
        target = mx.array(train_targets[idx], dtype=mx.float32)
        loss, grads = grad_fn(model, token_ids, target)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        if step % max(1, budget // 8) == 0:
            log(f"[train] step={step:04d}/{budget} loss={float(loss):.6f}")


def toy_victim_probs(vocab_size: int, token_ids: List[int]) -> np.ndarray:
    vec = np.zeros(vocab_size, dtype=np.float64)
    s = sum(token_ids) % vocab_size
    vec[s] = 2.0
    vec[(s + 1) % vocab_size] = 1.0
    vec[(s + 7) % vocab_size] = 0.5
    vec += 0.01
    vec /= vec.sum()
    return vec


def make_toy_dataset(vocab_size: int, n: int, seed: int) -> List[List[int]]:
    rng = random.Random(seed)
    data: List[List[int]] = []
    for _ in range(n):
        length = rng.randint(3, 8)
        data.append([rng.randint(0, vocab_size - 1) for _ in range(length)])
    return data


def evaluate(
    model: TinyStudent,
    eval_inputs: List[List[int]],
    eval_victim: List[np.ndarray],
    support_ids: Optional[np.ndarray],
    debug_rows: Optional[List[Dict]],
    prefix_meta: Dict,
) -> Dict[str, float]:
    agreements = 0
    valid_kl = 0
    invalid_kl = 0
    kl_values = []

    for i, token_ids in enumerate(eval_inputs):
        logits = np.array(model(safe_token_ids_to_mx(token_ids)))
        stud_full = softmax_np(logits)
        vict_full = eval_victim[i]

        if support_ids is not None:
            vict = vict_full[support_ids]
            stud = stud_full[support_ids]
            vmass = vict.sum()
            smass = stud.sum()
            aligned = vmass > 0 and smass > 0 and np.isfinite(vmass) and np.isfinite(smass)
            reason = ""
            if not aligned:
                reason = f"invalid support mass vmass={vmass}, smass={smass}"
            else:
                vict = vict / vmass
                stud = stud / smass
        else:
            vict = vict_full
            stud = stud_full
            aligned = True
            reason = ""

        stud_arg = int(np.argmax(stud))
        vict_arg = int(np.argmax(vict))
        agreements += int(stud_arg == vict_arg)

        kl = None
        try:
            if not aligned:
                raise ValueError(reason)
            kl = kl_divergence_np(vict, stud)
            kl_values.append(kl)
            valid_kl += 1
        except Exception as exc:
            invalid_kl += 1
            reason = str(exc)

        if debug_rows is not None and len(debug_rows) < 10:
            debug_rows.append(
                {
                    **prefix_meta,
                    "example_index": i,
                    "token_ids": token_ids,
                    "python_token_types": [type(t).__name__ for t in token_ids],
                    "mlx_dtype_after_conversion": str(safe_token_ids_to_mx(token_ids).dtype),
                    "victim_top10_ids": np.argsort(vict)[-10:][::-1].tolist(),
                    "victim_top10_probs": np.sort(vict)[-10:][::-1].tolist(),
                    "student_top10_ids": np.argsort(stud)[-10:][::-1].tolist(),
                    "student_top10_probs": np.sort(stud)[-10:][::-1].tolist(),
                    "victim_prob_sum": float(vict.sum()),
                    "student_prob_sum": float(stud.sum()),
                    "nan_inf_victim": bool(np.any(~np.isfinite(vict))),
                    "nan_inf_student": bool(np.any(~np.isfinite(stud))),
                    "vocab_size": len(stud),
                    "support_alignment": aligned,
                    "exact_kl": kl,
                    "kl_skip_reason": "" if kl is not None else reason,
                }
            )

    return {
        "agreement": agreements / max(1, len(eval_inputs)),
        "kl_divergence": float(np.mean(kl_values)) if kl_values else float("nan"),
        "valid_kl_count": valid_kl,
        "invalid_kl_count": invalid_kl,
    }


def run_toy_family(cfg: RunConfig, summary_rows: List[Dict]) -> None:
    log("\n===== TOY BASELINE FAMILY =====")
    interfaces = ["argmax", "top3", "probs"]
    for seed in cfg.seeds:
        random.seed(seed)
        np.random.seed(seed)
        train_inputs = make_toy_dataset(cfg.toy_vocab, cfg.toy_train_samples, seed)
        eval_inputs = make_toy_dataset(cfg.toy_vocab, cfg.toy_eval_samples, seed + 99)

        for interface in interfaces:
            train_victim_full = [toy_victim_probs(cfg.toy_vocab, x) for x in train_inputs]
            eval_victim_full = [toy_victim_probs(cfg.toy_vocab, x) for x in eval_inputs]
            train_targets = [build_interface_target(p, interface) for p in train_victim_full]

            for budget in cfg.budgets:
                log(f"[toy] seed={seed} interface={interface} budget={budget}")
                student = TinyStudent(vocab_size=cfg.toy_vocab, hidden=64)
                opt = optim.Adam(learning_rate=cfg.toy_lr)
                train_student_mlx(student, opt, train_inputs, train_targets, budget)
                metrics = evaluate(student, eval_inputs, eval_victim_full, None, None, {})
                summary_rows.append(
                    {
                        "family": "toy",
                        "seed": seed,
                        "interface": interface,
                        "budget": budget,
                        **metrics,
                    }
                )


def make_qwen_vocab_subset(vocab_size: int, subset_size: int) -> np.ndarray:
    ids = np.arange(min(vocab_size, subset_size), dtype=np.int64)
    pd.DataFrame({"subset_index": np.arange(len(ids)), "token_id": ids}).to_csv(
        RESULTS_DIR / "qwen_vocab_subset_mapping.csv", index=False
    )
    return ids


def run_qwen_family(cfg: RunConfig, summary_rows: List[Dict], debug_llm: bool = False) -> None:
    log("\n===== QWEN2-0.5B FAMILY =====")
    model, tokenizer = load_qwen()
    vocab_size = len(tokenizer)
    subset_ids = make_qwen_vocab_subset(vocab_size, cfg.qwen_subset_vocab_size)

    train_prompts = PROMPTS_QWEN[: cfg.qwen_train_prompts]
    eval_prompts = PROMPTS_QWEN[cfg.qwen_train_prompts : cfg.qwen_train_prompts + cfg.qwen_eval_prompts]
    if len(eval_prompts) == 0:
        eval_prompts = PROMPTS_QWEN[-cfg.qwen_eval_prompts :]

    interfaces_budget = ["argmax", "top3", "probs"]
    nan_rows = []
    align_rows = []

    for seed in cfg.seeds:
        random.seed(seed)
        np.random.seed(seed)

        train_inputs = [encode_prompt(tokenizer, p) for p in train_prompts]
        eval_inputs = [encode_prompt(tokenizer, p) for p in eval_prompts]

        train_full = [softmax_np(get_next_token_logits(model, t)) for t in train_inputs]
        eval_full = [softmax_np(get_next_token_logits(model, t)) for t in eval_inputs]

        for interface in interfaces_budget:
            train_targets_full = [build_interface_target(v, interface) for v in train_full]
            train_targets_sub = []
            filtered_inputs = []
            for token_ids, target_full in zip(train_inputs, train_targets_full):
                target_sub = target_full[subset_ids]
                mass = target_sub.sum()
                if mass <= 0 or not np.isfinite(mass):
                    continue
                target_sub = target_sub / mass
                filtered_inputs.append(token_ids)
                train_targets_sub.append(target_sub)

            for budget in cfg.budgets:
                log(f"[qwen-budget] seed={seed} interface={interface} budget={budget}")
                student = TinyStudent(vocab_size=len(subset_ids), hidden=96)
                opt = optim.Adam(learning_rate=cfg.qwen_lr)
                train_student_mlx(student, opt, filtered_inputs, train_targets_sub, budget)

                debug_rows = [] if debug_llm else None
                metrics = evaluate(
                    student,
                    eval_inputs,
                    eval_full,
                    subset_ids,
                    debug_rows,
                    {"family": "qwen", "seed": seed, "interface": interface, "budget": budget},
                )

                summary_rows.append(
                    {
                        "family": "qwen",
                        "seed": seed,
                        "interface": interface,
                        "budget": budget,
                        **metrics,
                    }
                )

                if debug_rows is not None:
                    jpath = RESULTS_DIR / f"debug_qwen_seed{seed}_{interface}_budget{budget}.jsonl"
                    cpath = RESULTS_DIR / f"debug_qwen_seed{seed}_{interface}_budget{budget}.csv"
                    with jpath.open("w", encoding="utf-8") as f:
                        for row in debug_rows:
                            f.write(json.dumps(row) + "\n")
                    pd.DataFrame(debug_rows).to_csv(cpath, index=False)

                nan_rows.append(
                    {
                        "seed": seed,
                        "interface": interface,
                        "budget": budget,
                        "invalid_kl_count": metrics["invalid_kl_count"],
                        "valid_kl_count": metrics["valid_kl_count"],
                        "kl_is_nan": bool(np.isnan(metrics["kl_divergence"])),
                    }
                )
                align_rows.append(
                    {
                        "seed": seed,
                        "interface": interface,
                        "budget": budget,
                        "subset_size": len(subset_ids),
                        "full_vocab_size": vocab_size,
                        "supports_aligned": metrics["invalid_kl_count"] == 0,
                    }
                )

        topk_interfaces = ["argmax", "top2", "top3", "top5", "probs"]
        fixed_budget = 256
        for interface in topk_interfaces:
            train_targets_full = [build_interface_target(v, interface) for v in train_full]
            train_targets_sub = []
            filtered_inputs = []
            for token_ids, target_full in zip(train_inputs, train_targets_full):
                tsub = target_full[subset_ids]
                mass = tsub.sum()
                if mass <= 0 or not np.isfinite(mass):
                    continue
                train_targets_sub.append(tsub / mass)
                filtered_inputs.append(token_ids)

            log(f"[qwen-topk] seed={seed} interface={interface} budget={fixed_budget}")
            student = TinyStudent(vocab_size=len(subset_ids), hidden=96)
            opt = optim.Adam(learning_rate=cfg.qwen_lr)
            train_student_mlx(student, opt, filtered_inputs, train_targets_sub, fixed_budget)
            metrics = evaluate(student, eval_inputs, eval_full, subset_ids, None, {})
            summary_rows.append(
                {
                    "family": "qwen_topk",
                    "seed": seed,
                    "interface": interface,
                    "budget": fixed_budget,
                    **metrics,
                }
            )

    pd.DataFrame(nan_rows).to_csv(RESULTS_DIR / "qwen_nan_report.csv", index=False)
    pd.DataFrame(align_rows).to_csv(RESULTS_DIR / "qwen_vocab_alignment_check.csv", index=False)


def render_plots(df: pd.DataFrame) -> None:
    base = df[df["family"].isin(["toy", "qwen"])]
    agg = (
        base.groupby(["family", "interface", "budget"], as_index=False)
        .agg(agreement=("agreement", "mean"), kl=("kl_divergence", "mean"))
    )

    plt.figure(figsize=(10, 5))
    for (family, interface), sub in agg.groupby(["family", "interface"]):
        plt.plot(sub["budget"], sub["agreement"], marker="o", label=f"{family}-{interface}")
    plt.xlabel("Budget")
    plt.ylabel("Top-1 Agreement")
    plt.title("Agreement vs Budget")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "agreement_vs_budget.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 5))
    for (family, interface), sub in agg.groupby(["family", "interface"]):
        plt.plot(sub["budget"], sub["kl"], marker="o", label=f"{family}-{interface}")
    plt.xlabel("Budget")
    plt.ylabel("KL divergence")
    plt.title("KL vs Budget")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "kl_vs_budget.png", dpi=150)
    plt.close()

    topk = df[df["family"] == "qwen_topk"]
    if not topk.empty:
        topk_agg = topk.groupby("interface", as_index=False).agg(kl=("kl_divergence", "mean"))
        plt.figure(figsize=(8, 5))
        plt.bar(topk_agg["interface"], topk_agg["kl"])
        plt.title("Qwen top-k interface KL (fixed budget)")
        plt.xlabel("Interface")
        plt.ylabel("KL divergence")
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "qwen_topk_kl.png", dpi=150)
        plt.close()


def write_summaries(summary_rows: List[Dict]) -> None:
    df = pd.DataFrame(summary_rows)
    df.to_csv(RESULTS_DIR / "results_summary.csv", index=False)

    multi = (
        df.groupby(["family", "interface", "budget"], as_index=False)
        .agg(
            agreement_mean=("agreement", "mean"),
            agreement_std=("agreement", "std"),
            kl_mean=("kl_divergence", "mean"),
            kl_std=("kl_divergence", "std"),
            valid_kl_count=("valid_kl_count", "sum"),
            invalid_kl_count=("invalid_kl_count", "sum"),
        )
    )
    multi.to_csv(RESULTS_DIR / "multi_seed_summary.csv", index=False)

    render_plots(df)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MLX interface leakage experiments")
    p.add_argument("--startup-selfcheck", action="store_true")
    p.add_argument("--sanity-check-llm", action="store_true")
    p.add_argument("--run-all", action="store_true")
    p.add_argument("--debug-llm", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = RunConfig(budgets=DEFAULT_BUDGETS, seeds=DEFAULT_SEEDS)

    if args.startup_selfcheck:
        startup_selfcheck()
        return

    if args.sanity_check_llm:
        sanity_check_llm()
        return

    if args.run_all:
        summary_rows: List[Dict] = []
        run_toy_family(cfg, summary_rows)
        run_qwen_family(cfg, summary_rows, debug_llm=args.debug_llm)
        write_summaries(summary_rows)
        log("[done] experiments completed")
        return

    raise SystemExit("No mode selected. Use --startup-selfcheck, --sanity-check-llm, or --run-all")


if __name__ == "__main__":
    main()
