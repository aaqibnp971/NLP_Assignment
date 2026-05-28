"""
demo.py — Track A: Dynamic Uncertainty-Aware Attribution
CS F429 Natural Language Processing, BITS Pilani Dubai Campus
Team 8: Zayaan Ali, Aaqib Pangarkar, Ridhwan Ahamed, Mariah Shania

Usage:
    python demo.py --passage "Your answer text here" --context "Your retrieved context here"

Example:
    python demo.py \
        --passage "Anne Frank was born in Frankfurt in 1929 and died in 1945." \
        --context "Anne Frank was a Jewish diarist born in Frankfurt am Main on 12 June 1929."
"""

import argparse
import warnings
import sys

warnings.filterwarnings("ignore")

# ── Dependency check ────────────────────────────────────────────────────────
REQUIRED = ["torch", "transformers", "numpy", "scipy", "sklearn"]
missing = []
for pkg in REQUIRED:
    try:
        __import__(pkg)
    except ImportError:
        missing.append(pkg)
if missing:
    print(f"[ERROR] Missing packages: {missing}")
    print("Run: pip install torch transformers numpy scipy scikit-learn")
    sys.exit(1)

import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# ── Model loading ────────────────────────────────────────────────────────────
MODEL_NAME = "gpt2"
DEVICE = "cpu"

print(f"[INFO] Loading {MODEL_NAME} ...")
tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
model.eval()
model.to(DEVICE)
print("[INFO] Model loaded.\n")


# ── Forward pass helper ──────────────────────────────────────────────────────
def get_token_probs(prefix: str, answer: str):
    """
    Run a single forward pass of GPT-2.
    Returns a list of probability distributions (one per answer token),
    each of shape [vocab_size].
    """
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    answer_ids = tokenizer.encode(answer, add_special_tokens=False)

    input_ids = torch.tensor([prefix_ids + answer_ids], dtype=torch.long).to(DEVICE)

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits  # [1, seq_len, vocab_size]

    # Extract logits for answer tokens only (offset by 1 for next-token prediction)
    n_prefix = len(prefix_ids)
    answer_logits = logits[0, n_prefix - 1 : n_prefix - 1 + len(answer_ids), :]

    probs = torch.softmax(answer_logits, dim=-1).cpu().numpy()  # [T, vocab_size]
    return probs, answer_ids


# ── Signal computations ──────────────────────────────────────────────────────
def shannon_entropy(prob_vec: np.ndarray) -> float:
    p = prob_vec + 1e-10
    p /= p.sum()
    return float(-np.sum(p * np.log(p)))


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p || q)"""
    p = p + 1e-10; p /= p.sum()
    q = q + 1e-10; q /= q.sum()
    return float(np.sum(p * np.log(p / q)))


def compute_signals(passage: str, context: str):
    """
    Compute four token-level uncertainty signals.

    Args:
        passage : the generated answer text
        context : the retrieved context document(s)

    Returns:
        dict with per-token arrays for each signal,
        token strings, and composite score
    """
    # Two forward passes
    probs_with, answer_ids = get_token_probs(context + " " + passage, passage)
    probs_no, _            = get_token_probs(passage, passage)

    T = min(len(probs_with), len(probs_no))
    probs_with = probs_with[:T]
    probs_no   = probs_no[:T]
    answer_ids = answer_ids[:T]

    tokens = [tokenizer.decode([tok_id]) for tok_id in answer_ids]

    info_gain    = np.zeros(T)
    kl_divs      = np.zeros(T)
    conf_drop    = np.zeros(T)
    sem_entropy  = np.zeros(T)  # sequence-level approx: replicated per token

    for t in range(T):
        h_no   = shannon_entropy(probs_no[t])
        h_with = shannon_entropy(probs_with[t])
        info_gain[t] = h_no - h_with                         # ΔH_t
        kl_divs[t]   = kl_divergence(probs_with[t], probs_no[t])  # KL_t
        conf_drop[t] = probs_no[t].max() - probs_with[t].max()    # Δc_t

    # Semantic entropy: sequence-level approximation (constant across tokens)
    overall_h = float(np.mean([shannon_entropy(probs_no[t]) for t in range(T)]))
    sem_entropy[:] = overall_h

    # Min-max normalise each signal to [0, 1]
    def minmax(arr):
        r = arr.max() - arr.min()
        return (arr - arr.min()) / (r + 1e-10)

    ig_n  = minmax(info_gain)
    kl_n  = minmax(kl_divs)
    cd_n  = minmax(conf_drop)
    se_n  = minmax(sem_entropy)

    # Composite: weights learned by logistic regression on RAGTruth
    # Semantic entropy excluded (negative weight). Remaining weights normalised.
    W = {"ig": 0.15, "kl": 0.55, "cd": 0.30}
    composite = W["ig"] * ig_n + W["kl"] * kl_n + W["cd"] * cd_n

    return {
        "tokens"        : tokens,
        "information_gain" : info_gain.tolist(),
        "kl_divergence"    : kl_divs.tolist(),
        "confidence_drop"  : conf_drop.tolist(),
        "semantic_entropy" : sem_entropy.tolist(),
        "composite_score"  : composite.tolist(),
    }


# ── Output formatter ─────────────────────────────────────────────────────────
THRESHOLD = 0.50

def print_results(results: dict):
    tokens    = results["tokens"]
    composite = results["composite_score"]
    ig        = results["information_gain"]
    kl        = results["kl_divergence"]
    cd        = results["confidence_drop"]
    se        = results["semantic_entropy"]

    print("=" * 90)
    print(f"{'TOKEN':<20} {'IG':>8} {'KL':>10} {'CONF_DROP':>10} {'SE':>8} {'COMPOSITE':>10} {'LABEL':>12}")
    print("-" * 90)

    for i, tok in enumerate(tokens):
        label = "HALLUCINATED" if composite[i] < THRESHOLD else "faithful"
        print(
            f"{repr(tok):<20} "
            f"{ig[i]:>8.4f} "
            f"{kl[i]:>10.4f} "
            f"{cd[i]:>10.4f} "
            f"{se[i]:>8.4f} "
            f"{composite[i]:>10.4f} "
            f"{label:>12}"
        )

    print("=" * 90)
    avg_comp = float(np.mean(composite))
    seq_label = "HALLUCINATED" if avg_comp < THRESHOLD else "FAITHFUL"
    print(f"\nSequence-level composite score : {avg_comp:.4f}")
    print(f"Sequence-level prediction      : {seq_label}  (threshold τ = {THRESHOLD})")
    print()


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Track A: Token-level hallucination detection via uncertainty attribution"
    )
    parser.add_argument(
        "--passage",
        type=str,
        required=True,
        help="The generated answer text to evaluate"
    )
    parser.add_argument(
        "--context",
        type=str,
        required=True,
        help="The retrieved context document(s)"
    )
    args = parser.parse_args()

    print(f"[INFO] Passage : {args.passage[:80]}{'...' if len(args.passage) > 80 else ''}")
    print(f"[INFO] Context : {args.context[:80]}{'...' if len(args.context) > 80 else ''}\n")

    results = compute_signals(args.passage, args.context)
    print_results(results)


if __name__ == "__main__":
    main()
