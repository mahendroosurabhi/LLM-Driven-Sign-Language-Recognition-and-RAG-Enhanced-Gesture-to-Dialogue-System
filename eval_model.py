"""
Evaluate trained sign models on a test set and save graphs + tables.

Usage:
  python eval_model.py --models best_model_aug.pt --data test_data
  python eval_model.py --models model_a.pt model_b.pt --data test_data      # also makes a comparison chart
  python eval_model.py --probs cv_trim_aug_rich_attn.npz                    # from saved probabilities (e.g. CV output)

Test data layout:  test_data/<class_name>/<video>.npy   (each file a raw (T, 225) array)
Or, if all files are in one folder and a CSV says which are train/val/test:
  python eval_model.py --models best_model_aug.pt --split-csv splits.csv --root videos_npy \\
         --path-col filename --label-col sign --split-col split --split test

Outputs (in eval_out/<model name>/):
  confusion_matrix.png     where the model confuses classes
  per_class_metrics.png    precision / recall / F1 for every class
  top_confusions.png       the most frequent wrong (true -> predicted) pairs
  confidence_hist.png      confidence of correct vs wrong predictions
  coverage_accuracy.png    accuracy vs share of predictions accepted, as the gate changes
  topk_accuracy.png        top-1 / top-3 / top-5 accuracy
  per_class_metrics.csv    the same per-class numbers as a table
  summary.txt              overall numbers
"""
import argparse
import csv
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

N_FRAMES = 32


# ----------------------------------------------------------------------------
# Getting probabilities from a checkpoint
# ----------------------------------------------------------------------------
def resample(frames, n=N_FRAMES):
    idx = np.linspace(0, len(frames) - 1, n).round().astype(int)
    return np.stack([frames[i] for i in idx])


def normalize(seq):
    p = seq.reshape(len(seq), 75, 3).copy()
    for t in range(len(p)):
        l, r = p[t, 11], p[t, 12]
        if not (l.any() and r.any()):
            continue
        center = (l + r) / 2
        scale = np.linalg.norm(l[:2] - r[:2]) + 1e-6
        found = p[t].any(axis=1)
        p[t, found] = (p[t, found] - center) / scale
    return p.reshape(len(p), 225)


def load_test_dir(data_dir):
    seqs, labels = [], []
    for path in sorted(glob.glob(os.path.join(data_dir, "*", "*.npy"))):
        seqs.append(np.load(path))
        labels.append(os.path.basename(os.path.dirname(path)))
    if not seqs:
        raise SystemExit(f"No .npy files found under {data_dir}/<class>/")
    return seqs, labels


def load_split_csv(csv_path, split, path_col, label_col, split_col, root):
    """Read your split CSV and return only the rows belonging to `split` (e.g. "test").
    Files are looked up relative to --root (default: the CSV's folder). If the CSV lists
    videos (.mp4 etc.) but the keypoints are saved as .npy, the extension is swapped."""
    base = root or os.path.dirname(os.path.abspath(csv_path))
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit("CSV is empty.")
    cols = list(rows[0].keys())
    for c in (path_col, label_col) + ((split_col,) if split_col else ()):
        if c not in cols:
            raise SystemExit(f"Column '{c}' not in CSV. Columns found: {cols}")

    if split_col:
        counts = {}
        for r in rows:
            counts[r[split_col]] = counts.get(r[split_col], 0) + 1
        print(f"split counts in CSV: {counts}")
        rows = [r for r in rows if r[split_col].strip().lower() == split.lower()]
        if not rows:
            raise SystemExit(f"No rows with {split_col} == '{split}'.")

    def find(p):
        stem = os.path.splitext(p)[0]
        for cand in (p, stem + ".npy", os.path.join(base, p), os.path.join(base, stem + ".npy"),
                     os.path.join(base, os.path.basename(stem) + ".npy")):
            if os.path.isfile(cand) and cand.endswith(".npy"):
                return cand
        return None

    seqs, labels, missing = [], [], 0
    for r in rows:
        p = find(r[path_col])
        if p is None:
            missing += 1
            continue
        seqs.append(np.load(p))
        labels.append(r[label_col])
    if missing:
        print(f"  warning: {missing} listed files not found as .npy (check --root / --path-col)")
    if not seqs:
        raise SystemExit("No keypoint files could be loaded from the CSV.")
    return seqs, labels


def model_probs(model_path, seqs, labels, hidden):
    """Runs your SignGRU checkpoint. If you trained with a different preprocessing
    or model class, change this function to match."""
    import torch
    import torch.nn as nn

    class SignGRU(nn.Module):
        def __init__(self, n_classes, in_dim=225, hidden=128):
            super().__init__()
            self.gru = nn.GRU(in_dim, hidden, batch_first=True, bidirectional=True)
            self.drop = nn.Dropout(0.45)
            self.fc = nn.Linear(hidden * 2, n_classes)

        def forward(self, x):
            out, _ = self.gru(x)
            return self.fc(self.drop(out.mean(dim=1)))

    ckpt = torch.load(model_path, map_location="cpu")
    classes = list(ckpt["labels"])
    model = SignGRU(len(classes), hidden=hidden)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    keep = [i for i, l in enumerate(labels) if l in classes]
    if len(keep) < len(labels):
        print(f"  warning: skipped {len(labels) - len(keep)} videos whose class is not in the checkpoint")
    y = np.array([classes.index(labels[i]) for i in keep])
    X = np.stack([normalize(resample(np.asarray(seqs[i], dtype=np.float32))) for i in keep])
    with torch.no_grad():
        probs = torch.softmax(model(torch.tensor(X, dtype=torch.float32)), dim=1).numpy()
    return probs, y, classes


def load_probs_file(path):
    d = np.load(path, allow_pickle=True)
    probs = d["probs"] if "probs" in d else d["oof"]
    return probs, d["y"], [str(c) for c in d["classes"]]


# ----------------------------------------------------------------------------
# Evaluation + plots
# ----------------------------------------------------------------------------
def topk_acc(probs, y, k):
    k = min(k, probs.shape[1])
    top = np.argsort(-probs, axis=1)[:, :k]
    return float((top == y[:, None]).any(axis=1).mean())


def evaluate(probs, y, classes, out_dir, gate=0.5):
    os.makedirs(out_dir, exist_ok=True)
    n = len(classes)
    pred, conf = probs.argmax(1), probs.max(1)
    correct = pred == y

    prec, rec, f1, sup = precision_recall_fscore_support(
        y, pred, labels=range(n), zero_division=0)
    cm = confusion_matrix(y, pred, labels=range(n))

    m = {
        "samples": len(y),
        "top1": float(correct.mean()),
        "top3": topk_acc(probs, y, 3),
        "top5": topk_acc(probs, y, 5),
        "macro_f1": float(f1[sup > 0].mean()),
        "gate": gate,
        "accepted": float((conf >= gate).mean()),
        "gated_acc": float(correct[conf >= gate].mean()) if (conf >= gate).any() else float("nan"),
    }

    # --- table ---
    with open(os.path.join(out_dir, "per_class_metrics.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "precision", "recall", "f1", "support"])
        for i, c in enumerate(classes):
            w.writerow([c, f"{prec[i]:.3f}", f"{rec[i]:.3f}", f"{f1[i]:.3f}", int(sup[i])])
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        for k, v in m.items():
            f.write(f"{k}: {v:.4f}\n" if isinstance(v, float) else f"{k}: {v}\n")

    # --- 1. confusion matrix (row-normalised colour, raw counts as text) ---
    rows = cm.sum(1, keepdims=True)
    cmn = np.divide(cm, rows, out=np.zeros_like(cm, dtype=float), where=rows > 0)
    size = max(8, n * 0.33)
    fig, ax = plt.subplots(figsize=(size + 2, size))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    fs = 7 if n > 30 else 9
    ax.set_xticklabels(classes, rotation=90, fontsize=fs)
    ax.set_yticklabels(classes, fontsize=fs)
    for i in range(n):
        for j in range(n):
            if cm[i, j] > 0:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=fs - 1,
                        color="white" if cmn[i, j] > 0.5 else "black")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix (colour = share of true class, numbers = counts)\n"
                 f"accuracy {m['top1']:.1%}")
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=150); plt.close(fig)

    # --- 2. per-class precision / recall / F1 ---
    order = np.argsort(f1)
    fig, ax = plt.subplots(figsize=(10, max(6, n * 0.28)))
    h = 0.27
    pos = np.arange(n)
    ax.barh(pos - h, prec[order], h, label="precision", color="#4c78a8")
    ax.barh(pos, rec[order], h, label="recall", color="#f58518")
    ax.barh(pos + h, f1[order], h, label="F1", color="#54a24b")
    ax.set_yticks(pos)
    ax.set_yticklabels([f"{classes[i]} (n={int(sup[i])})" for i in order], fontsize=8)
    ax.set_xlim(0, 1.05); ax.axvline(m["top1"], color="grey", ls="--", lw=1)
    ax.set_xlabel("score (dashed line = overall accuracy)")
    ax.set_title("Per-class metrics, worst F1 at the bottom")
    ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "per_class_metrics.png"), dpi=150); plt.close(fig)

    # --- 3. most frequent confusions ---
    off = [(cm[i, j], classes[i], classes[j]) for i in range(n) for j in range(n) if i != j and cm[i, j] > 0]
    off = sorted(off, reverse=True)[:15]
    if off:
        fig, ax = plt.subplots(figsize=(9, max(3, len(off) * 0.4)))
        labels = [f"{t} -> {p}" for _, t, p in off][::-1]
        ax.barh(labels, [c for c, _, _ in off][::-1], color="#e45756")
        ax.set_xlabel("number of videos"); ax.set_title("Most frequent confusions (true -> predicted)")
        fig.tight_layout(); fig.savefig(os.path.join(out_dir, "top_confusions.png"), dpi=150); plt.close(fig)

    # --- 4. confidence histogram ---
    bins = np.linspace(0, 1, 21)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(conf[correct], bins=bins, alpha=0.7, label=f"correct ({correct.sum()})", color="#54a24b")
    ax.hist(conf[~correct], bins=bins, alpha=0.7, label=f"wrong ({(~correct).sum()})", color="#e45756")
    ax.axvline(gate, color="black", ls="--", label=f"gate = {gate}")
    ax.set_xlabel("model confidence (max probability)"); ax.set_ylabel("videos")
    ax.set_title("Confidence of correct vs wrong predictions")
    ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "confidence_hist.png"), dpi=150); plt.close(fig)

    # --- 5. accuracy vs coverage as the gate changes ---
    ths = np.linspace(0, 0.95, 40)
    cov = np.array([(conf >= t).mean() for t in ths])
    acc = np.array([correct[conf >= t].mean() if (conf >= t).any() else np.nan for t in ths])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ths, acc, label="accuracy of accepted predictions", color="#4c78a8", lw=2)
    ax.plot(ths, cov, label="share of predictions accepted", color="#f58518", lw=2)
    ax.axvline(gate, color="black", ls="--", lw=1)
    ax.scatter([gate], [m["gated_acc"]], color="#4c78a8", zorder=5)
    ax.scatter([gate], [m["accepted"]], color="#f58518", zorder=5)
    ax.set_xlabel("confidence gate"); ax.set_ylim(0, 1.02)
    ax.set_title("Raising the gate: more accurate, but fewer predictions accepted")
    ax.legend(loc="center left"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "coverage_accuracy.png"), dpi=150); plt.close(fig)

    # --- 6. top-k accuracy ---
    fig, ax = plt.subplots(figsize=(5, 4))
    vals = [m["top1"], m["top3"], m["top5"]]
    bars = ax.bar(["top-1", "top-3", "top-5"], vals, color=["#4c78a8", "#72b7b2", "#54a24b"])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.1%}", ha="center")
    ax.set_ylim(0, 1.1); ax.set_title("Is the right sign in the top-k guesses?")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "topk_accuracy.png"), dpi=150); plt.close(fig)

    return m


def compare_plot(results, out_path):
    names = list(results)
    keys = [("top1", "accuracy"), ("top3", "top-3"), ("macro_f1", "macro F1"), ("gated_acc", "gated accuracy")]
    fig, ax = plt.subplots(figsize=(max(7, len(names) * 2.2), 5))
    w = 0.8 / len(keys)
    for k, (key, label) in enumerate(keys):
        vals = [results[n][key] for n in names]
        bars = ax.bar(np.arange(len(names)) + k * w, vals, w, label=label)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(np.arange(len(names)) + w * (len(keys) - 1) / 2)
    ax.set_xticklabels(names); ax.set_ylim(0, 1.1)
    ax.set_title("Model comparison"); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=[], help="checkpoint .pt files")
    ap.add_argument("--data", default="test_data", help="test folder: <class>/<video>.npy")
    ap.add_argument("--split-csv", default=None,
                    help="your CSV with file names, labels and train/val/test split; use instead of --data")
    ap.add_argument("--split", default="test", help="which split value to evaluate (default: test)")
    ap.add_argument("--path-col", default="path", help="CSV column with the file name/path")
    ap.add_argument("--label-col", default="label", help="CSV column with the class name")
    ap.add_argument("--split-col", default="split", help="CSV column with train/val/test")
    ap.add_argument("--root", default=None, help="folder holding the .npy files (default: CSV's folder)")
    ap.add_argument("--probs", nargs="*", default=[], help=".npz files with probs (or oof), y, classes")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--gate", type=float, default=0.5)
    ap.add_argument("--out", default="eval_out")
    args = ap.parse_args()
    if not args.models and not args.probs:
        raise SystemExit("Give --models and --data, or --probs.")

    results = {}
    if args.models:
        if args.split_csv:
            seqs, labels = load_split_csv(args.split_csv, args.split, args.path_col,
                                          args.label_col, args.split_col, args.root)
        else:
            seqs, labels = load_test_dir(args.data)
        print(f"{len(seqs)} test videos")
    for mp in args.models:
        name = os.path.splitext(os.path.basename(mp))[0]
        print(f"evaluating {name}")
        probs, y, classes = model_probs(mp, seqs, labels, args.hidden)
        results[name] = evaluate(probs, y, classes, os.path.join(args.out, name), args.gate)
    for pp in args.probs:
        name = os.path.splitext(os.path.basename(pp))[0]
        print(f"evaluating {name}")
        probs, y, classes = load_probs_file(pp)
        results[name] = evaluate(probs, y, classes, os.path.join(args.out, name), args.gate)

    for name, m in results.items():
        print(f"\n{name}: acc {m['top1']:.3f} | top-3 {m['top3']:.3f} | macro F1 {m['macro_f1']:.3f} | "
              f"gate {m['gate']}: accepts {m['accepted']:.1%}, acc {m['gated_acc']:.3f}")
    if len(results) > 1:
        compare_plot(results, os.path.join(args.out, "comparison.png"))
        print(f"\ncomparison chart: {os.path.join(args.out, 'comparison.png')}")
    print(f"graphs saved under {args.out}/")


if __name__ == "__main__":
    main()