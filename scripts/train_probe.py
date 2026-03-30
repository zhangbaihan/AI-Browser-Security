#!/usr/bin/env python3
"""
Train probes on extracted LLaMA activations to detect prompt injection.

DEFENSIVE AI SECURITY RESEARCH: Trains linear, attention-based, and combined
probes on residual stream activations and attention patterns extracted from
LLaMA 3.1 8B-Instruct. The goal is to identify which internal representations
best distinguish clean prompts from prompt-injected observations.

Usage:
    # Basic usage
    python scripts/train_probe.py --activations-dir activations/ --results-dir results/

    # Specify which probes to train
    python scripts/train_probe.py --activations-dir activations/ --probes linear attention combined

    # Adjust training
    python scripts/train_probe.py --activations-dir activations/ --linear-epochs 100 --combined-epochs 200

Requirements:
    pip install torch numpy scikit-learn matplotlib seaborn
    (No Modal dependency -- runs locally on CPU)
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# ============================================================================
# Data Loading
# ============================================================================


def load_activations(activations_dir: str) -> list[dict]:
    """Load all .pt activation files from a directory.

    Each file contains:
        id: str
        label: int (0=clean, 1=poisoned)
        residual_streams: tensor (32, 4096)
        attention_ratios: tensor (32, 32, 3)
        generated_text: str
        complied_with_injection: bool
    """
    samples = []
    pt_files = sorted(Path(activations_dir).glob("*.pt"))

    if not pt_files:
        print(f"ERROR: No .pt files found in {activations_dir}")
        sys.exit(1)

    print(f"Loading {len(pt_files)} activation files from {activations_dir}...")

    for fpath in pt_files:
        data = torch.load(fpath, map_location="cpu", weights_only=False)
        samples.append(data)

    labels = [s["label"] for s in samples]
    n_clean = sum(1 for l in labels if l == 0)
    n_poisoned = sum(1 for l in labels if l == 1)
    print(f"  Loaded {len(samples)} samples: {n_clean} clean, {n_poisoned} poisoned")

    if n_poisoned > 0:
        n_complied = sum(
            1 for s in samples
            if s["label"] == 1 and s.get("complied_with_injection", False)
        )
        print(
            f"  Compliance rate: {n_complied}/{n_poisoned} "
            f"({100*n_complied/n_poisoned:.1f}%) of poisoned prompts"
        )

    return samples


def prepare_tensors(samples: list[dict]):
    """Convert list of sample dicts into stacked tensors.

    Returns:
        residual_streams: (N, 32, 4096)
        attention_ratios: (N, 32, 32, 3)
        labels: (N,)
    """
    residual_streams = torch.stack([s["residual_streams"] for s in samples]).float()
    attention_ratios = torch.stack([s["attention_ratios"] for s in samples]).float()
    labels = torch.tensor([s["label"] for s in samples], dtype=torch.float32)
    return residual_streams, attention_ratios, labels


# ============================================================================
# Probe Architectures
# ============================================================================


class LinearProbe(nn.Module):
    """Per-layer linear probe on residual stream activations.

    For a given layer L, maps (4096,) -> (1,) via a linear layer + sigmoid.
    """

    def __init__(self, input_dim: int = 4096):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        # x: (batch, 4096)
        return torch.sigmoid(self.linear(x)).squeeze(-1)


class AttentionProbe(nn.Module):
    """Probe using attention ratio features and per-head entropy.

    Input features:
        - attention_ratios flattened: 32 * 32 * 3 = 3072
        - attention entropy per head: 32 * 32 = 1024
        Total: 4096
    """

    def __init__(self, input_dim: int = 4096):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        # x: (batch, 4096) -- pre-computed attention features
        return torch.sigmoid(self.linear(x)).squeeze(-1)


class CombinedProbe(nn.Module):
    """MLP probe combining residual stream + attention features.

    Input: best_layer_residual (4096) + attention_features (4096) +
           cosine_sim(user_residual, obs_residual) (1) = 8193
    """

    def __init__(self, input_dim: int = 8193):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        # x: (batch, 8193)
        return torch.sigmoid(self.mlp(x)).squeeze(-1)


# ============================================================================
# Feature Engineering
# ============================================================================


def compute_attention_entropy(attention_ratios: torch.Tensor) -> torch.Tensor:
    """Compute Shannon entropy of attention distribution per head.

    Args:
        attention_ratios: (N, 32, 32, 3) -- attention fractions to 3 segments

    Returns:
        entropy: (N, 32, 32) -- per-head entropy
    """
    # Clamp to avoid log(0)
    p = attention_ratios.clamp(min=1e-10)
    # Normalize to ensure valid probability distribution
    p = p / p.sum(dim=-1, keepdim=True).clamp(min=1e-10)
    entropy = -(p * torch.log2(p)).sum(dim=-1)  # (N, 32, 32)
    return entropy


def build_attention_features(attention_ratios: torch.Tensor) -> torch.Tensor:
    """Build the full attention feature vector.

    Concatenates flattened attention ratios (3072) and per-head entropy (1024).

    Args:
        attention_ratios: (N, 32, 32, 3)

    Returns:
        features: (N, 4096)
    """
    N = attention_ratios.shape[0]
    flat_ratios = attention_ratios.reshape(N, -1)  # (N, 3072)
    entropy = compute_attention_entropy(attention_ratios).reshape(N, -1)  # (N, 1024)
    return torch.cat([flat_ratios, entropy], dim=1)  # (N, 4096)


def build_combined_features(
    residual_streams: torch.Tensor,
    attention_ratios: torch.Tensor,
    best_layer: int,
) -> torch.Tensor:
    """Build features for the CombinedProbe.

    Concatenates:
        - Best-layer residual stream (4096)
        - Attention features (4096)
        - Cosine similarity between user and observation attention patterns (1)

    Args:
        residual_streams: (N, 32, 4096)
        attention_ratios: (N, 32, 32, 3)
        best_layer: Index of the best-performing layer from LinearProbe

    Returns:
        features: (N, 8193)
    """
    N = residual_streams.shape[0]

    best_residual = residual_streams[:, best_layer, :]  # (N, 4096)
    attn_features = build_attention_features(attention_ratios)  # (N, 4096)

    # Cosine similarity between attention to user segment vs observation segment
    # across all heads, as a scalar summary of attention shift
    user_attn = attention_ratios[:, :, :, 1].reshape(N, -1)   # (N, 1024)
    obs_attn = attention_ratios[:, :, :, 2].reshape(N, -1)    # (N, 1024)
    cos_sim = nn.functional.cosine_similarity(user_attn, obs_attn, dim=1)  # (N,)
    cos_sim = cos_sim.unsqueeze(1)  # (N, 1)

    return torch.cat([best_residual, attn_features, cos_sim], dim=1)  # (N, 8193)


# ============================================================================
# Training Loop
# ============================================================================


def train_probe(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    lr: float = 1e-3,
    epochs: int = 50,
    patience: int = 10,
    batch_size: int = 64,
    verbose: bool = True,
) -> dict:
    """Train a probe with early stopping on validation loss.

    Returns:
        dict with train/val metrics and the trained model state.
    """
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_ds = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / max(n_batches, 1)

        # --- Validate ---
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val)
            val_loss = criterion(val_preds, y_val).item()
            val_acc = ((val_preds > 0.5).float() == y_val).float().mean().item()

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # --- Early stopping ---
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if verbose and (epoch + 1) % 10 == 0:
            print(
                f"  Epoch {epoch+1:3d}/{epochs} | "
                f"Train loss: {avg_train_loss:.4f} | "
                f"Val loss: {val_loss:.4f} | "
                f"Val acc: {val_acc:.4f}"
            )

        if patience_counter >= patience:
            if verbose:
                print(f"  Early stopping at epoch {epoch+1}")
            break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final evaluation
    model.eval()
    with torch.no_grad():
        val_preds = model(X_val).numpy()

    y_val_np = y_val.numpy()
    y_pred_binary = (val_preds > 0.5).astype(float)

    metrics = {
        "accuracy": accuracy_score(y_val_np, y_pred_binary),
        "precision": precision_score(y_val_np, y_pred_binary, zero_division=0),
        "recall": recall_score(y_val_np, y_pred_binary, zero_division=0),
        "f1": f1_score(y_val_np, y_pred_binary, zero_division=0),
        "auroc": roc_auc_score(y_val_np, val_preds) if len(np.unique(y_val_np)) > 1 else 0.0,
        "val_preds": val_preds,
        "val_labels": y_val_np,
        "history": history,
    }

    return metrics


# ============================================================================
# Visualization
# ============================================================================


def plot_layer_accuracy_heatmap(layer_accuracies: list[float], results_dir: str):
    """Plot per-layer accuracy as a heatmap to show which layers encode task-drift."""
    fig, ax = plt.subplots(figsize=(14, 3))

    data = np.array(layer_accuracies).reshape(1, -1)
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=0.5, vmax=1.0)

    ax.set_yticks([])
    ax.set_xticks(range(32))
    ax.set_xticklabels([str(i) for i in range(32)], fontsize=8)
    ax.set_xlabel("Layer Index")
    ax.set_title("Linear Probe Accuracy per Layer (clean vs poisoned)")

    # Annotate cells
    for i in range(32):
        color = "white" if data[0, i] > 0.85 or data[0, i] < 0.6 else "black"
        ax.text(i, 0, f"{data[0, i]:.2f}", ha="center", va="center",
                fontsize=7, color=color)

    plt.colorbar(im, ax=ax, shrink=0.8, label="Accuracy")
    plt.tight_layout()

    fpath = os.path.join(results_dir, "layer_accuracy_heatmap.png")
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fpath}")


def plot_attention_comparison(
    attention_ratios: torch.Tensor,
    labels: torch.Tensor,
    results_dir: str,
):
    """Plot mean attention ratios (clean vs poisoned) across layers.

    Shows how much attention the last token pays to system, user, and
    observation segments for clean vs poisoned inputs.
    """
    clean_mask = labels == 0
    poisoned_mask = labels == 1

    segment_names = ["System", "User", "Observation"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    for seg_idx, seg_name in enumerate(segment_names):
        ax = axes[seg_idx]

        # Mean across samples and heads for each layer
        # attention_ratios shape: (N, 32, 32, 3)
        clean_data = attention_ratios[clean_mask, :, :, seg_idx].mean(dim=2)  # (N_clean, 32)
        poisoned_data = attention_ratios[poisoned_mask, :, :, seg_idx].mean(dim=2)  # (N_poison, 32)

        clean_mean = clean_data.mean(dim=0).numpy()
        poisoned_mean = poisoned_data.mean(dim=0).numpy()
        clean_std = clean_data.std(dim=0).numpy()
        poisoned_std = poisoned_data.std(dim=0).numpy()

        layers = np.arange(32)
        ax.plot(layers, clean_mean, "b-", label="Clean", linewidth=2)
        ax.fill_between(layers, clean_mean - clean_std, clean_mean + clean_std,
                        alpha=0.2, color="blue")
        ax.plot(layers, poisoned_mean, "r-", label="Poisoned", linewidth=2)
        ax.fill_between(layers, poisoned_mean - poisoned_std, poisoned_mean + poisoned_std,
                        alpha=0.2, color="red")

        ax.set_xlabel("Layer")
        ax.set_ylabel("Mean Attention Fraction" if seg_idx == 0 else "")
        ax.set_title(f"Attention to {seg_name} Segment")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle("Attention Distribution: Clean vs Poisoned", fontsize=14)
    plt.tight_layout()

    fpath = os.path.join(results_dir, "attention_comparison.png")
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fpath}")


def plot_roc_curves(probe_results: dict[str, dict], results_dir: str):
    """Plot ROC curves for each probe type."""
    fig, ax = plt.subplots(figsize=(8, 8))

    colors = {"linear": "blue", "attention": "orange", "combined": "green"}

    for probe_name, metrics in probe_results.items():
        if "val_preds" not in metrics or "val_labels" not in metrics:
            continue

        fpr, tpr, _ = roc_curve(metrics["val_labels"], metrics["val_preds"])
        auroc = metrics["auroc"]
        color = colors.get(probe_name, "gray")

        ax.plot(fpr, tpr, color=color, linewidth=2,
                label=f"{probe_name} (AUROC={auroc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves: Prompt Injection Detection Probes")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])

    plt.tight_layout()
    fpath = os.path.join(results_dir, "roc_curves.png")
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fpath}")


def plot_training_curves(probe_results: dict[str, dict], results_dir: str):
    """Plot training and validation loss curves for each probe."""
    n_probes = len(probe_results)
    fig, axes = plt.subplots(1, n_probes, figsize=(6 * n_probes, 4))
    if n_probes == 1:
        axes = [axes]

    for ax, (name, metrics) in zip(axes, probe_results.items()):
        history = metrics.get("history", {})
        if not history:
            continue

        epochs_range = range(1, len(history["train_loss"]) + 1)
        ax.plot(epochs_range, history["train_loss"], "b-", label="Train Loss")
        ax.plot(epochs_range, history["val_loss"], "r-", label="Val Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("BCE Loss")
        ax.set_title(f"{name} Probe")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle("Training Curves", fontsize=14)
    plt.tight_layout()

    fpath = os.path.join(results_dir, "training_curves.png")
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fpath}")


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Train probes on LLaMA activation data to detect prompt injection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--activations-dir", required=True,
        help="Directory containing .pt activation files",
    )
    parser.add_argument(
        "--results-dir", default="results",
        help="Directory to save plots and metrics",
    )
    parser.add_argument(
        "--probes", nargs="+", default=["linear", "attention", "combined"],
        choices=["linear", "attention", "combined"],
        help="Which probes to train",
    )
    parser.add_argument("--linear-epochs", type=int, default=50)
    parser.add_argument("--combined-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.results_dir, exist_ok=True)

    # --- Load data ---
    samples = load_activations(args.activations_dir)
    residual_streams, attention_ratios, labels = prepare_tensors(samples)
    N = len(labels)

    print(f"\nData shapes:")
    print(f"  residual_streams: {residual_streams.shape}")
    print(f"  attention_ratios: {attention_ratios.shape}")
    print(f"  labels: {labels.shape} (mean={labels.mean():.3f})")

    # --- Stratified train/test split ---
    indices = np.arange(N)
    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, random_state=args.seed,
        stratify=labels.numpy(),
    )

    print(f"\nSplit: {len(train_idx)} train, {len(test_idx)} test")

    probe_results = {}

    # =======================================================================
    # LINEAR PROBE (per-layer)
    # =======================================================================
    if "linear" in args.probes:
        print("\n" + "=" * 70)
        print("LINEAR PROBE (per-layer)")
        print("=" * 70)

        layer_accuracies = []
        best_layer = -1
        best_layer_acc = 0.0
        best_layer_metrics = None

        for layer in range(32):
            # Extract this layer's residual stream
            X = residual_streams[:, layer, :]  # (N, 4096)
            y = labels

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model = LinearProbe(input_dim=4096)
            metrics = train_probe(
                model, X_train, y_train, X_test, y_test,
                lr=1e-3, epochs=args.linear_epochs,
                patience=args.patience, batch_size=args.batch_size,
                verbose=False,
            )

            layer_accuracies.append(metrics["accuracy"])

            if metrics["accuracy"] > best_layer_acc:
                best_layer_acc = metrics["accuracy"]
                best_layer = layer
                best_layer_metrics = metrics

            print(
                f"  Layer {layer:2d}: "
                f"Acc={metrics['accuracy']:.4f} | "
                f"F1={metrics['f1']:.4f} | "
                f"AUROC={metrics['auroc']:.4f}"
            )

        print(f"\n  Best layer: {best_layer} (accuracy={best_layer_acc:.4f})")
        probe_results["linear"] = best_layer_metrics

        # Plot layer accuracy heatmap
        plot_layer_accuracy_heatmap(layer_accuracies, args.results_dir)

        # Save per-layer results
        layer_results = {
            "layer_accuracies": layer_accuracies,
            "best_layer": best_layer,
            "best_accuracy": best_layer_acc,
        }
        with open(os.path.join(args.results_dir, "linear_probe_results.json"), "w") as f:
            json.dump(layer_results, f, indent=2)

    # =======================================================================
    # ATTENTION PROBE
    # =======================================================================
    if "attention" in args.probes:
        print("\n" + "=" * 70)
        print("ATTENTION PROBE")
        print("=" * 70)

        X_attn = build_attention_features(attention_ratios)  # (N, 4096)
        X_train, X_test = X_attn[train_idx], X_attn[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        model = AttentionProbe(input_dim=4096)
        metrics = train_probe(
            model, X_train, y_train, X_test, y_test,
            lr=1e-3, epochs=args.linear_epochs,
            patience=args.patience, batch_size=args.batch_size,
            verbose=True,
        )

        print(f"\n  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1:        {metrics['f1']:.4f}")
        print(f"  AUROC:     {metrics['auroc']:.4f}")

        probe_results["attention"] = metrics

    # =======================================================================
    # COMBINED PROBE
    # =======================================================================
    if "combined" in args.probes:
        print("\n" + "=" * 70)
        print("COMBINED PROBE (MLP)")
        print("=" * 70)

        # Determine best layer (from linear probe, or default to layer 16)
        if "linear" in args.probes:
            bl = best_layer
            print(f"  Using best layer from linear probe: {bl}")
        else:
            bl = 16
            print(f"  Using default layer: {bl}")

        X_combined = build_combined_features(
            residual_streams, attention_ratios, best_layer=bl
        )  # (N, 8193)

        X_train, X_test = X_combined[train_idx], X_combined[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        model = CombinedProbe(input_dim=8193)
        metrics = train_probe(
            model, X_train, y_train, X_test, y_test,
            lr=1e-4, epochs=args.combined_epochs,
            patience=args.patience, batch_size=args.batch_size,
            verbose=True,
        )

        print(f"\n  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1:        {metrics['f1']:.4f}")
        print(f"  AUROC:     {metrics['auroc']:.4f}")

        probe_results["combined"] = metrics

    # =======================================================================
    # Plots
    # =======================================================================
    print("\n" + "=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)

    if len(probe_results) > 0:
        plot_roc_curves(probe_results, args.results_dir)
        plot_training_curves(probe_results, args.results_dir)

    if attention_ratios is not None:
        plot_attention_comparison(attention_ratios, labels, args.results_dir)

    # =======================================================================
    # Summary
    # =======================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    summary = {}
    for name, metrics in probe_results.items():
        row = {
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "auroc": metrics["auroc"],
        }
        summary[name] = row
        print(
            f"  {name:12s} | "
            f"Acc={row['accuracy']:.4f} | "
            f"P={row['precision']:.4f} | "
            f"R={row['recall']:.4f} | "
            f"F1={row['f1']:.4f} | "
            f"AUROC={row['auroc']:.4f}"
        )

    # Save summary
    summary_path = os.path.join(args.results_dir, "probe_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary saved to {summary_path}")
    print(f"  All plots saved to {args.results_dir}/")


if __name__ == "__main__":
    main()
