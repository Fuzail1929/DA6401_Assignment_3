"""
experiments.py — W&B Report Experiments
DA6401 Assignment 3

Runs all 5 experiments one by one:
  2.1 Noam vs Fixed LR  (train loss + val BLEU overlay)
  2.2 Scaling Factor ablation  (Q/K grad norms + train/val loss)
  2.3 Attention Rollout & Head Specialization  (beautiful per-head heatmaps)
  2.4 Sinusoidal PE vs Learned Embeddings  (val BLEU comparison)
  2.5 Label Smoothing vs Cross-Entropy  (prediction confidence)

Usage:
    python experiments.py --exp all        # run all
    python experiments.py --exp 2.1        # run specific one
    python experiments.py --exp 2.3        # attention viz only (no training needed)
"""

import argparse
import os
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from functools import partial
from torch.utils.data import DataLoader

# ── local imports ────────────────────────────────────────────────────
from model import (Transformer, EncoderLayer, DecoderLayer, Encoder, Decoder,
                   PositionalEncoding, PositionwiseFeedForward,
                   MultiHeadAttention, scaled_dot_product_attention,
                   make_src_mask, make_tgt_mask)
from lr_scheduler import NoamScheduler
from train import run_epoch, LabelSmoothingLoss, evaluate_bleu, save_checkpoint
from dataset import build_datasets, collate_fn, PAD_IDX


# ══════════════════════════════════════════════════════════════════════
#  SHARED CONFIG
# ══════════════════════════════════════════════════════════════════════

BASE_CONFIG = {
    "d_model":      256,
    "N":            4,
    "num_heads":    8,
    "d_ff":         1024,
    "dropout":      0.1,
    "warmup_steps": 4000,
    "batch_size":   128,
    "num_epochs":   15,
    "smoothing":    0.1,
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Plot style ───────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "font.size":        11,
    "axes.titlesize":   13,
    "axes.labelsize":   11,
})
COLORS = {
    "noam":        "#2196F3",
    "fixed":       "#F44336",
    "with_scale":  "#4CAF50",
    "no_scale":    "#FF9800",
    "sinusoidal":  "#9C27B0",
    "learned":     "#FF5722",
    "smoothing":   "#2196F3",
    "cross_entropy": "#F44336",
}


# ══════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════

def get_dataloaders(batch_size):
    train_ds, val_ds, test_ds, src_vocab, tgt_vocab = build_datasets()
    col = partial(collate_fn, pad_idx=PAD_IDX)
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  collate_fn=col)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, collate_fn=col)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, collate_fn=col)
    return train_loader, val_loader, test_loader, src_vocab, tgt_vocab


def build_model(src_vocab, tgt_vocab, config):
    return Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=config["d_model"],
        N=config["N"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        dropout=config["dropout"],
    ).to(DEVICE)


# ══════════════════════════════════════════════════════════════════════
#  2.1  NOAM vs FIXED LR
#  Deliverable: overlay TRAINING LOSS + VALIDATION BLEU curves
# ══════════════════════════════════════════════════════════════════════

def exp_2_1():
    print("\n=== Experiment 2.1: Noam vs Fixed LR ===")
    config = BASE_CONFIG.copy()
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = \
        get_dataloaders(config["batch_size"])

    all_train_losses = {}
    all_val_bleus    = {}
    all_lrs          = {}

    for mode in ["noam", "fixed"]:
        wandb.init(
            project="da6401-a3",
            name=f"2.1-{mode}-lr",
            config={**config, "lr_mode": mode},
            reinit=True,
        )

        model   = build_model(src_vocab, tgt_vocab, config)
        loss_fn = LabelSmoothingLoss(len(tgt_vocab), PAD_IDX, config["smoothing"])

        if mode == "noam":
            optimizer = torch.optim.Adam(model.parameters(),
                                         lr=1.0, betas=(0.9, 0.98), eps=1e-9)
            scheduler = NoamScheduler(optimizer, config["d_model"],
                                      config["warmup_steps"])
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            scheduler = None

        train_losses, val_accs, lrs = [], [], []

        for epoch in range(config["num_epochs"]):
            model.train()
            epoch_loss, n_batches = 0.0, 0

            for src, tgt in train_loader:
                src, tgt  = src.to(DEVICE), tgt.to(DEVICE)
                tgt_in    = tgt[:, :-1]; tgt_out = tgt[:, 1:]
                sm        = make_src_mask(src).to(DEVICE)
                tm        = make_tgt_mask(tgt_in).to(DEVICE)
                logits    = model(src, tgt_in, sm, tm)
                B, T, V   = logits.shape
                loss      = loss_fn(logits.reshape(B*T, V), tgt_out.reshape(B*T))
                optimizer.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if scheduler:
                    scheduler.step()
                lrs.append(optimizer.param_groups[0]["lr"])
                epoch_loss += loss.item(); n_batches += 1

            avg_train = epoch_loss / max(n_batches, 1)

            # Validation accuracy = token-level accuracy on val set
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for src, tgt in val_loader:
                    src, tgt = src.to(DEVICE), tgt.to(DEVICE)
                    tgt_in   = tgt[:, :-1]; tgt_out = tgt[:, 1:]
                    sm       = make_src_mask(src).to(DEVICE)
                    tm       = make_tgt_mask(tgt_in).to(DEVICE)
                    logits   = model(src, tgt_in, sm, tm)
                    preds    = logits.argmax(dim=-1)            # [B, T]
                    mask     = (tgt_out != PAD_IDX)
                    correct += (preds == tgt_out)[mask].sum().item()
                    total   += mask.sum().item()
            val_acc = correct / max(total, 1)

            train_losses.append(avg_train)
            val_accs.append(val_acc)

            # BLEU every epoch
            val_bleu = evaluate_bleu(model, val_loader, tgt_vocab, device=DEVICE)

            wandb.log({
                "train_loss":      avg_train,
                "val_accuracy":    val_acc,
                "val_bleu":        val_bleu,
                "lr":              optimizer.param_groups[0]["lr"],
                "epoch":           epoch,
            })
            print(f"  [2.1-{mode}] Epoch {epoch+1} | "
                  f"Train Loss {avg_train:.4f} | Val Acc {val_acc:.4f} | Val BLEU {val_bleu:.2f}")

        all_train_losses[mode] = train_losses
        all_val_bleus[mode]    = val_accs
        all_lrs[mode]          = lrs
        wandb.finish()

    # ── Overlay Plot ──────────────────────────────────────────────────
    epochs = range(1, config["num_epochs"] + 1)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Experiment 2.1: Noam Scheduler vs Fixed Learning Rate",
                 fontsize=14, fontweight="bold", y=1.02)

    for mode in ["noam", "fixed"]:
        label = "Noam Scheduler" if mode == "noam" else "Fixed LR (1e-4)"
        axes[0].plot(epochs, all_train_losses[mode],
                     label=label, color=COLORS[mode], linewidth=2)
        axes[1].plot(epochs, all_val_bleus[mode],
                     label=label, color=COLORS[mode], linewidth=2)

    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Training Loss")
    axes[0].set_title("Training Loss"); axes[0].legend()

    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Validation Accuracy (token-level)")
    axes[1].set_title("Validation Accuracy"); axes[1].legend()

    axes[2].plot(all_lrs["noam"],  label="Noam Scheduler",
                 color=COLORS["noam"], linewidth=1.5)
    axes[2].axhline(1e-4, color=COLORS["fixed"], linestyle="--",
                    label="Fixed LR=1e-4", linewidth=2)
    axes[2].set_xlabel("Step"); axes[2].set_ylabel("Learning Rate")
    axes[2].set_title("LR Schedule"); axes[2].legend()

    plt.tight_layout()
    plt.savefig("exp_2_1_overlay.png", dpi=150, bbox_inches="tight")
    print("  Saved exp_2_1_overlay.png")


# ══════════════════════════════════════════════════════════════════════
#  2.2  SCALING FACTOR ABLATION
#  Deliverable: Q/K grad norms first 1000 steps + train/val loss
# ══════════════════════════════════════════════════════════════════════

def scaled_dot_product_NO_SCALE(Q, K, V, mask=None):
    """Attention WITHOUT the 1/sqrt(dk) scaling."""
    scores = torch.matmul(Q, K.transpose(-2, -1))
    if mask is not None:
        scores = scores.masked_fill(mask, float('-inf'))
    attn_w = F.softmax(scores, dim=-1)
    attn_w = torch.nan_to_num(attn_w, nan=0.0)
    return torch.matmul(attn_w, V), attn_w


class MultiHeadAttentionNoScale(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model; self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch = query.size(0)
        def split(x):
            return x.view(batch, -1, self.num_heads, self.d_k).transpose(1, 2)
        Q = split(self.W_q(query)); K = split(self.W_k(key)); V = split(self.W_v(value))
        out, _ = scaled_dot_product_NO_SCALE(Q, K, V, mask)
        return self.W_o(out.transpose(1, 2).contiguous().view(batch, -1, self.d_model))


class EncoderLayerNoScale(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttentionNoScale(d_model, num_heads, dropout)
        self.ffn   = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model); self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, x, mask):
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, mask)))
        return self.norm2(x + self.dropout(self.ffn(x)))


class DecoderLayerNoScale(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn  = MultiHeadAttentionNoScale(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttentionNoScale(d_model, num_heads, dropout)
        self.ffn   = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model); self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, x, memory, src_mask, tgt_mask):
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, tgt_mask)))
        x = self.norm2(x + self.dropout(self.cross_attn(x, memory, memory, src_mask)))
        return self.norm3(x + self.dropout(self.ffn(x)))


def build_noscale_model(src_vocab, tgt_vocab, config):
    d, N, h, ff, dr = (config["d_model"], config["N"], config["num_heads"],
                       config["d_ff"], config["dropout"])
    model = Transformer(len(src_vocab), len(tgt_vocab), d, N, h, ff, dr)
    model.encoder = Encoder(EncoderLayerNoScale(d, h, ff, dr), N)
    model.decoder = Decoder(DecoderLayerNoScale(d, h, ff, dr), N)
    return model.to(DEVICE)


def exp_2_2():
    print("\n=== Experiment 2.2: Scaling Factor Ablation ===")
    config = BASE_CONFIG.copy()
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = \
        get_dataloaders(config["batch_size"])

    grad_norms       = {"with_scale": [], "no_scale": []}
    all_train_losses = {"with_scale": [], "no_scale": []}
    all_val_losses   = {"with_scale": [], "no_scale": []}

    for mode in ["with_scale", "no_scale"]:
        wandb.init(
            project="da6401-a3",
            name=f"2.2-{mode}",
            config={**config, "scaling": mode},
            reinit=True,
        )

        model = build_model(src_vocab, tgt_vocab, config) if mode == "with_scale" \
                else build_noscale_model(src_vocab, tgt_vocab, config)

        loss_fn   = LabelSmoothingLoss(len(tgt_vocab), PAD_IDX, config["smoothing"])
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=1.0, betas=(0.9, 0.98), eps=1e-9)
        scheduler = NoamScheduler(optimizer, config["d_model"], config["warmup_steps"])

        # Phase 1: log Q/K grad norms for first 1000 steps
        # One epoch = ~226 batches (29000/128), so loop multiple epochs
        print(f"  [2.2-{mode}] Logging grad norms for 1000 steps...")
        step = 0
        model.train()
        while step < 1000:
            for src, tgt in train_loader:
                if step >= 1000:
                    break
                src, tgt = src.to(DEVICE), tgt.to(DEVICE)
                tgt_in = tgt[:, :-1]; tgt_out = tgt[:, 1:]
                sm = make_src_mask(src).to(DEVICE)
                tm = make_tgt_mask(tgt_in).to(DEVICE)
                logits = model(src, tgt_in, sm, tm)
                B, T, V = logits.shape
                loss = loss_fn(logits.reshape(B*T, V), tgt_out.reshape(B*T))
                optimizer.zero_grad(); loss.backward()

                q_grad = model.encoder.layers[0].self_attn.W_q.weight.grad
                k_grad = model.encoder.layers[0].self_attn.W_k.weight.grad
                if q_grad is not None and k_grad is not None:
                    q_norm = q_grad.norm().item()
                    k_norm = k_grad.norm().item()
                    avg    = (q_norm + k_norm) / 2
                    grad_norms[mode].append(avg)
                    wandb.log({"qk_grad_norm": avg,
                               "q_grad_norm":  q_norm,
                               "k_grad_norm":  k_norm,
                               "step":         step})

                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step(); scheduler.step()
                step += 1
        print(f"  [2.2-{mode}] Completed {step} steps for grad norm logging")

        # Phase 2: full training
        for epoch in range(config["num_epochs"]):
            tr = run_epoch(train_loader, model, loss_fn, optimizer,
                           scheduler, epoch, is_train=True, device=DEVICE)
            vl = run_epoch(val_loader, model, loss_fn, None,
                           None, epoch, is_train=False, device=DEVICE)
            all_train_losses[mode].append(tr)
            all_val_losses[mode].append(vl)
            val_bleu = evaluate_bleu(model, val_loader, tgt_vocab, device=DEVICE)
            wandb.log({"train_loss": tr, "val_loss": vl,
                       "val_bleu": val_bleu, "epoch": epoch})
            print(f"  [2.2-{mode}] Epoch {epoch+1} | Train {tr:.4f} | "
                  f"Val {vl:.4f} | BLEU {val_bleu:.2f}")

        wandb.finish()

    # ── Plots ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Experiment 2.2: Scaling Factor 1/√dk Ablation",
                 fontsize=14, fontweight="bold", y=1.02)

    for mode, clr in [("with_scale", COLORS["with_scale"]),
                      ("no_scale",   COLORS["no_scale"])]:
        label    = "With 1/√dk" if mode == "with_scale" else "Without 1/√dk"
        norms    = grad_norms[mode]
        window   = 20
        smoothed = np.convolve(norms, np.ones(window)/window, mode='valid')
        axes[0].plot(norms,    color=clr, alpha=0.2)
        axes[0].plot(range(window-1, len(norms)), smoothed,
                     color=clr, linewidth=2, label=f"{label} (smoothed)")

    axes[0].set_xlabel("Step"); axes[0].set_ylabel("Q/K Gradient Norm")
    axes[0].set_title("Q & K Gradient Norms (First 1000 Steps)")
    axes[0].legend()

    epochs = range(1, config["num_epochs"] + 1)
    for mode, clr in [("with_scale", COLORS["with_scale"]),
                      ("no_scale",   COLORS["no_scale"])]:
        label = "With 1/√dk" if mode == "with_scale" else "Without 1/√dk"
        axes[1].plot(epochs, all_train_losses[mode], color=clr, linewidth=2, label=label)
        axes[2].plot(epochs, all_val_losses[mode],   color=clr, linewidth=2, label=label)

    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Train Loss")
    axes[1].set_title("Training Loss"); axes[1].legend()
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Val Loss")
    axes[2].set_title("Validation Loss"); axes[2].legend()

    plt.tight_layout()
    plt.savefig("exp_2_2_scaling_ablation.png", dpi=150, bbox_inches="tight")
    print("  Saved exp_2_2_scaling_ablation.png")


# ══════════════════════════════════════════════════════════════════════
#  2.3  ATTENTION ROLLOUT & HEAD SPECIALIZATION
#  Deliverable: beautiful per-head heatmaps + entropy analysis
# ══════════════════════════════════════════════════════════════════════

class MultiHeadAttentionWithWeights(nn.Module):
    """MHA that captures attention weights after each forward pass."""
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model; self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.last_attn_weights = None

    def forward(self, query, key, value, mask=None):
        batch = query.size(0)
        def split(x):
            return x.view(batch, -1, self.num_heads, self.d_k).transpose(1, 2)
        Q = split(self.W_q(query)); K = split(self.W_k(key)); V = split(self.W_v(value))
        out, attn_w = scaled_dot_product_attention(Q, K, V, mask)
        self.last_attn_weights = attn_w.detach().cpu()
        return self.W_o(out.transpose(1, 2).contiguous().view(batch, -1, self.d_model))


def _clean_tokens(tokens):
    """Replace special tokens with readable symbols."""
    replacements = {"<sos>": "⟨S⟩", "<eos>": "⟨E⟩",
                    "<pad>": "⟨P⟩", "<unk>": "⟨?⟩"}
    return [replacements.get(t, t) for t in tokens]


def exp_2_3():
    print("\n=== Experiment 2.3: Attention Head Visualization ===")
    config = BASE_CONFIG.copy()
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = \
        get_dataloaders(config["batch_size"])

    # ── Load checkpoint ───────────────────────────────────────────────
    if os.path.exists("checkpoint.pt"):
        ckpt = torch.load("checkpoint.pt", map_location=DEVICE, weights_only=False)
        ckpt_cfg = ckpt.get("model_config", {})
        for k in ["N", "d_ff", "d_model", "num_heads"]:
            if k in ckpt_cfg:
                config[k] = ckpt_cfg[k]
        model = build_model(src_vocab, tgt_vocab, config)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"  Loaded checkpoint.pt  (N={config['N']}, d_ff={config['d_ff']})")
    else:
        print("  No checkpoint.pt found — training 5 epochs from scratch...")
        model   = build_model(src_vocab, tgt_vocab, config)
        loss_fn = LabelSmoothingLoss(len(tgt_vocab), PAD_IDX, config["smoothing"])
        opt     = torch.optim.Adam(model.parameters(),
                                   lr=1.0, betas=(0.9, 0.98), eps=1e-9)
        sch     = NoamScheduler(opt, config["d_model"], config["warmup_steps"])
        for ep in range(5):
            run_epoch(train_loader, model, loss_fn, opt, sch,
                      ep, is_train=True, device=DEVICE)

    # ── Hook attention weights in last encoder layer ───────────────────
    last_enc = model.encoder.layers[-1]
    mha_vis  = MultiHeadAttentionWithWeights(
        config["d_model"], config["num_heads"], config["dropout"]
    ).to(DEVICE)
    mha_vis.load_state_dict(last_enc.self_attn.state_dict())
    last_enc.self_attn = mha_vis

    # ── Pick a good sample (8-18 real tokens, not too padded) ──────────
    model.eval()
    chosen_src, chosen_tokens = None, None
    for src_batch, _ in val_loader:
        for i in range(src_batch.size(0)):
            row     = src_batch[i]
            non_pad = (row != PAD_IDX).sum().item()
            if 8 <= non_pad <= 18:
                chosen_src    = src_batch[i:i+1].to(DEVICE)
                chosen_tokens = [src_vocab.itos[idx]
                                 for idx in row.tolist()][:non_pad]
                break
        if chosen_src is not None:
            break

    if chosen_src is None:
        src_batch, _ = next(iter(val_loader))
        chosen_src    = src_batch[:1].to(DEVICE)
        chosen_tokens = [src_vocab.itos[idx] for idx in src_batch[0].tolist()]

    src_mask = make_src_mask(chosen_src).to(DEVICE)
    with torch.no_grad():
        model.encode(chosen_src, src_mask)

    attn_weights = mha_vis.last_attn_weights  # [1, heads, seq, seq]
    num_heads    = attn_weights.shape[1]
    seq_len      = attn_weights.shape[2]

    display_tokens = _clean_tokens(chosen_tokens[:seq_len])
    n_tok = len(display_tokens)

    # ── Per-head entropy ──────────────────────────────────────────────
    entropies = []
    for h in range(num_heads):
        w      = attn_weights[0, h, :n_tok, :n_tok].numpy()
        w_safe = np.clip(w, 1e-9, 1.0)
        ent    = -(w_safe * np.log(w_safe)).sum(axis=-1).mean()
        entropies.append(ent)

    # ── Main heatmap figure ───────────────────────────────────────────
    ncols = 4
    nrows = math.ceil(num_heads / ncols)
    fig   = plt.figure(figsize=(ncols * 5, nrows * 5 + 1.5))
    fig.patch.set_facecolor("#0F1117")

    gs_outer = gridspec.GridSpec(nrows + 1, 1,
                                 height_ratios=[0.06] + [1]*nrows,
                                 hspace=0.4)
    ax_title = fig.add_subplot(gs_outer[0])
    ax_title.axis("off")
    ax_title.text(
        0.5, 0.5,
        "Experiment 2.3 — Last Encoder Layer: Self-Attention per Head",
        ha="center", va="center", fontsize=16,
        fontweight="bold", color="white",
        transform=ax_title.transAxes,
    )

    for row in range(nrows):
        gs_inner = gridspec.GridSpecFromSubplotSpec(
            1, ncols, subplot_spec=gs_outer[row + 1], wspace=0.4)
        for col in range(ncols):
            h  = row * ncols + col
            ax = fig.add_subplot(gs_inner[col])
            ax.set_facecolor("#1A1A2E")

            if h >= num_heads:
                ax.axis("off"); continue

            w   = attn_weights[0, h, :n_tok, :n_tok].numpy()
            ent = entropies[h]

            # Classify head behaviour
            diag     = np.diag(w).mean()
            upper    = np.triu(w, k=1).sum() / max(n_tok*(n_tok-1)/2, 1)
            lower    = np.tril(w, k=-1).sum() / max(n_tok*(n_tok-1)/2, 1)

            if ent < 0.8:
                behaviour, b_color = "Focused / Local",     "#4CAF50"
            elif diag > 0.4:
                behaviour, b_color = "Self-Attending",      "#2196F3"
            elif upper > lower * 1.5:
                behaviour, b_color = "Forward Attending",   "#FF9800"
            elif ent > 2.0:
                behaviour, b_color = "Diffuse (Global)",    "#9C27B0"
            else:
                behaviour, b_color = "Mixed",               "#F44336"

            im = ax.imshow(w, cmap="magma", vmin=0, vmax=w.max(),
                           aspect="auto", interpolation="nearest")

            fontsize = max(5, min(9, 80 // n_tok))
            ax.set_xticks(range(n_tok))
            ax.set_xticklabels(display_tokens, rotation=45, ha="right",
                               fontsize=fontsize, color="white")
            ax.set_yticks(range(n_tok))
            ax.set_yticklabels(display_tokens, fontsize=fontsize, color="white")
            ax.tick_params(colors="white")

            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(colors="white", labelsize=7)

            ax.set_title(
                f"Head {h+1}  |  H={ent:.2f}\n{behaviour}",
                fontsize=10, fontweight="bold", color=b_color, pad=6,
            )
            for spine in ax.spines.values():
                spine.set_edgecolor(b_color); spine.set_linewidth(2)

    plt.savefig("exp_2_3_attention_heads.png", dpi=150,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    print("  Saved exp_2_3_attention_heads.png")

    # ── Entropy bar chart ─────────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    fig2.patch.set_facecolor("#0F1117")
    ax2.set_facecolor("#1A1A2E")
    bar_colors = ["#4CAF50" if e < 0.8 else "#FF9800" if e < 1.5 else "#F44336"
                  for e in entropies]
    ax2.bar(range(1, num_heads+1), entropies, color=bar_colors, width=0.6)
    ax2.axhline(np.mean(entropies), color="white", linestyle="--",
                linewidth=1.5, label=f"Mean H = {np.mean(entropies):.2f}")
    ax2.set_xlabel("Head", color="white")
    ax2.set_ylabel("Entropy (nats)", color="white")
    ax2.set_title("Per-Head Attention Entropy  (lower = more specialised)",
                  color="white", fontweight="bold")
    ax2.tick_params(colors="white")
    ax2.legend(facecolor="#1A1A2E", labelcolor="white")
    for spine in ax2.spines.values():
        spine.set_edgecolor("#444")
    plt.tight_layout()
    plt.savefig("exp_2_3_entropy.png", dpi=150,
                bbox_inches="tight", facecolor=fig2.get_facecolor())
    print("  Saved exp_2_3_entropy.png")

    # ── Log everything to W&B ─────────────────────────────────────────
    wandb.init(project="da6401-a3", name="2.3-attention-heads",
               config=config, reinit=True)
    wandb.log({
        "attention_heatmaps": wandb.Image("exp_2_3_attention_heads.png"),
        "head_entropy_chart": wandb.Image("exp_2_3_entropy.png"),
        **{f"head_{h+1}_entropy": entropies[h] for h in range(num_heads)},
    })
    wandb.finish()


# ══════════════════════════════════════════════════════════════════════
#  2.4  SINUSOIDAL PE vs LEARNED EMBEDDINGS
#  Deliverable: val BLEU comparison + training curves
# ══════════════════════════════════════════════════════════════════════

class LearnedPositionalEmbedding(nn.Module):
    """Learned positional embedding via torch.nn.Embedding."""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Embedding(max_len, d_model)
        nn.init.normal_(self.pe.weight, std=0.02)

    def forward(self, x):
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        return self.dropout(x + self.pe(positions))


def build_learned_pe_model(src_vocab, tgt_vocab, config):
    model = build_model(src_vocab, tgt_vocab, config)
    d, dr = config["d_model"], config["dropout"]
    model.src_pe = LearnedPositionalEmbedding(d, dr).to(DEVICE)
    model.tgt_pe = LearnedPositionalEmbedding(d, dr).to(DEVICE)
    return model


def exp_2_4():
    print("\n=== Experiment 2.4: Sinusoidal PE vs Learned Embeddings ===")
    config = BASE_CONFIG.copy()
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = \
        get_dataloaders(config["batch_size"])

    bleu_results     = {}
    all_train_losses = {}
    all_val_losses   = {}

    for mode in ["sinusoidal", "learned"]:
        wandb.init(
            project="da6401-a3",
            name=f"2.4-pe-{mode}",
            config={**config, "pe_type": mode},
            reinit=True,
        )

        model = build_model(src_vocab, tgt_vocab, config) if mode == "sinusoidal" \
                else build_learned_pe_model(src_vocab, tgt_vocab, config)

        loss_fn   = LabelSmoothingLoss(len(tgt_vocab), PAD_IDX, config["smoothing"])
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=1.0, betas=(0.9, 0.98), eps=1e-9)
        scheduler = NoamScheduler(optimizer, config["d_model"], config["warmup_steps"])

        t_losses, v_losses = [], []
        for epoch in range(config["num_epochs"]):
            tr = run_epoch(train_loader, model, loss_fn, optimizer,
                           scheduler, epoch, is_train=True, device=DEVICE)
            vl = run_epoch(val_loader, model, loss_fn, None,
                           None, epoch, is_train=False, device=DEVICE)
            t_losses.append(tr); v_losses.append(vl)
            val_bleu = evaluate_bleu(model, val_loader, tgt_vocab, device=DEVICE)
            wandb.log({"train_loss": tr, "val_loss": vl,
                       "val_bleu": val_bleu, "epoch": epoch})
            print(f"  [2.4-{mode}] Epoch {epoch+1} | Train {tr:.4f} | "
                  f"Val {vl:.4f} | BLEU {val_bleu:.2f}")

        test_bleu = evaluate_bleu(model, test_loader, tgt_vocab, device=DEVICE)
        bleu_results[mode]     = test_bleu
        all_train_losses[mode] = t_losses
        all_val_losses[mode]   = v_losses
        wandb.log({"test_bleu": test_bleu})
        print(f"  [2.4-{mode}] Test BLEU: {test_bleu:.2f}")
        wandb.finish()

    # ── Plots ─────────────────────────────────────────────────────────
    epochs = range(1, config["num_epochs"] + 1)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Experiment 2.4: Sinusoidal PE vs Learned Positional Embeddings",
                 fontsize=14, fontweight="bold", y=1.02)

    for mode in ["sinusoidal", "learned"]:
        label = "Sinusoidal PE" if mode == "sinusoidal" else "Learned Embedding"
        axes[0].plot(epochs, all_train_losses[mode],
                     color=COLORS[mode], linewidth=2, label=label)
        axes[1].plot(epochs, all_val_losses[mode],
                     color=COLORS[mode], linewidth=2, label=label)

    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Train Loss")
    axes[0].set_title("Training Loss"); axes[0].legend()
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Val Loss")
    axes[1].set_title("Validation Loss"); axes[1].legend()

    keys  = list(bleu_results.keys())
    vals  = [bleu_results[k] for k in keys]
    clrs  = [COLORS[k] for k in keys]
    bars  = axes[2].bar(["Sinusoidal PE", "Learned Embed"],
                        vals, color=clrs, width=0.4)
    for bar, v in zip(bars, vals):
        axes[2].text(bar.get_x() + bar.get_width()/2, v + 0.3,
                     f"{v:.2f}", ha="center", fontweight="bold")
    axes[2].set_ylabel("BLEU Score"); axes[2].set_ylim(0, max(vals)*1.15)
    axes[2].set_title("Test BLEU Score Comparison")

    plt.tight_layout()
    plt.savefig("exp_2_4_pe_comparison.png", dpi=150, bbox_inches="tight")
    print("  Saved exp_2_4_pe_comparison.png")


# ══════════════════════════════════════════════════════════════════════
#  2.5  LABEL SMOOTHING vs CROSS-ENTROPY
#  Deliverable: prediction confidence (softmax prob of CORRECT token)
# ══════════════════════════════════════════════════════════════════════

def exp_2_5():
    print("\n=== Experiment 2.5: Label Smoothing vs Cross-Entropy ===")
    config = BASE_CONFIG.copy()
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = \
        get_dataloaders(config["batch_size"])

    all_train_losses = {}
    all_val_losses   = {}
    all_confidences  = {}
    bleu_results     = {}

    for smoothing in [0.1, 0.0]:
        mode  = "smoothing_0.1" if smoothing > 0 else "cross_entropy"
        label = "Label Smoothing ε=0.1" if smoothing > 0 else "Cross-Entropy ε=0.0"

        wandb.init(
            project="da6401-a3",
            name=f"2.5-{mode}",
            config={**config, "smoothing": smoothing},
            reinit=True,
        )

        model     = build_model(src_vocab, tgt_vocab, config)
        loss_fn   = LabelSmoothingLoss(len(tgt_vocab), PAD_IDX, smoothing)
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=1.0, betas=(0.9, 0.98), eps=1e-9)
        scheduler = NoamScheduler(optimizer, config["d_model"], config["warmup_steps"])

        t_losses, v_losses, confidences = [], [], []

        for epoch in range(config["num_epochs"]):
            model.train()
            epoch_loss, total_conf, total_tok, n_batches = 0.0, 0.0, 0, 0

            for src, tgt in train_loader:
                src, tgt = src.to(DEVICE), tgt.to(DEVICE)
                tgt_in   = tgt[:, :-1]; tgt_out = tgt[:, 1:]
                sm       = make_src_mask(src).to(DEVICE)
                tm       = make_tgt_mask(tgt_in).to(DEVICE)
                logits   = model(src, tgt_in, sm, tm)
                B, T, V  = logits.shape
                loss     = loss_fn(logits.reshape(B*T, V), tgt_out.reshape(B*T))
                optimizer.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step(); scheduler.step()

                # Confidence = softmax prob of the CORRECT token
                with torch.no_grad():
                    probs         = torch.softmax(logits, dim=-1)
                    tgt_flat      = tgt_out.reshape(-1)
                    probs_flat    = probs.reshape(-1, V)
                    correct_probs = probs_flat.gather(
                        1, tgt_flat.unsqueeze(1)).squeeze(1)
                    non_pad       = (tgt_flat != PAD_IDX)
                    total_conf   += correct_probs[non_pad].sum().item()
                    total_tok    += non_pad.sum().item()

                epoch_loss += loss.item(); n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            avg_conf = total_conf / max(total_tok, 1)
            vl       = run_epoch(val_loader, model, loss_fn, None,
                                 None, epoch, is_train=False, device=DEVICE)

            t_losses.append(avg_loss); v_losses.append(vl)
            confidences.append(avg_conf)

            val_bleu = evaluate_bleu(model, val_loader, tgt_vocab, device=DEVICE)
            wandb.log({
                "train_loss":            avg_loss,
                "val_loss":              vl,
                "val_bleu":              val_bleu,
                "prediction_confidence": avg_conf,
                "epoch":                 epoch,
            })
            print(f"  [2.5-{mode}] Epoch {epoch+1} | "
                  f"Train {avg_loss:.4f} | Val {vl:.4f} | "
                  f"BLEU {val_bleu:.2f} | Conf {avg_conf:.4f}")

        test_bleu = evaluate_bleu(model, test_loader, tgt_vocab, device=DEVICE)
        bleu_results[mode]     = test_bleu
        all_train_losses[mode] = t_losses
        all_val_losses[mode]   = v_losses
        all_confidences[mode]  = confidences
        wandb.log({"test_bleu": test_bleu})
        print(f"  [2.5-{mode}] Test BLEU: {test_bleu:.2f}")
        wandb.finish()

    # ── Plots ─────────────────────────────────────────────────────────
    epochs = range(1, config["num_epochs"] + 1)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Experiment 2.5: Label Smoothing vs Cross-Entropy",
                 fontsize=14, fontweight="bold", y=1.02)

    mode_labels = {
        "smoothing_0.1": ("Label Smoothing ε=0.1", COLORS["smoothing"]),
        "cross_entropy":  ("Cross-Entropy ε=0.0",  COLORS["cross_entropy"]),
    }

    for mode, (lbl, clr) in mode_labels.items():
        axes[0].plot(epochs, all_train_losses[mode], color=clr, linewidth=2, label=lbl)
        axes[1].plot(epochs, all_confidences[mode],  color=clr, linewidth=2, label=lbl)

    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Training Loss")
    axes[0].set_title("Training Loss"); axes[0].legend()

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Softmax Prob of Correct Token")
    axes[1].set_title("Prediction Confidence\n(prob of correct token)")
    axes[1].legend()

    modes  = list(bleu_results.keys())
    vals   = [bleu_results[m] for m in modes]
    labels = [mode_labels[m][0] for m in modes]
    clrs   = [mode_labels[m][1] for m in modes]
    bars   = axes[2].bar(labels, vals, color=clrs, width=0.4)
    for bar, v in zip(bars, vals):
        axes[2].text(bar.get_x() + bar.get_width()/2, v + 0.3,
                     f"{v:.2f}", ha="center", fontweight="bold")
    axes[2].set_ylabel("Test BLEU Score"); axes[2].set_ylim(0, max(vals)*1.15)
    axes[2].set_title("Test BLEU Score")

    plt.tight_layout()
    plt.savefig("exp_2_5_label_smoothing.png", dpi=150, bbox_inches="tight")
    print("  Saved exp_2_5_label_smoothing.png")


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, default="all",
                        choices=["all", "2.1", "2.2", "2.3", "2.4", "2.5"],
                        help="Which experiment to run")
    args = parser.parse_args()

    exp_map = {
        "2.1": exp_2_1,
        "2.2": exp_2_2,
        "2.3": exp_2_3,
        "2.4": exp_2_4,
        "2.5": exp_2_5,
    }

    if args.exp == "all":
        for name, fn in exp_map.items():
            print(f"\n{'='*60}")
            print(f"Running Experiment {name}")
            print('='*60)
            fn()
    else:
        exp_map[args.exp]()

    print("\n✓ Done! Check your W&B dashboard for all plots and metrics.")


if __name__ == "__main__":
    main()