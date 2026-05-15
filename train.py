"""
train.py — Training Pipeline, Inference & Evaluation
DA6401 Assignment 3: "Attention Is All You Need"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional

from model import Transformer, make_src_mask, make_tgt_mask


# ══════════════════════════════════════════════════════════════════════
#  LABEL SMOOTHING LOSS
# ══════════════════════════════════════════════════════════════════════

class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing as in "Attention Is All You Need".
    y_smooth = (1 - eps) * one_hot(y) + eps / (vocab_size - 1)
    """

    def __init__(self, vocab_size: int, pad_idx: int, smoothing: float = 0.1) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx    = pad_idx
        self.smoothing  = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        logits : [batch * tgt_len, vocab_size]
        target : [batch * tgt_len]
        """
        log_probs = F.log_softmax(logits, dim=-1)

        # Build smoothed target distribution
        with torch.no_grad():
            smooth_val = self.smoothing / (self.vocab_size - 2)  # exclude pad and true class
            dist = torch.full_like(log_probs, smooth_val)
            dist[:, self.pad_idx] = 0.0
            dist.scatter_(1, target.unsqueeze(1), self.confidence)

        # Mask out pad positions
        mask = (target != self.pad_idx)
        loss = -(dist * log_probs).sum(dim=-1)
        loss = loss[mask].mean()
        return loss


# ══════════════════════════════════════════════════════════════════════
#  TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════

def run_epoch(
    data_iter,
    model: Transformer,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler=None,
    epoch_num: int = 0,
    is_train: bool = True,
    device: str = "cpu",
) -> float:
    model.train() if is_train else model.eval()

    total_loss = 0.0
    total_tokens = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for src, tgt in data_iter:
            src = src.to(device)
            tgt = tgt.to(device)

            tgt_input  = tgt[:, :-1]   # decoder input (drop <eos>)
            tgt_output = tgt[:, 1:]    # gold output  (drop <sos>)

            src_mask = make_src_mask(src).to(device)
            tgt_mask = make_tgt_mask(tgt_input).to(device)

            logits = model(src, tgt_input, src_mask, tgt_mask)

            # Flatten for loss
            B, T, V = logits.shape
            loss = loss_fn(logits.reshape(B * T, V), tgt_output.reshape(B * T))

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            n_tokens = (tgt_output != 1).sum().item()
            total_loss   += loss.item() * n_tokens
            total_tokens += n_tokens

    return total_loss / max(total_tokens, 1)


# ══════════════════════════════════════════════════════════════════════
#  GREEDY DECODING
# ══════════════════════════════════════════════════════════════════════

def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    start_symbol: int,
    end_symbol: int,
    device: str = "cpu",
) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        memory = model.encode(src, src_mask)
        ys = torch.full((1, 1), start_symbol, dtype=torch.long, device=device)

        for _ in range(max_len - 1):
            tgt_mask = make_tgt_mask(ys).to(device)
            logits = model.decode(memory, src_mask, ys, tgt_mask)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            ys = torch.cat([ys, next_token], dim=1)
            if next_token.item() == end_symbol:
                break

    return ys


# ══════════════════════════════════════════════════════════════════════
#  BLEU EVALUATION
# ══════════════════════════════════════════════════════════════════════

def evaluate_bleu(
    model: Transformer,
    test_dataloader: DataLoader,
    tgt_vocab,
    device: str = "cpu",
    max_len: int = 100,
) -> float:
    from nltk.translate.bleu_score import corpus_bleu

    model.eval()
    hypotheses = []
    references = []

    SOS = tgt_vocab.stoi.get("<sos>", 2)
    EOS = tgt_vocab.stoi.get("<eos>", 3)
    PAD = tgt_vocab.stoi.get("<pad>", 1)

    with torch.no_grad():
        for src, tgt in test_dataloader:
            src = src.to(device)
            tgt = tgt.to(device)

            for i in range(src.size(0)):
                src_i    = src[i].unsqueeze(0)
                src_mask = make_src_mask(src_i).to(device)

                pred = greedy_decode(
                    model, src_i, src_mask, max_len,
                    start_symbol=SOS, end_symbol=EOS, device=device
                )

                # Convert predicted indices to tokens
                pred_tokens = []
                for idx in pred[0].tolist():
                    if idx == EOS:
                        break
                    if idx not in (SOS, PAD):
                        pred_tokens.append(tgt_vocab.lookup_token(idx))

                # Reference tokens (strip SOS/EOS/PAD)
                ref_tokens = []
                for idx in tgt[i].tolist():
                    if idx == EOS:
                        break
                    if idx not in (SOS, PAD):
                        ref_tokens.append(tgt_vocab.lookup_token(idx))

                hypotheses.append(pred_tokens)
                references.append([ref_tokens])

    bleu = corpus_bleu(references, hypotheses) * 100
    return bleu


# ══════════════════════════════════════════════════════════════════════
#  CHECKPOINT UTILITIES
# ══════════════════════════════════════════════════════════════════════

def save_checkpoint(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    path: str = "checkpoint.pt",
    src_vocab=None,
    tgt_vocab=None,
) -> None:
    torch.save({
        'epoch': epoch,
        'model_state_dict':     model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'src_vocab': src_vocab,
        'tgt_vocab': tgt_vocab,
        'model_config': {
            'src_vocab_size': model.src_embed.num_embeddings,
            'tgt_vocab_size': model.tgt_embed.num_embeddings,
            'd_model':   model.d_model,
            'N':         len(model.encoder.layers),
            'num_heads': model.encoder.layers[0].self_attn.num_heads,
            'd_ff':      model.encoder.layers[0].ffn.linear1.out_features,
            'dropout':   model.encoder.layers[0].dropout.p,
        }
    }, path)


def load_checkpoint(
    path: str,
    model: Transformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> int:
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint['epoch']


# ══════════════════════════════════════════════════════════════════════
#  EXPERIMENT ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def run_training_experiment() -> None:
    import wandb
    from torch.utils.data import DataLoader
    from dataset import build_datasets, collate_fn, PAD_IDX
    from lr_scheduler import NoamScheduler

  
    config = {
    "d_model":      256,
    "N":            4,
    "num_heads":    8,
    "d_ff":         1024,
    "dropout":      0.2,      # was 0.1, increase to reduce overfitting
    "warmup_steps": 4000,
    "batch_size":   128,
    "num_epochs":   40,
    "smoothing":    0.1,
}

    wandb.init(project="da6401-a3", config=config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build datasets
    train_ds, val_ds, test_ds, src_vocab, tgt_vocab = build_datasets()

    from functools import partial
    collate = partial(collate_fn, pad_idx=PAD_IDX)

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"],
                              shuffle=True,  collate_fn=collate)
    val_loader   = DataLoader(val_ds,   batch_size=config["batch_size"],
                              shuffle=False, collate_fn=collate)
    test_loader  = DataLoader(test_ds,  batch_size=config["batch_size"],
                              shuffle=False, collate_fn=collate)

    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=config["d_model"],
        N=config["N"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        dropout=config["dropout"],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9
    )
    scheduler = NoamScheduler(optimizer, d_model=config["d_model"],
                              warmup_steps=config["warmup_steps"])
    loss_fn   = LabelSmoothingLoss(len(tgt_vocab), PAD_IDX, config["smoothing"])

    best_val_loss = float('inf')
    for epoch in range(config["num_epochs"]):
        train_loss = run_epoch(train_loader, model, loss_fn, optimizer,
                               scheduler, epoch, is_train=True, device=device)
        val_loss   = run_epoch(val_loader, model, loss_fn, None,
                               None, epoch, is_train=False, device=device)

        wandb.log({"train_loss": train_loss, "val_loss": val_loss, "epoch": epoch})
        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, scheduler, epoch, "checkpoint.pt", src_vocab=src_vocab, tgt_vocab=tgt_vocab)

    bleu = evaluate_bleu(model, test_loader, tgt_vocab, device=device)
    wandb.log({"test_bleu": bleu})
    print(f"Test BLEU: {bleu:.2f}")
    wandb.finish()


if __name__ == "__main__":
    run_training_experiment()