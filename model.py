"""
model.py — Transformer Architecture
DA6401 Assignment 3: "Attention Is All You Need"
"""

import math
import copy
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════
#  SCALED DOT-PRODUCT ATTENTION
# ══════════════════════════════════════════════════════════════════════

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask, float('-inf'))
    attn_w = F.softmax(scores, dim=-1)
    # Replace NaN (from all-masked rows) with 0
    attn_w = torch.nan_to_num(attn_w, nan=0.0)
    output = torch.matmul(attn_w, V)
    return output, attn_w


# ══════════════════════════════════════════════════════════════════════
#  MASK HELPERS
# ══════════════════════════════════════════════════════════════════════

def make_src_mask(src: torch.Tensor, pad_idx: int = 1) -> torch.Tensor:
    # [batch, 1, 1, src_len]  True = PAD
    return (src == pad_idx).unsqueeze(1).unsqueeze(2)


def make_tgt_mask(tgt: torch.Tensor, pad_idx: int = 1) -> torch.Tensor:
    tgt_len = tgt.size(1)
    # Padding mask [batch, 1, 1, tgt_len]
    pad_mask = (tgt == pad_idx).unsqueeze(1).unsqueeze(2)
    # Causal mask [1, 1, tgt_len, tgt_len]
    causal_mask = torch.triu(
        torch.ones(tgt_len, tgt_len, device=tgt.device), diagonal=1
    ).bool().unsqueeze(0).unsqueeze(0)
    # Combine: [batch, 1, tgt_len, tgt_len]
    return pad_mask | causal_mask


# ══════════════════════════════════════════════════════════════════════
#  MULTI-HEAD ATTENTION
# ══════════════════════════════════════════════════════════════════════

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        query: torch.Tensor,
        key:   torch.Tensor,
        value: torch.Tensor,
        mask:  Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch = query.size(0)

        def split_heads(x):
            x = x.view(batch, -1, self.num_heads, self.d_k)
            return x.transpose(1, 2)  # [batch, heads, seq, d_k]

        Q = split_heads(self.W_q(query))
        K = split_heads(self.W_k(key))
        V = split_heads(self.W_v(value))

        out, _ = scaled_dot_product_attention(Q, K, V, mask)
        out = out.transpose(1, 2).contiguous().view(batch, -1, self.d_model)
        return self.W_o(out)


# ══════════════════════════════════════════════════════════════════════
#  POSITIONAL ENCODING
# ══════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ══════════════════════════════════════════════════════════════════════
#  FEED-FORWARD NETWORK
# ══════════════════════════════════════════════════════════════════════

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# ══════════════════════════════════════════════════════════════════════
#  ENCODER LAYER
# ══════════════════════════════════════════════════════════════════════

class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn       = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1     = nn.LayerNorm(d_model)
        self.norm2     = nn.LayerNorm(d_model)
        self.dropout   = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        # Self-attention + Add & Norm (Post-LN)
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, src_mask)))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


# ══════════════════════════════════════════════════════════════════════
#  DECODER LAYER
# ══════════════════════════════════════════════════════════════════════

class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn  = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn        = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1      = nn.LayerNorm(d_model)
        self.norm2      = nn.LayerNorm(d_model)
        self.norm3      = nn.LayerNorm(d_model)
        self.dropout    = nn.Dropout(p=dropout)

    def forward(self, x, memory, src_mask, tgt_mask):
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, tgt_mask)))
        x = self.norm2(x + self.dropout(self.cross_attn(x, memory, memory, src_mask)))
        x = self.norm3(x + self.dropout(self.ffn(x)))
        return x


# ══════════════════════════════════════════════════════════════════════
#  ENCODER & DECODER STACKS
# ══════════════════════════════════════════════════════════════════════

class Encoder(nn.Module):
    def __init__(self, layer: EncoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm   = nn.LayerNorm(layer.norm1.normalized_shape)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, layer: DecoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm   = nn.LayerNorm(layer.norm1.normalized_shape)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


# ══════════════════════════════════════════════════════════════════════
#  FULL TRANSFORMER
# ══════════════════════════════════════════════════════════════════════

class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size: int = 8000,
        tgt_vocab_size: int = 8000,
        d_model:   int   = 256,
        N:         int   = 4,
        num_heads: int   = 8,
        d_ff:      int   = 1024,
        dropout:   float = 0.1,
        weights_gdrive_id: str = "1-HzSGF4_Fzk_ARqIfjTekl7W4gLEOmCs",   # Google Drive file ID for checkpoint
        device: str = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.device  = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # ── build architecture first with placeholder sizes ──────────────
        # Real vocab sizes are set after loading checkpoint/vocab
        self._src_vocab_size = src_vocab_size
        self._tgt_vocab_size = tgt_vocab_size

        enc_layer = EncoderLayer(d_model, num_heads, d_ff, dropout)
        dec_layer = DecoderLayer(d_model, num_heads, d_ff, dropout)

        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)
        self.src_pe    = PositionalEncoding(d_model, dropout)
        self.tgt_pe    = PositionalEncoding(d_model, dropout)
        self.encoder   = Encoder(enc_layer, N)
        self.decoder   = Decoder(dec_layer, N)
        self.fc_out    = nn.Linear(d_model, tgt_vocab_size)

        self._init_weights()

        # ── load tokenizers ──────────────────────────────────────────────
        self._load_tokenizers()

        # ── load vocab & weights from checkpoint ─────────────────────────
        if weights_gdrive_id is not None:
            self._download_and_load(weights_gdrive_id)
        # Note: do NOT auto-load checkpoint.pt here — training handles that

        self.to(self.device)

    # ------------------------------------------------------------------
    def _load_tokenizers(self):
        """Load spaCy tokenizers for German and English."""
        import spacy
        try:
            self._spacy_de = spacy.load("de_core_news_sm")
        except OSError:
            import subprocess
            subprocess.run(["python3", "-m", "spacy", "download", "de_core_news_sm"],
                           check=True)
            self._spacy_de = spacy.load("de_core_news_sm")

        try:
            self._spacy_en = spacy.load("en_core_web_sm")
        except OSError:
            import subprocess
            subprocess.run(["python3", "-m", "spacy", "download", "en_core_web_sm"],
                           check=True)
            self._spacy_en = spacy.load("en_core_web_sm")

    # ------------------------------------------------------------------
    def _download_and_load(self, gdrive_id: str):
        """Download checkpoint from Google Drive using gdown, then load."""
        import os
        try:
            import gdown
        except ImportError:
            import subprocess
            subprocess.run(["pip", "install", "gdown", "-q"], check=True)
            import gdown

        ckpt_path = "checkpoint_downloaded.pt"
        if not os.path.exists(ckpt_path):
            url = f"https://drive.google.com/uc?id={gdrive_id}"
            gdown.download(url, ckpt_path, quiet=False)

        self._load_local(ckpt_path)

    # ------------------------------------------------------------------
    def _load_local(self, path: str):
        """Load weights + vocab from a local checkpoint file."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)

        # Restore vocab if saved in checkpoint
        if "src_vocab" in ckpt:
            self._src_vocab = ckpt["src_vocab"]
        if "tgt_vocab" in ckpt:
            self._tgt_vocab = ckpt["tgt_vocab"]

        # Rebuild embeddings/output with correct vocab sizes if needed
        if "src_vocab" in ckpt and "tgt_vocab" in ckpt:
            sv = len(self._src_vocab.itos)
            tv = len(self._tgt_vocab.itos)
            if sv != self._src_vocab_size or tv != self._tgt_vocab_size:
                self.src_embed = nn.Embedding(sv, self.d_model)
                self.tgt_embed = nn.Embedding(tv, self.d_model)
                self.fc_out    = nn.Linear(self.d_model, tv)

        self.load_state_dict(ckpt["model_state_dict"])
        self.eval()

    # ------------------------------------------------------------------
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # ------------------------------------------------------------------
    def tokenize_de(self, text: str):
        return [tok.text.lower() for tok in self._spacy_de.tokenizer(text)]

    # ------------------------------------------------------------------
    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        x = self.src_pe(self.src_embed(src) * math.sqrt(self.d_model))
        return self.encoder(x, src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        x = self.tgt_pe(self.tgt_embed(tgt) * math.sqrt(self.d_model))
        x = self.decoder(x, memory, src_mask, tgt_mask)
        return self.fc_out(x)

    def forward(self, src, tgt, src_mask, tgt_mask):
        memory = self.encode(src, src_mask)
        return self.decode(memory, src_mask, tgt, tgt_mask)

    # ------------------------------------------------------------------
    def infer(self, german_sentence: str, max_len: int = 100) -> str:
        """
        End-to-end German → English translation.

        Args:
            german_sentence : Raw German string.
            max_len         : Maximum output tokens.

        Returns:
            Translated English string.
        """
        self.eval()

        SOS = self._tgt_vocab.stoi.get("<sos>", 2)
        EOS = self._tgt_vocab.stoi.get("<eos>", 3)
        PAD = self._tgt_vocab.stoi.get("<pad>", 1)
        UNK = self._tgt_vocab.stoi.get("<unk>", 0)

        # 1. Tokenize German input
        tokens = self.tokenize_de(german_sentence)
        src_ids = ([self._src_vocab.stoi.get("<sos>", 2)]
                   + [self._src_vocab.stoi.get(t, UNK) for t in tokens]
                   + [self._src_vocab.stoi.get("<eos>", 3)])

        src = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(self.device)
        src_mask = make_src_mask(src).to(self.device)

        # 2. Encode
        with torch.no_grad():
            memory = self.encode(src, src_mask)

            # 3. Autoregressive decoding
            ys = torch.tensor([[SOS]], dtype=torch.long, device=self.device)
            for _ in range(max_len):
                tgt_mask = make_tgt_mask(ys).to(self.device)
                logits   = self.decode(memory, src_mask, ys, tgt_mask)
                next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                ys = torch.cat([ys, next_tok], dim=1)
                if next_tok.item() == EOS:
                    break

        # 4. Detokenize — skip SOS/EOS/PAD
        pred_tokens = []
        for idx in ys[0].tolist():
            if idx == EOS:
                break
            if idx not in (SOS, PAD):
                pred_tokens.append(self._tgt_vocab.lookup_token(idx))

        return " ".join(pred_tokens)