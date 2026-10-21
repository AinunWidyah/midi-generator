"""model.py — Mini Music Transformer (decoder-only), dibangun dari nol.

Pemilik modul: Ketua (Prompt Engineer & LLM Navigator).
Kontrak: lihat docs/KONTRAK_MODUL.md bagian `src/model.py`.

Status pengerjaan:
    [x] Tahap 1 — CausalSelfAttentionHead (satu head)
    [ ] Tahap 2 — MultiHeadAttention
    [ ] Tahap 3 — FeedForward
    [ ] Tahap 4 — TransformerBlock
    [ ] Tahap 5 — MiniMusicTransformer
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttentionHead(nn.Module):
    """Satu head self-attention dengan causal mask.

    Setiap token membuat tiga vektor:
        Query (Q) — "aku sedang mencari informasi apa?"
        Key   (K) — "aku berisi informasi apa?"
        Value (V) — "informasi yang aku berikan jika dipilih"

    Skor perhatian = Q · K^T / sqrt(head_size). Causal mask membuat token
    ke-i hanya boleh memperhatikan token 0..i (tidak boleh "mengintip" nada
    yang belum dimainkan), sehingga model bisa dipakai untuk prediksi
    token berikutnya.
    """

    def __init__(self, d_model: int, head_size: int, block_size: int, dropout: float = 0.1):
        super().__init__()
        self.head_size = head_size
        # Proyeksi linear tanpa bias: (C) -> (head_size)
        self.query = nn.Linear(d_model, head_size, bias=False)
        self.key = nn.Linear(d_model, head_size, bias=False)
        self.value = nn.Linear(d_model, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

        # Matriks segitiga bawah berisi 1: posisi yang BOLEH dilihat.
        # register_buffer -> ikut pindah ke GPU dan tersimpan, tapi bukan parameter yang dilatih.
        mask = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer("causal_mask", mask)  # (block_size, block_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C) -> out: (B, T, head_size)"""
        B, T, C = x.shape

        q = self.query(x)  # (B, T, hs)
        k = self.key(x)    # (B, T, hs)
        v = self.value(x)  # (B, T, hs)

        # Skor kemiripan setiap pasangan token, diskalakan agar softmax tidak terlalu tajam
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_size)  # (B, T, T)

        # Tutup posisi masa depan dengan -inf supaya bobotnya menjadi 0 setelah softmax
        mask = self.causal_mask[:T, :T]                          # (T, T)
        scores = scores.masked_fill(mask == 0, float("-inf"))    # (B, T, T)

        weights = F.softmax(scores, dim=-1)  # (B, T, T), tiap baris berjumlah 1
        weights = self.dropout(weights)

        out = weights @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


# ---------------------------------------------------------------------------
# Tes cepat — jalankan dengan:  python src/model.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(42)
    B, T, C, HS, BLOCK = 2, 8, 256, 64, 256

    head = CausalSelfAttentionHead(d_model=C, head_size=HS, block_size=BLOCK, dropout=0.0)
    head.eval()
    x = torch.randn(B, T, C)

    # Tes 1: shape output
    out = head(x)
    print(f"[Tes 1] input {tuple(x.shape)} -> output {tuple(out.shape)}")
    assert out.shape == (B, T, HS), "Shape output salah!"

    # Tes 2: causal — mengubah token terakhir tidak boleh mengubah output token sebelumnya
    x2 = x.clone()
    x2[:, -1, :] = torch.randn(B, C)
    out2 = head(x2)
    past_same = torch.allclose(out[:, :-1], out2[:, :-1], atol=1e-6)
    last_diff = not torch.allclose(out[:, -1], out2[:, -1], atol=1e-6)
    print(f"[Tes 2] token masa lalu tidak terpengaruh masa depan: {past_same}")
    print(f"[Tes 2] token terakhir ikut berubah: {last_diff}")
    assert past_same and last_diff, "Causal mask tidak bekerja!"

    # Tes 3: lihat pola bobot attention (harus segitiga bawah)
    with torch.no_grad():
        q, k = head.query(x[:1, :5]), head.key(x[:1, :5])
        s = (q @ k.transpose(-2, -1) / math.sqrt(HS)).masked_fill(
            head.causal_mask[:5, :5] == 0, float("-inf"))
        w = F.softmax(s, dim=-1)[0]
    print("[Tes 3] bobot attention 5 token pertama (atas kanan harus 0):")
    print(w.round(decimals=2))

    print("\nSemua tes Tahap 1 LULUS ✅")
