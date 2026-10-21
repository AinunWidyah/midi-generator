"""model.py — Mini Music Transformer (decoder-only), dibangun dari nol.

Pemilik modul: Ketua (Prompt Engineer & LLM Navigator).
Kontrak: lihat docs/KONTRAK_MODUL.md bagian `src/model.py`.

Status pengerjaan:
    [x] Tahap 1 — CausalSelfAttentionHead (satu head)
    [x] Tahap 2 — MultiHeadAttention
    [x] Tahap 3 — FeedForward
    [x] Tahap 4 — TransformerBlock
    [x] Tahap 5 — MiniMusicTransformer
    [x] Tahap 6 — Uji model lengkap (shape, jumlah parameter, causal)
    [x] Tahap 7 — Uji overfit 1 batch

Cara menguji (dari folder utama proyek):
    python src/model.py --tahap 1      # uji satu tahap
    python src/model.py                # uji semua tahap berurutan
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # dipanggil sebagai paket: from src.model import ...
    from .config import ModelConfig, PAD_ID
except ImportError:  # dijalankan langsung: python src/model.py
    from config import ModelConfig, PAD_ID


# ===========================================================================
# Tahap 1 — Satu head Causal Self-Attention
# ===========================================================================
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


# ===========================================================================
# Tahap 2 — Multi-Head Attention
# ===========================================================================
class MultiHeadAttention(nn.Module):
    """Beberapa head attention yang berjalan paralel lalu digabung.

    Setiap head bisa belajar "memperhatikan" hal berbeda, misalnya satu head
    fokus ke nada sebelumnya (melodi), head lain fokus ke pola ritme beberapa
    ketukan ke belakang. Dimensi d_model dibagi rata ke n_head head:
        head_size = d_model // n_head   (256 // 4 = 64)
    Hasil semua head digabung (concat) kembali menjadi d_model, lalu dilewatkan
    ke satu lapisan linear (proj) agar informasi antar-head bisa bercampur.
    """

    def __init__(self, d_model: int, n_head: int, block_size: int, dropout: float = 0.1):
        super().__init__()
        if d_model % n_head != 0:
            raise ValueError(f"d_model ({d_model}) harus habis dibagi n_head ({n_head})")
        head_size = d_model // n_head
        self.heads = nn.ModuleList(
            [CausalSelfAttentionHead(d_model, head_size, block_size, dropout) for _ in range(n_head)]
        )
        self.proj = nn.Linear(d_model, d_model)  # (C) -> (C)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C) -> out: (B, T, C)"""
        out = torch.cat([head(x) for head in self.heads], dim=-1)  # n_head x (B, T, hs) -> (B, T, C)
        out = self.dropout(self.proj(out))                          # (B, T, C)
        return out


# ===========================================================================
# Tahap 3 — Feed-Forward Network
# ===========================================================================
class FeedForward(nn.Module):
    """MLP dua lapis yang diterapkan ke setiap token secara terpisah.

    Attention = token "berdiskusi" dengan token lain.
    FeedForward = setiap token "berpikir sendiri" mengolah hasil diskusi itu.
    Dimensi diperlebar 4x (256 -> 1024) lalu dikembalikan (1024 -> 256),
    dengan aktivasi GELU di tengah (non-linear, standar pada GPT).
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),  # (B, T, C) -> (B, T, 4C)
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),  # (B, T, 4C) -> (B, T, C)
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C) -> out: (B, T, C)"""
        return self.net(x)


# ===========================================================================
# Tahap 4 — Transformer Block
# ===========================================================================
class TransformerBlock(nn.Module):
    """Satu blok Transformer: Attention + FeedForward, masing-masing dengan
    LayerNorm dan residual connection (gaya "pre-norm" seperti GPT-2).

        x = x + Attention(LayerNorm(x))
        x = x + FeedForward(LayerNorm(x))

    - Residual (x + ...): informasi asli tetap mengalir, gradien tidak hilang
      walau blok ditumpuk banyak.
    - LayerNorm: menstabilkan skala angka di setiap token agar training stabil.
    """

    def __init__(self, d_model: int, n_head: int, block_size: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C) -> out: (B, T, C)"""
        x = x + self.attn(self.ln1(x))  # (B, T, C)
        x = x + self.ff(self.ln2(x))    # (B, T, C)
        return x


# ===========================================================================
# Tahap 5 — Model lengkap
# ===========================================================================
class MiniMusicTransformer(nn.Module):
    """Transformer decoder-only untuk memprediksi token musik berikutnya.

    Alur:
        idx (B, T) token ID
          -> token embedding     (B, T, C)   "nada apa ini?"
          +  positional embedding (T, C)     "di posisi ke berapa?"
          -> n_layer x TransformerBlock (B, T, C)
          -> LayerNorm akhir
          -> lm_head             (B, T, V)   skor untuk setiap kemungkinan token berikutnya

    Sesuai kontrak: model HANYA mengembalikan logits; loss dihitung di train.py.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)   # (V) -> (C)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.d_model)     # (block_size) -> (C)
        self.dropout = nn.Dropout(cfg.dropout)
        self.blocks = nn.Sequential(
            *[TransformerBlock(cfg.d_model, cfg.n_head, cfg.block_size, cfg.dropout)
              for _ in range(cfg.n_layer)]
        )
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)  # (C) -> (V)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """Inisialisasi bobot kecil (std 0.02) seperti GPT-2 agar awal training stabil."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """idx: LongTensor (B, T) -> logits: FloatTensor (B, T, V)"""
        B, T = idx.shape
        if T > self.cfg.block_size:
            raise ValueError(f"Panjang sekuens {T} melebihi block_size {self.cfg.block_size}")

        positions = torch.arange(T, device=idx.device)              # (T,)
        x = self.token_emb(idx) + self.pos_emb(positions)            # (B, T, C) + (T, C) -> (B, T, C)
        x = self.dropout(x)
        x = self.blocks(x)                                           # (B, T, C)
        x = self.ln_f(x)                                             # (B, T, C)
        logits = self.lm_head(x)                                     # (B, T, V)
        return logits

    def num_parameters(self) -> int:
        """Jumlah parameter yang bisa dilatih."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ===========================================================================
# Tes per tahap — jalankan dari folder utama:  python src/model.py --tahap N
# ===========================================================================
def _check(cond: bool, pesan: str) -> None:
    if not cond:
        raise AssertionError(pesan)


def tes_tahap_1() -> None:
    print("=== Tahap 1: CausalSelfAttentionHead ===")
    torch.manual_seed(42)
    B, T, C, HS, BLOCK = 2, 8, 256, 64, 256
    head = CausalSelfAttentionHead(d_model=C, head_size=HS, block_size=BLOCK, dropout=0.0).eval()
    x = torch.randn(B, T, C)

    out = head(x)
    print(f"[1a] input {tuple(x.shape)} -> output {tuple(out.shape)}")
    _check(out.shape == (B, T, HS), "Shape output salah!")

    x2 = x.clone()
    x2[:, -1, :] = torch.randn(B, C)
    out2 = head(x2)
    past_same = torch.allclose(out[:, :-1], out2[:, :-1], atol=1e-6)
    last_diff = not torch.allclose(out[:, -1], out2[:, -1], atol=1e-6)
    print(f"[1b] token masa lalu tidak terpengaruh masa depan: {past_same}")
    print(f"[1b] token terakhir ikut berubah: {last_diff}")
    _check(past_same and last_diff, "Causal mask tidak bekerja!")

    with torch.no_grad():
        q, k = head.query(x[:1, :5]), head.key(x[:1, :5])
        s = (q @ k.transpose(-2, -1) / math.sqrt(HS)).masked_fill(
            head.causal_mask[:5, :5] == 0, float("-inf"))
        w = F.softmax(s, dim=-1)[0]
    print("[1c] bobot attention 5 token pertama (atas kanan harus 0):")
    print(w.round(decimals=2))
    print("Tahap 1 LULUS ✅\n")


def tes_tahap_2() -> None:
    print("=== Tahap 2: MultiHeadAttention ===")
    torch.manual_seed(42)
    B, T, C, NH, BLOCK = 2, 8, 256, 4, 256
    mha = MultiHeadAttention(d_model=C, n_head=NH, block_size=BLOCK, dropout=0.0).eval()
    x = torch.randn(B, T, C)
    out = mha(x)
    print(f"[2a] jumlah head: {len(mha.heads)}, head_size tiap head: {C // NH}")
    print(f"[2b] input {tuple(x.shape)} -> output {tuple(out.shape)}")
    _check(out.shape == (B, T, C), "Shape output MHA harus sama dengan input!")

    try:
        MultiHeadAttention(d_model=250, n_head=4, block_size=BLOCK)
        _check(False, "Seharusnya error karena 250 tidak habis dibagi 4")
    except ValueError as e:
        print(f"[2c] validasi d_model % n_head bekerja: {e}")
    print("Tahap 2 LULUS ✅\n")


def tes_tahap_3() -> None:
    print("=== Tahap 3: FeedForward ===")
    torch.manual_seed(42)
    B, T, C = 2, 8, 256
    ff = FeedForward(d_model=C, dropout=0.0).eval()
    x = torch.randn(B, T, C)
    out = ff(x)
    hidden = ff.net[0].out_features
    print(f"[3a] dimensi tersembunyi: {C} -> {hidden} -> {C}")
    print(f"[3b] input {tuple(x.shape)} -> output {tuple(out.shape)}")
    _check(out.shape == (B, T, C) and hidden == 4 * C, "Shape FeedForward salah!")

    # Token diproses terpisah: mengubah token terakhir tidak memengaruhi token lain
    x2 = x.clone()
    x2[:, -1, :] = torch.randn(B, C)
    independen = torch.allclose(ff(x2)[:, :-1], out[:, :-1], atol=1e-6)
    print(f"[3c] setiap token diproses terpisah: {independen}")
    _check(independen, "FeedForward seharusnya per-token!")
    print("Tahap 3 LULUS ✅\n")


def tes_tahap_4() -> None:
    print("=== Tahap 4: TransformerBlock ===")
    torch.manual_seed(42)
    B, T, C, NH, BLOCK = 2, 8, 256, 4, 256
    block = TransformerBlock(d_model=C, n_head=NH, block_size=BLOCK, dropout=0.0).eval()
    x = torch.randn(B, T, C)
    out = block(x)
    print(f"[4a] input {tuple(x.shape)} -> output {tuple(out.shape)}")
    _check(out.shape == (B, T, C), "Shape output block salah!")

    beda = (out - x).abs().mean().item()
    print(f"[4b] rata-rata perubahan dari residual: {beda:.4f} (harus > 0, artinya blok bekerja)")
    _check(beda > 0, "Blok tidak mengubah input sama sekali!")

    x2 = x.clone()
    x2[:, -1, :] = torch.randn(B, C)
    causal = torch.allclose(block(x2)[:, :-1], out[:, :-1], atol=1e-5)
    print(f"[4c] causal tetap terjaga di dalam blok: {causal}")
    _check(causal, "Causal bocor di TransformerBlock!")
    print("Tahap 4 LULUS ✅\n")


def tes_tahap_5() -> None:
    print("=== Tahap 5: MiniMusicTransformer ===")
    torch.manual_seed(42)
    cfg = ModelConfig()
    model = MiniMusicTransformer(cfg).eval()
    B, T = 2, 16
    idx = torch.randint(0, cfg.vocab_size, (B, T))
    logits = model(idx)
    print(f"[5a] config: {cfg}")
    print(f"[5b] idx {tuple(idx.shape)} {idx.dtype} -> logits {tuple(logits.shape)} {logits.dtype}")
    _check(logits.shape == (B, T, cfg.vocab_size), "Shape logits salah!")

    try:
        model(torch.zeros(1, cfg.block_size + 1, dtype=torch.long))
        _check(False, "Seharusnya error karena T > block_size")
    except ValueError as e:
        print(f"[5c] validasi panjang sekuens bekerja: {e}")
    print("Tahap 5 LULUS ✅\n")


def tes_tahap_6() -> None:
    print("=== Tahap 6: Uji model lengkap ===")
    torch.manual_seed(42)
    cfg = ModelConfig()
    model = MiniMusicTransformer(cfg).eval()

    n = model.num_parameters()
    print(f"[6a] jumlah parameter: {n:,} (~{n / 1e6:.2f} juta)")

    B, T = 4, cfg.block_size
    idx = torch.randint(0, cfg.vocab_size, (B, T))
    logits = model(idx)
    print(f"[6b] input penuh {tuple(idx.shape)} -> logits {tuple(logits.shape)}")
    _check(logits.shape == (B, T, cfg.vocab_size), "Shape logits salah pada block_size penuh!")

    idx2 = idx.clone()
    idx2[:, -1] = (idx2[:, -1] + 1) % cfg.vocab_size
    causal = torch.allclose(model(idx2)[:, :-1], logits[:, :-1], atol=1e-5)
    print(f"[6c] model lengkap tetap causal: {causal}")
    _check(causal, "Model lengkap bocor melihat token masa depan!")

    # Loss awal harus mendekati ln(V) karena model belum belajar apa-apa (tebakan acak)
    target = torch.randint(1, cfg.vocab_size, (B, T))
    loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), target.reshape(-1), ignore_index=PAD_ID)
    print(f"[6d] loss awal: {loss.item():.3f} | teori tebakan acak ln({cfg.vocab_size}) = "
          f"{math.log(cfg.vocab_size):.3f}")
    _check(abs(loss.item() - math.log(cfg.vocab_size)) < 0.5, "Loss awal terlalu jauh dari ln(V)!")
    print("Tahap 6 LULUS ✅\n")


def tes_tahap_7(steps: int = 300) -> None:
    print("=== Tahap 7: Uji overfit 1 batch ===")
    print("Model harus bisa 'menghafal' satu batch kecil. Jika gagal, ada bug di model.")
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = ModelConfig(dropout=0.0)
    model = MiniMusicTransformer(cfg).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    B, T = 4, 32
    data = torch.randint(1, cfg.vocab_size, (B, T + 1), device=device)
    x, y = data[:, :-1], data[:, 1:]  # y = x digeser satu posisi (prediksi token berikutnya)
    print(f"[7a] device: {device} | batch x {tuple(x.shape)}, y {tuple(y.shape)}")

    for step in range(1, steps + 1):
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), y.reshape(-1), ignore_index=PAD_ID)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == 1 or step % 50 == 0:
            print(f"[7b] step {step:4d} | loss {loss.item():.4f}")

    acc = (model(x).argmax(dim=-1) == y).float().mean().item()
    print(f"[7c] loss akhir: {loss.item():.4f} | akurasi hafalan: {acc * 100:.1f}%")
    _check(loss.item() < 0.2 and acc > 0.95, "Model gagal menghafal satu batch, cek kembali arsitektur!")
    print("Tahap 7 LULUS ✅\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Uji model.py per tahap")
    parser.add_argument("--tahap", type=int, choices=range(1, 8),
                        help="nomor tahap yang diuji (1-7). Kosongkan untuk menguji semua.")
    args = parser.parse_args()

    semua_tes = [tes_tahap_1, tes_tahap_2, tes_tahap_3, tes_tahap_4,
                 tes_tahap_5, tes_tahap_6, tes_tahap_7]
    if args.tahap:
        semua_tes[args.tahap - 1]()
    else:
        for tes in semua_tes:
            tes()
        print("Semua tahap LULUS ✅ — model.py siap dipakai train.py")
