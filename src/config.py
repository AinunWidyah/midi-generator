"""config.py — konfigurasi & vocabulary bersama (milik Ketua).

Semua modul mengimpor nilai dari file ini. Lihat docs/KONTRAK_MODUL.md.

Keputusan dasar (Kontrak v1.0):
    Framework       : PyTorch
    Dataset         : Nottingham (cadangan: JSB Chorales)
    Library I/O MIDI: pretty_midi (hanya baca/tulis .mid; tokenizer ditulis sendiri)
"""
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Token khusus & vocabulary
# ---------------------------------------------------------------------------
PAD_TOKEN = "<PAD>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"
REST_TOKEN = "REST"

NUM_PITCHES = 128      # pitch MIDI 0..127
MAX_DURATION = 16      # durasi dalam satuan not 1/16 (1..16)


def build_vocab() -> list[str]:
    """Bangun daftar token dengan urutan tetap (indeks = ID token)."""
    vocab = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN]
    vocab += [f"NOTE_{p}" for p in range(NUM_PITCHES)]
    vocab += [REST_TOKEN]
    vocab += [f"DUR_{d}" for d in range(1, MAX_DURATION + 1)]
    return vocab


VOCAB: list[str] = build_vocab()
TOKEN_TO_ID: dict[str, int] = {tok: i for i, tok in enumerate(VOCAB)}
VOCAB_SIZE: int = len(VOCAB)  # 148

PAD_ID = TOKEN_TO_ID[PAD_TOKEN]  # 0
BOS_ID = TOKEN_TO_ID[BOS_TOKEN]  # 1
EOS_ID = TOKEN_TO_ID[EOS_TOKEN]  # 2


# ---------------------------------------------------------------------------
# Konfigurasi model & training
# ---------------------------------------------------------------------------
@dataclass
class ModelConfig:
    vocab_size: int = VOCAB_SIZE
    block_size: int = 256   # panjang konteks maksimum (T)
    d_model: int = 256      # dimensi embedding (C)
    n_head: int = 4
    n_layer: int = 4
    dropout: float = 0.1


@dataclass
class TrainConfig:
    dataset_name: str = "nottingham"  # "nottingham" atau "jsb_chorales"
    data_dir: str = "data"
    output_dir: str = "outputs"
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    epochs: int = 30
    warmup_steps: int = 200
    grad_clip: float = 1.0
    label_smoothing: float = 0.0
    val_ratio: float = 0.1
    transpose_range: int = 5  # augmentasi: geser -5..+5 semiton
    seed: int = 42


if __name__ == "__main__":
    print(f"VOCAB_SIZE = {VOCAB_SIZE}")
    print(f"PAD/BOS/EOS = {PAD_ID}/{BOS_ID}/{EOS_ID}")
    print(f"NOTE_60 -> {TOKEN_TO_ID['NOTE_60']}, REST -> {TOKEN_TO_ID[REST_TOKEN]}, "
          f"DUR_1 -> {TOKEN_TO_ID['DUR_1']}, DUR_16 -> {TOKEN_TO_ID['DUR_16']}")
    print(ModelConfig())
    print(TrainConfig())
