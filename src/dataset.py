
"""dataset.py — Dataset dan DataLoader untuk MIDI String Generator."""

from pathlib import Path
import random
import warnings

import torch
from torch.utils.data import Dataset, DataLoader

try:
    from .config import TrainConfig, ModelConfig
    from . import tokenizer
except ImportError:
    from config import TrainConfig, ModelConfig
    import tokenizer


def find_midi_files(data_dir: Path) -> list[Path]:
    """Cari file MIDI, dengan melewati folder hasil karantina."""
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Folder dataset tidak ditemukan: {data_dir}"
        )

    excluded_dir = "_excluded_non_monophonic"
    files = []

    for path in data_dir.rglob("*"):
        if not path.is_file():
            continue

        if path.suffix.lower() not in {".mid", ".midi"}:
            continue

        # Jangan masukkan file dari folder karantina.
        if excluded_dir in path.parts:
            continue

        files.append(path)

    return sorted(files)


def resolve_data_dir(data_dir: str) -> Path:
    """Resolusi path relatif terhadap root proyek."""
    module_dir = Path(__file__).resolve().parent

    # Jika dataset.py berada di src/, root proyek satu tingkat di atasnya.
    if module_dir.name == "src":
        project_root = module_dir.parent
    else:
        project_root = module_dir

    path = Path(data_dir)
    if path.is_absolute():
        return path

    return project_root / path


def split_files(
    files: list[Path],
    val_ratio: float,
    seed: int,
) -> tuple[list[Path], list[Path]]:
    """Pisahkan file sebelum tokenisasi dan chunking."""
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio harus >= 0 dan < 1.")

    shuffled = list(files)
    random.Random(seed).shuffle(shuffled)

    if len(shuffled) < 2 or val_ratio == 0:
        return shuffled, []

    n_val = max(1, round(len(shuffled) * val_ratio))
    n_val = min(n_val, len(shuffled) - 1)

    val_files = shuffled[:n_val]
    train_files = shuffled[n_val:]

    return train_files, val_files


def encode_midi_file(path: Path) -> list[int]:
    """Encode MIDI memakai vocabulary resmi tokenizer.py."""
    tokens = tokenizer.encode(str(path))
    ids = tokenizer.tokens_to_ids(tokens)

    if not ids:
        raise ValueError("Tokenizer menghasilkan sequence kosong.")

    return ids


def make_chunks(
    token_ids: list[int],
    block_size: int,
) -> list[list[int]]:
    """
    Bentuk window untuk causal language modeling.

    Setiap window memiliki paling banyak block_size token input
    dan satu token tambahan sebagai target berikutnya.
    """
    if block_size < 1:
        raise ValueError("block_size harus minimal 1.")

    if len(token_ids) < 2:
        return []

    chunks = []

    # Window memiliki block_size + 1 token supaya input dan target
    # masing-masing dapat memiliki panjang block_size.
    window_size = block_size + 1

    for start in range(0, len(token_ids) - 1, block_size):
        chunk = token_ids[start:start + window_size]

        if len(chunk) < 2:
            continue

        chunks.append(chunk)

    return chunks


class MIDIStringDataset(Dataset):
    """Dataset token MIDI untuk pelatihan causal language model."""

    def __init__(
        self,
        midi_files: list[Path],
        block_size: int = 256,
    ):
        self.block_size = block_size
        self.examples: list[list[int]] = []
        self.skipped_files: list[tuple[str, str]] = []

        for path in midi_files:
            try:
                ids = encode_midi_file(path)
                chunks = make_chunks(ids, block_size)
                self.examples.extend(chunks)

                if not chunks:
                    self.skipped_files.append(
                        (str(path), "Sequence terlalu pendek")
                    )

            except Exception as exc:
                self.skipped_files.append((str(path), str(exc)))
                warnings.warn(
                    f"Gagal memproses {path}: {exc}",
                    RuntimeWarning,
                )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        chunk = self.examples[index]

        # Teacher forcing: prediksi token berikutnya.
        input_ids = chunk[:-1]
        labels = chunk[1:]

        # Panjang input maksimum block_size.
        input_ids = input_ids[:self.block_size]
        labels = labels[:self.block_size]

        valid_length = len(input_ids)
        pad_length = self.block_size - valid_length

        # PAD_ID berasal dari tokenizer, bukan ID buatan sendiri.
        input_ids += [tokenizer.PAD_ID] * pad_length

        # -100 diabaikan oleh CrossEntropyLoss PyTorch.
        labels += [-100] * pad_length

        attention_mask = (
            [1] * valid_length + [0] * pad_length
        )

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(
                attention_mask, dtype=torch.long
            ),
        }


def create_dataloaders(
    config: TrainConfig | None = None,
    model_config: ModelConfig | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Buat DataLoader train dan validation."""
    config = config or TrainConfig()
    model_config = model_config or ModelConfig()

    data_dir = resolve_data_dir(config.data_dir)
    files = find_midi_files(data_dir)

    if not files:
        raise RuntimeError(
            f"Tidak ada file MIDI di {data_dir}"
        )

    train_files, val_files = split_files(
        files,
        val_ratio=config.val_ratio,
        seed=config.seed,
    )

    train_dataset = MIDIStringDataset(
        train_files,
        block_size=model_config.block_size,
    )

    val_dataset = MIDIStringDataset(
        val_files,
        block_size=model_config.block_size,
    )

    if len(train_dataset) == 0:
        raise RuntimeError(
            "Dataset train kosong. Periksa file MIDI dan tokenizer."
        )

    # Gunakan generator dengan seed agar pengacakan reproducible.
    generator = torch.Generator()
    generator.manual_seed(config.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
    )

    print(f"Folder dataset : {data_dir}")
    print(f"File MIDI      : {len(files)}")
    print(f"File train     : {len(train_files)}")
    print(f"File validation: {len(val_files)}")
    print(f"Chunk train    : {len(train_dataset)}")
    print(f"Chunk validation: {len(val_dataset)}")
    print(f"Ukuran batch   : {config.batch_size}")
    print(f"Block size     : {model_config.block_size}")
    print(f"Vocabulary size: {len(tokenizer.TOKEN_TO_ID)}")
    print(
        f"File gagal     : "
        f"{len(train_dataset.skipped_files) + len(val_dataset.skipped_files)}"
    )

    return train_loader, val_loader


if __name__ == "__main__":
    train_loader, val_loader = create_dataloaders()

    batch = next(iter(train_loader))

    print("\nContoh batch:")
    for name, tensor in batch.items():
        print(f"{name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}")

    print("\nContoh input token ID:")
    print(batch["input_ids"][0][:20].tolist())

    print("\nContoh label token ID:")
    print(batch["labels"][0][:20].tolist())
