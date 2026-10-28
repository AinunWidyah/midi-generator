"""train.py — diisi melalui sesi vibe coding (lihat logs/)."""

from __future__ import annotations
 
import argparse
import csv
import math
import random
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
 
import matplotlib
 
matplotlib.use("Agg")  # tanpa layar, aman untuk Colab/Kaggle/server
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
 
# ---- ANTARMUKA KE MODUL ANGGOTA LAIN (SESUAIKAN BILA PERLU) -----------------
from src import tokenizer
from src.dataset import MidiDataset
from src.model import MiniGPT, ModelConfig
from src.tokenizer import PAD_ID, VOCAB_SIZE, tokens_to_ids
# -----------------------------------------------------------------------------
 
 
@dataclass
class TrainConfig:
    data_dir: str = "data/raw"
    out_dir: str = "outputs"
    run_name: str = ""          # isi untuk eksperimen tuning, mis. "lr1e-3"
    epochs: int = 30
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 200
    grad_clip: float = 1.0      # 0 = nonaktif
    label_smoothing: float = 0.0
    block_size: int = 256
    stride: int = 128
    val_ratio: float = 0.1
    patience: int = 0           # 0 = early stopping nonaktif
    num_workers: int = 0
    seed: int = 42
 
 
METRIC_FIELDS = [
    "epoch", "train_loss", "val_loss", "val_perplexity", "val_accuracy", "lr", "seconds",
]
 
 
# ----------------------------------------------------------------------------
# Utilitas
# ----------------------------------------------------------------------------
def parse_args() -> tuple[TrainConfig, bool]:
    parser = argparse.ArgumentParser(description="Training Transformer MIDI")
    for f in fields(TrainConfig):
        parser.add_argument(
            "--" + f.name.replace("_", "-"), type=type(f.default), default=f.default
        )
    parser.add_argument(
        "--smoke", action="store_true", help="uji loop dengan data acak (tanpa file MIDI)"
    )
    args = parser.parse_args()
    cfg = TrainConfig(**{f.name: getattr(args, f.name) for f in fields(TrainConfig)})
    return cfg, args.smoke
 
 
def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
 
 
def compute_loss(
    logits: torch.Tensor, targets: torch.Tensor, label_smoothing: float = 0.0
) -> torch.Tensor:
    """Cross-entropy yang mengabaikan PAD.
 
    Memakai .reshape, bukan .view: targets hasil slicing (data[:, 1:]) tidak
    contiguous di memori (lihat log 03, Eror 2.1).
    """
    vocab = logits.size(-1)
    return F.cross_entropy(
        logits.reshape(-1, vocab),
        targets.reshape(-1),
        ignore_index=PAD_ID,
        label_smoothing=label_smoothing,
    )
 
 
def build_optimizer(model: torch.nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    """AdamW; weight decay hanya untuk parameter 2D+ (bukan bias/LayerNorm)."""
    decay, no_decay = [], []
    for p in model.parameters():
        if p.requires_grad:
            (decay if p.ndim >= 2 else no_decay).append(p)
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=(0.9, 0.95))
 
 
def build_scheduler(optimizer, warmup_steps: int, total_steps: int):
    """Warmup linear lalu cosine decay sampai 10% dari lr awal."""
 
    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(progress, 1.0)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))
 
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
 
 
# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
class RandomTokenDataset(Dataset):
    """Data acak untuk --smoke. Loss yang wajar mendekati ln(V); tidak akan turun."""
 
    def __init__(self, n_items: int, block_size: int, vocab_size: int) -> None:
        self.n_items, self.block_size, self.vocab_size = n_items, block_size, vocab_size
 
    def __len__(self) -> int:
        return self.n_items
 
    def __getitem__(self, idx: int):
        seq = torch.randint(3, self.vocab_size, (self.block_size + 1,))
        return seq[:-1], seq[1:]
 
 
def load_sequences(paths: list[Path]) -> list[list[int]]:
    sequences = []
    for p in paths:
        try:
            tokens = tokenizer.encode(str(p))  # list[str]   (SESUAIKAN bila perlu)
            sequences.append(tokens_to_ids(tokens))  # list[int]
        except Exception as exc:  # file MIDI rusak jangan menghentikan training
            print(f"[lewati] {p.name}: {exc}")
    return sequences
 
 
def build_dataloaders(cfg: TrainConfig, smoke: bool, vocab_size: int):
    pin = torch.cuda.is_available()
    if smoke:
        train_ds = RandomTokenDataset(256, cfg.block_size, vocab_size)
        val_ds = RandomTokenDataset(64, cfg.block_size, vocab_size)
    else:
        paths = sorted(
            p for p in Path(cfg.data_dir).rglob("*") if p.suffix.lower() in {".mid", ".midi"}
        )
        if len(paths) < 2:
            raise FileNotFoundError(
                f"Butuh minimal 2 file MIDI di '{cfg.data_dir}', ditemukan {len(paths)}."
            )
        random.Random(cfg.seed).shuffle(paths)
        # Split per FILE (bukan per jendela) agar tidak ada kebocoran data ke validasi.
        n_val = max(1, int(len(paths) * cfg.val_ratio))
        val_seqs = load_sequences(paths[:n_val])
        train_seqs = load_sequences(paths[n_val:])
        print(f"File MIDI: {len(paths) - n_val} train, {n_val} validasi")
        # SESUAIKAN konstruktor dengan dataset.py (Anggota 2).
        train_ds = MidiDataset(train_seqs, block_size=cfg.block_size, stride=cfg.stride)
        val_ds = MidiDataset(val_seqs, block_size=cfg.block_size, stride=cfg.block_size)
        print(f"Jendela: {len(train_ds)} train, {len(val_ds)} validasi")
 
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=pin,
    )
    return train_loader, val_loader
 
 
# ----------------------------------------------------------------------------
# Training dan evaluasi
# ----------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, scheduler, cfg, device, epoch) -> float:
    model.train()
    total_loss, total_tokens = 0.0, 0
    bar = tqdm(loader, desc=f"epoch {epoch}", leave=False)
    for x, y in bar:
        x, y = x.to(device), y.to(device)
        logits = model(x)  # (B, T, V)
        loss = compute_loss(logits, y, cfg.label_smoothing)
        if not torch.isfinite(loss):
            raise RuntimeError(
                "Loss NaN/Inf. Coba turunkan --lr, periksa data/token di luar kosakata, "
                "dan pastikan --grad-clip > 0."
            )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        scheduler.step()
 
        n_tok = int((y != PAD_ID).sum())
        total_loss += loss.item() * n_tok
        total_tokens += n_tok
        bar.set_postfix(loss=f"{loss.item():.3f}")
    return total_loss / max(total_tokens, 1)
 
 
@torch.no_grad()
def evaluate(model, loader, device) -> dict[str, float]:
    """Loss/akurasi validasi dihitung per token non-PAD, tanpa label smoothing."""
    model.eval()
    loss_sum, correct, n_tokens = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        flat_logits = logits.reshape(-1, logits.size(-1))
        flat_y = y.reshape(-1)
        loss_sum += F.cross_entropy(
            flat_logits, flat_y, ignore_index=PAD_ID, reduction="sum"
        ).item()
        mask = flat_y != PAD_ID
        correct += int(((flat_logits.argmax(dim=-1) == flat_y) & mask).sum())
        n_tokens += int(mask.sum())
    val_loss = loss_sum / max(n_tokens, 1)
    return {
        "val_loss": val_loss,
        "val_perplexity": math.exp(min(val_loss, 20.0)),  # exp(val_loss), dijaga overflow
        "val_accuracy": correct / max(n_tokens, 1),
    }
 
 
# ----------------------------------------------------------------------------
# Pencatatan hasil
# ----------------------------------------------------------------------------
def save_checkpoint(path: Path, model, model_cfg, cfg, epoch: int, val_loss: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": asdict(model_cfg),
            "train_config": asdict(cfg),
            "epoch": epoch,
            "val_loss": val_loss,
        },
        path,
    )
 
 
def write_metrics_header(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        csv.DictWriter(f, fieldnames=METRIC_FIELDS).writeheader()
 
 
def append_metrics(path: Path, row: dict) -> None:
    with path.open("a", newline="") as f:
        csv.DictWriter(f, fieldnames=METRIC_FIELDS).writerow(row)
 
 
def plot_curves(history: list[dict], path: Path) -> None:
    epochs = [h["epoch"] for h in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(epochs, [h["train_loss"] for h in history], label="train")
    ax1.plot(epochs, [h["val_loss"] for h in history], label="validasi")
    ax1.set(xlabel="epoch", ylabel="loss", title="Kurva loss")
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax2.plot(epochs, [h["val_accuracy"] for h in history], color="tab:green")
    ax2.set(xlabel="epoch", ylabel="akurasi", title="Akurasi validasi (tanpa PAD)")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
 
 
# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> None:
    cfg, smoke = parse_args()
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
    # Tambahkan argumen lain (n_layer, n_head, d_model, ...) sesuai ModelConfig di model.py.
    model_cfg = ModelConfig(vocab_size=VOCAB_SIZE, block_size=cfg.block_size)
    model = MiniGPT(model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Device: {device} | parameter: {n_params / 1e6:.2f} juta | vocab: {VOCAB_SIZE}")
 
    train_loader, val_loader = build_dataloaders(cfg, smoke, VOCAB_SIZE)
    total_steps = cfg.epochs * max(1, len(train_loader))
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg.warmup_steps, total_steps)
 
    out = Path(cfg.out_dir)
    tag = f"_{cfg.run_name}" if cfg.run_name else ""
    metrics_path = out / f"metrics{tag}.csv"
    curve_path = out / f"loss_curve{tag}.png"
    ckpt_dir = out / "checkpoints"
    write_metrics_header(metrics_path)
 
    history: list[dict] = []
    best_val, bad_epochs = float("inf"), 0
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, cfg, device, epoch)
        val = evaluate(model, val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "val_loss": round(val["val_loss"], 5),
            "val_perplexity": round(val["val_perplexity"], 4),
            "val_accuracy": round(val["val_accuracy"], 5),
            "lr": f"{optimizer.param_groups[0]['lr']:.3e}",
            "seconds": round(time.time() - t0, 1),
        }
        history.append(row)
        append_metrics(metrics_path, row)
        plot_curves(history, curve_path)
        print(
            f"epoch {epoch:3d}/{cfg.epochs} | train {row['train_loss']:.4f} | "
            f"val {row['val_loss']:.4f} | ppl {row['val_perplexity']:.2f} | "
            f"acc {row['val_accuracy']:.3f} | {row['seconds']}s"
        )
 
        save_checkpoint(ckpt_dir / f"last{tag}.pt", model, model_cfg, cfg, epoch, val["val_loss"])
        if val["val_loss"] < best_val:
            best_val, bad_epochs = val["val_loss"], 0
            save_checkpoint(ckpt_dir / f"best{tag}.pt", model, model_cfg, cfg, epoch, best_val)
        else:
            bad_epochs += 1
            if cfg.patience > 0 and bad_epochs >= cfg.patience:
                print(f"Early stopping: val_loss tidak membaik selama {cfg.patience} epoch.")
                break
 
    print(f"Selesai. val_loss terbaik {best_val:.4f}")
    print(f"Metrik: {metrics_path} | grafik: {curve_path} | checkpoint: {ckpt_dir}/best{tag}.pt")
 
 
if __name__ == "__main__":
    main()
 
