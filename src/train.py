"""train.py — diisi melalui sesi vibe coding (lihat logs/)."""


from __future__ import annotations
 
import argparse
import csv
import math
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
 
import matplotlib
 
matplotlib.use("Agg")  # tanpa layar, aman untuk Colab/Kaggle/server
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
 
try:
    from . import tokenizer
    from .config import ModelConfig, TrainConfig
    from .dataset import create_dataloaders, resolve_data_dir as resolve_path
    from .model import MiniGPT  # SESUAIKAN nama kelas model
except ImportError:
    import tokenizer
    from config import ModelConfig, TrainConfig
    from dataset import create_dataloaders, resolve_data_dir as resolve_path
    from model import MiniGPT  # SESUAIKAN nama kelas model
 
# Label padding dari dataset.py. Harus sama dengan nilai -100 di MIDIStringDataset.
IGNORE_INDEX = -100
 
METRIC_FIELDS = [
    "epoch", "train_loss", "val_loss", "val_perplexity", "val_accuracy", "lr", "seconds",
]
 
 
# ----------------------------------------------------------------------------
# Konfigurasi
# ----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Training Transformer MIDI")
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--weight-decay", type=float)
    p.add_argument("--warmup-steps", type=int)
    p.add_argument("--grad-clip", type=float, help="0 = nonaktif")
    p.add_argument("--label-smoothing", type=float)
    p.add_argument("--patience", type=int, help="0 = early stopping nonaktif")
    p.add_argument("--out-dir", type=str)
    p.add_argument("--run-name", type=str, default="", help="akhiran file untuk eksperimen tuning")
    p.add_argument("--smoke", action="store_true", help="uji loop dengan data acak")
    return p.parse_args()
 
 
def _pick(obj, names: tuple[str, ...], default):
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return default
 
 
def resolve_hparams(cfg, args: argparse.Namespace) -> dict:
    hp = {
        "epochs": _pick(cfg, ("epochs", "num_epochs", "max_epochs"), 30),
        "lr": _pick(cfg, ("learning_rate", "lr"), 3e-4),
        "weight_decay": _pick(cfg, ("weight_decay",), 0.01),
        "warmup_steps": _pick(cfg, ("warmup_steps",), 200),
        "grad_clip": _pick(cfg, ("grad_clip", "max_grad_norm", "clip_grad_norm"), 1.0),
        "label_smoothing": _pick(cfg, ("label_smoothing",), 0.0),
        "patience": _pick(cfg, ("patience", "early_stopping_patience"), 0),
        "out_dir": _pick(cfg, ("out_dir", "output_dir"), "outputs"),
    }
    for key in hp:
        override = getattr(args, key)
        if override is not None:
            hp[key] = override
    return hp
 
 
def _plain(value):
    """Ubah ke tipe dasar agar checkpoint aman dimuat (torch.load weights_only)."""
    if isinstance(value, (int, float, str, bool, type(None))):
        return value
    if isinstance(value, (list, tuple)):
        return type(value)(_plain(v) for v in value)
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    return str(value)
 
 
def _to_dict(obj) -> dict:
    raw = asdict(obj) if is_dataclass(obj) else dict(vars(obj))
    return _plain(raw)
 
 
def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
 
 
# ----------------------------------------------------------------------------
# Loss, optimizer, scheduler
# ----------------------------------------------------------------------------
def compute_loss(
    logits: torch.Tensor, labels: torch.Tensor, label_smoothing: float = 0.0
) -> torch.Tensor:
    """Cross-entropy yang mengabaikan padding label (-100).
 
    labels dari dataset.py sudah digeser satu token (labels = chunk[1:]), jadi
    logits[:, t] langsung dibandingkan dengan labels[:, t], tanpa shift lagi.
    Memakai .reshape, bukan .view (lihat log 03, Eror 2.1).
    """
    vocab = logits.size(-1)
    return F.cross_entropy(
        logits.reshape(-1, vocab),
        labels.reshape(-1),
        ignore_index=IGNORE_INDEX,
        label_smoothing=label_smoothing,
    )
 
 
def build_optimizer(model: torch.nn.Module, hp: dict) -> torch.optim.Optimizer:
    """AdamW; weight decay hanya untuk parameter 2D+ (bukan bias/LayerNorm)."""
    decay, no_decay = [], []
    for p in model.parameters():
        if p.requires_grad:
            (decay if p.ndim >= 2 else no_decay).append(p)
    groups = [
        {"params": decay, "weight_decay": hp["weight_decay"]},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=hp["lr"], betas=(0.9, 0.95))
 
 
def build_scheduler(optimizer, warmup_steps: int, total_steps: int):
    """Warmup linear, lalu cosine decay sampai 10% dari lr awal."""
 
    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = min((step - warmup_steps) / max(1, total_steps - warmup_steps), 1.0)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))
 
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
 
 
# ----------------------------------------------------------------------------
# Data uji (--smoke)
# ----------------------------------------------------------------------------
class RandomTokenDataset(Dataset):
    """Data acak berformat sama dengan dataset.py. Loss wajar mendekati ln(V)."""
 
    def __init__(self, n_items: int, block_size: int, vocab_size: int) -> None:
        self.n_items, self.block_size, self.vocab_size = n_items, block_size, vocab_size
 
    def __len__(self) -> int:
        return self.n_items
 
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        seq = torch.randint(0, self.vocab_size, (self.block_size + 1,))
        return {
            "input_ids": seq[:-1].clone(),
            "labels": seq[1:].clone(),
            "attention_mask": torch.ones(self.block_size, dtype=torch.long),
        }
 
 
def build_smoke_loaders(batch_size: int, block_size: int, vocab_size: int):
    train_ds = RandomTokenDataset(256, block_size, vocab_size)
    val_ds = RandomTokenDataset(64, block_size, vocab_size)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
    )
 
 
# ----------------------------------------------------------------------------
# Training dan evaluasi
# ----------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, scheduler, hp, device, epoch) -> float:
    model.train()
    total_loss, total_tokens = 0.0, 0
    bar = tqdm(loader, desc=f"epoch {epoch}", leave=False)
    for batch in bar:
        x = batch["input_ids"].to(device)  # (B, T)
        y = batch["labels"].to(device)  # (B, T)
        # attention_mask tidak dipakai: model causal + padding di kanan, sehingga
        # posisi valid tidak pernah melihat token PAD.
        logits = model(x)  # (B, T, V)
        loss = compute_loss(logits, y, hp["label_smoothing"])
        if not torch.isfinite(loss):
            raise RuntimeError(
                "Loss NaN/Inf. Coba turunkan --lr, periksa token di luar kosakata, "
                "dan pastikan --grad-clip > 0."
            )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if hp["grad_clip"] > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), hp["grad_clip"])
        optimizer.step()
        scheduler.step()
 
        n_tok = int((y != IGNORE_INDEX).sum())
        total_loss += loss.item() * n_tok
        total_tokens += n_tok
        bar.set_postfix(loss=f"{loss.item():.3f}")
    return total_loss / max(total_tokens, 1)
 
 
@torch.no_grad()
def evaluate(model, loader, device) -> dict[str, float]:
    """Metrik validasi per token non-PAD, tanpa label smoothing."""
    model.eval()
    loss_sum, correct, n_tokens = 0.0, 0, 0
    for batch in loader:
        x = batch["input_ids"].to(device)
        y = batch["labels"].to(device)
        logits = model(x)
        flat_logits = logits.reshape(-1, logits.size(-1))
        flat_y = y.reshape(-1)
        loss_sum += F.cross_entropy(
            flat_logits, flat_y, ignore_index=IGNORE_INDEX, reduction="sum"
        ).item()
        mask = flat_y != IGNORE_INDEX
        correct += int(((flat_logits.argmax(dim=-1) == flat_y) & mask).sum())
        n_tokens += int(mask.sum())
    if n_tokens == 0:  # tidak ada data validasi
        nan = float("nan")
        return {"val_loss": nan, "val_perplexity": nan, "val_accuracy": nan}
    val_loss = loss_sum / n_tokens
    return {
        "val_loss": val_loss,
        "val_perplexity": math.exp(min(val_loss, 20.0)),  # exp(val_loss), dijaga overflow
        "val_accuracy": correct / n_tokens,
    }
 
 
# ----------------------------------------------------------------------------
# Pencatatan hasil
# ----------------------------------------------------------------------------
def save_checkpoint(path: Path, model, model_cfg, train_state: dict, epoch: int, val_loss: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": _to_dict(model_cfg),
            "train_config": train_state,
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
    args = parse_args()
    cfg = TrainConfig()
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    model_cfg = ModelConfig()
    hp = resolve_hparams(cfg, args)
 
    set_seed(getattr(cfg, "seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
    vocab_size = len(tokenizer.TOKEN_TO_ID)
    cfg_vocab = getattr(model_cfg, "vocab_size", None)
    if cfg_vocab is not None and cfg_vocab != vocab_size:
        raise ValueError(
            f"ModelConfig.vocab_size={cfg_vocab} tidak sama dengan ukuran kosakata "
            f"tokenizer ({vocab_size})."
        )
 
    model = MiniGPT(model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Device: {device} | parameter: {n_params / 1e6:.2f} juta | vocab: {vocab_size}")
    print("Hyperparameter:", {k: v for k, v in hp.items()}, "| batch_size:", cfg.batch_size)
 
    if args.smoke:
        train_loader, val_loader = build_smoke_loaders(
            cfg.batch_size, model_cfg.block_size, vocab_size
        )
    else:
        train_loader, val_loader = create_dataloaders(cfg, model_cfg)
    if len(val_loader.dataset) == 0:
        print("PERINGATAN: data validasi kosong; checkpoint terbaik dipilih dari train_loss.")
 
    total_steps = hp["epochs"] * max(1, len(train_loader))
    optimizer = build_optimizer(model, hp)
    scheduler = build_scheduler(optimizer, hp["warmup_steps"], total_steps)
 
    out = resolve_path(hp["out_dir"])
    tag = f"_{args.run_name}" if args.run_name else ""
    metrics_path = out / f"metrics{tag}.csv"
    curve_path = out / f"loss_curve{tag}.png"
    ckpt_dir = out / "checkpoints"
    write_metrics_header(metrics_path)
    train_state = {**_to_dict(cfg), **hp, "run_name": args.run_name}
 
    history: list[dict] = []
    best, bad_epochs = float("inf"), 0
    for epoch in range(1, hp["epochs"] + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, hp, device, epoch)
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
            f"epoch {epoch:3d}/{hp['epochs']} | train {row['train_loss']:.4f} | "
            f"val {row['val_loss']:.4f} | ppl {row['val_perplexity']:.2f} | "
            f"acc {row['val_accuracy']:.3f} | {row['seconds']}s"
        )
 
        monitor = train_loss if math.isnan(val["val_loss"]) else val["val_loss"]
        save_checkpoint(ckpt_dir / f"last{tag}.pt", model, model_cfg, train_state, epoch, monitor)
        if monitor < best:
            best, bad_epochs = monitor, 0
            save_checkpoint(ckpt_dir / f"best{tag}.pt", model, model_cfg, train_state, epoch, best)
        else:
            bad_epochs += 1
            if hp["patience"] > 0 and bad_epochs >= hp["patience"]:
                print(f"Early stopping: tidak membaik selama {hp['patience']} epoch.")
                break
 
    print(f"Selesai. Nilai terbaik {best:.4f}")
    print(f"Metrik: {metrics_path} | grafik: {curve_path} | checkpoint: {ckpt_dir}/best{tag}.pt")
 
 
if __name__ == "__main__":
    main()
 
