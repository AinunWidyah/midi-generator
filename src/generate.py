"""generate.py — diisi melalui sesi vibe coding (lihat logs/)."""


from __future__ import annotations
 
import argparse
import time
from pathlib import Path
 
import torch
import torch.nn.functional as F
 
try:
    from . import tokenizer
    from .config import ModelConfig
    from .dataset import resolve_data_dir as resolve_path
    from .model import MiniGPT  # SESUAIKAN nama kelas model
except ImportError:
    import tokenizer
    from config import ModelConfig
    from dataset import resolve_data_dir as resolve_path
    from model import MiniGPT  # SESUAIKAN nama kelas model
 
PAD_ID = tokenizer.PAD_ID
EOS_ID = tokenizer.EOS_ID
BOS_ID = getattr(tokenizer, "BOS_ID", None)
 
 
def _get_block_size(model: torch.nn.Module) -> int:
    """Ambil block_size dari model (mencoba model.block_size / .cfg / .config)."""
    for owner in (model, getattr(model, "cfg", None), getattr(model, "config", None)):
        value = getattr(owner, "block_size", None)
        if value is not None:
            return int(value)
    raise AttributeError("block_size tidak ditemukan pada model (cek model.py)")
 
 
def _sample_next(logits: torch.Tensor, temperature: float, top_k: int | None) -> int:
    """Pilih satu token dari logits berbentuk (1, V)."""
    if temperature == 0:  # greedy
        return int(logits.argmax(dim=-1))
    logits = logits / temperature
    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        kth_value = torch.topk(logits, k).values[:, -1, None]
        logits = logits.masked_fill(logits < kth_value, float("-inf"))
    probs = F.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1))
 
 
@torch.no_grad()
def generate(
    model,
    prompt_ids: list[int],
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> list[int]:
    """Sampling autoregresif.
 
    - Bila sekuens lebih panjang dari block_size, hanya block_size token
      terakhir yang dimasukkan ke model.
    - Berhenti saat menghasilkan EOS_ID atau mencapai max_new_tokens.
    - Mengembalikan prompt + token baru (EOS penutup tidak disertakan).
    - temperature = 0 berarti greedy; PAD (dan BOS) tidak pernah dipilih.
    """
    if not prompt_ids:
        raise ValueError("prompt_ids tidak boleh kosong")
    if temperature < 0:
        raise ValueError("temperature tidak boleh negatif")
 
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    block_size = _get_block_size(model)
    banned = [i for i in (PAD_ID, BOS_ID) if i is not None]
    ids = list(prompt_ids)
    try:
        for _ in range(max_new_tokens):
            context = ids[-block_size:]
            x = torch.tensor([context], dtype=torch.long, device=device)  # (1, T)
            logits = model(x)[:, -1, :]  # (1, V): hanya posisi terakhir
            for bad_id in banned:
                logits[:, bad_id] = float("-inf")
            next_id = _sample_next(logits, temperature, top_k)
            if next_id == EOS_ID:
                break
            ids.append(next_id)
    finally:
        model.train(was_training)
    return ids
 
 
def load_model(checkpoint_path: Path, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model_cfg = ModelConfig(**ckpt["model_config"])
    model = MiniGPT(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt
 
 
def save_midi(ids: list[int], path: Path) -> None:
    """list[int] -> list[str] -> .mid (token khusus dibuang lebih dulu)."""
    special = {i for i in (PAD_ID, EOS_ID, BOS_ID) if i is not None}
    tokens = tokenizer.ids_to_tokens([i for i in ids if i not in special])
    midi = tokenizer.decode(tokens)  # SESUAIKAN bila decode langsung menulis file
    path.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(path))
 
 
def build_prompt(args: argparse.Namespace) -> list[int]:
    if args.prompt_midi:
        ids = tokenizer.tokens_to_ids(tokenizer.encode(str(args.prompt_midi)))
        ids = [i for i in ids if i != EOS_ID][: args.prompt_tokens]
        if not ids:
            raise ValueError(f"Prompt kosong dari {args.prompt_midi}")
        return ids
    if BOS_ID is None:
        raise ValueError("Tokenizer tidak punya BOS_ID; gunakan --prompt-midi.")
    return [BOS_ID]
 
 
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate musik MIDI")
    p.add_argument("--checkpoint", default="outputs/checkpoints/best.pt")
    p.add_argument("--out-dir", default="outputs/generated")
    p.add_argument("--n-samples", type=int, default=3)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=40, help="0 = nonaktif")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--prompt-midi", default=None, help="lanjutkan dari awal file MIDI ini")
    p.add_argument("--prompt-tokens", type=int, default=128)
    p.add_argument("--device", default=None)
    return p.parse_args()
 
 
def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if args.seed is not None:
        torch.manual_seed(args.seed)
 
    model, ckpt = load_model(resolve_path(args.checkpoint), device)
    print(f"Checkpoint epoch {ckpt.get('epoch')} (nilai {ckpt.get('val_loss'):.4f}) | {device}")
 
    prompt = build_prompt(args)
    top_k = args.top_k if args.top_k > 0 else None
    out_dir = resolve_path(args.out_dir)
    stamp = time.strftime("%Y%m%d_%H%M%S")
 
    for i in range(args.n_samples):
        ids = generate(model, prompt, args.max_new_tokens, args.temperature, top_k)
        name = f"sample_{stamp}_{i + 1:02d}_T{args.temperature}_k{top_k}.mid"
        save_midi(ids, out_dir / name)
        print(f"[{i + 1}/{args.n_samples}] {len(ids) - len(prompt)} token baru -> {out_dir / name}")
 
 
if __name__ == "__main__":
    main()
 
