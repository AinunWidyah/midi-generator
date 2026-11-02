"""src/baseline.py - Akurasi baseline sederhana sebagai pembanding model (Anggota 3).

Jalankan dari root repo atau dari folder src/:
    python src/baseline.py

Menghitung akurasi tiga tebakan tanpa belajar pada data VALIDASI, dengan statistik
yang diambil dari data TRAIN:
  A. Selalu menebak satu token yang paling umum.
  B. Menebak token paling umum menurut TIPE token sebelumnya (durasi DUR_ atau bukan).
  C. Bigram: token paling umum setelah token sebelumnya.

Akurasi model di metrics.csv dihitung dengan prefiks asli (teacher forcing), sama
seperti baseline di sini, jadi angkanya sebanding.
"""
from __future__ import annotations

from collections import Counter, defaultdict

IGNORE_INDEX = -100  # label padding dari dataset.py


def collect_pairs(loader) -> tuple[list[int], list[int]]:
    """Kumpulkan pasangan (token sebelumnya, token target) dari seluruh batch."""
    prev, nxt = [], []
    for batch in loader:
        x, y = batch["input_ids"], batch["labels"]
        mask = y != IGNORE_INDEX
        prev.extend(x[mask].tolist())
        nxt.extend(y[mask].tolist())
    return prev, nxt


def evaluate_baselines(train_pairs, val_pairs, is_dur) -> dict[str, float]:
    """Hitung akurasi baseline A, B, C pada data validasi."""
    t_prev, t_next = train_pairs
    v_prev, v_next = val_pairs
    if not t_next or not v_next:
        raise ValueError("Data train atau validasi kosong.")

    global_best = Counter(t_next).most_common(1)[0][0]
    by_type: dict[bool, Counter] = defaultdict(Counter)
    by_prev: dict[int, Counter] = defaultdict(Counter)
    for p, n in zip(t_prev, t_next):
        by_type[is_dur(p)][n] += 1
        by_prev[p][n] += 1
    type_best = {k: c.most_common(1)[0][0] for k, c in by_type.items()}
    prev_best = {k: c.most_common(1)[0][0] for k, c in by_prev.items()}

    hits = {"A": 0, "B": 0, "C": 0}
    for p, n in zip(v_prev, v_next):
        guess_b = type_best.get(is_dur(p), global_best)
        guess_c = prev_best.get(p, guess_b)  # token baru: jatuh ke tebakan B
        hits["A"] += global_best == n
        hits["B"] += guess_b == n
        hits["C"] += guess_c == n
    total = len(v_next)
    return {k: v / total for k, v in hits.items()}


def duration_stats(targets: list[int], is_dur) -> tuple[float, int | None, float]:
    """Porsi token durasi di target, id durasi terbanyak, dan porsinya di antara durasi."""
    if not targets:
        return 0.0, None, 0.0
    durations = [t for t in targets if is_dur(t)]
    if not durations:
        return 0.0, None, 0.0
    top_id, top_count = Counter(durations).most_common(1)[0]
    return len(durations) / len(targets), top_id, top_count / len(durations)


def _load_team_modules():
    try:
        from . import tokenizer
        from .config import ModelConfig, TrainConfig
        from .dataset import create_dataloaders
    except ImportError:
        import tokenizer
        from config import ModelConfig, TrainConfig
        from dataset import create_dataloaders
    return tokenizer, ModelConfig, TrainConfig, create_dataloaders


def main() -> None:
    tokenizer, ModelConfig, TrainConfig, create_dataloaders = _load_team_modules()
    train_loader, val_loader = create_dataloaders(TrainConfig(), ModelConfig())
    if len(val_loader.dataset) == 0:
        raise SystemExit("Data validasi kosong; atur val_ratio di config.py.")

    names = {i: t for t, i in tokenizer.TOKEN_TO_ID.items()}

    def is_dur(token_id: int) -> bool:
        return names.get(token_id, "").startswith("DUR_")

    train_pairs = collect_pairs(train_loader)
    val_pairs = collect_pairs(val_loader)
    scores = evaluate_baselines(train_pairs, val_pairs, is_dur)
    dur_share, top_id, top_share = duration_stats(val_pairs[1], is_dur)

    print(f"\nBaseline pada data validasi ({len(val_pairs[1])} token target)")
    print(f"  A  token paling umum (selalu sama)               : {scores['A']:.1%}")
    print(f"  B  paling umum menurut tipe token sebelumnya     : {scores['B']:.1%}")
    print(f"  C  bigram (paling umum setelah token sebelumnya) : {scores['C']:.1%}")
    print(f"\nToken durasi (DUR_) : {dur_share:.1%} dari seluruh target")
    if top_id is not None:
        print(f"Durasi terbanyak    : {names[top_id]} = {top_share:.1%} dari token durasi")
    print("\nBandingkan dengan val_accuracy epoch terbaik di outputs/metrics.csv.")


if __name__ == "__main__":
    main()
