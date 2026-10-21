"""
Tokenizer MIDI String Generator — Kontrak Modul v1.0

Format:
    <BOS> (NOTE_<pitch>|REST) DUR_<1..16> ... <EOS>

Vocabulary dan ID diambil dari src/config.py melalui build_vocab().
Library pretty_midi hanya digunakan untuk I/O MIDI.
"""

from pathlib import Path
from typing import Iterable

import pretty_midi

try:
    # Import saat dipakai sebagai package: from src import tokenizer
    from .config import build_vocab
except ImportError:
    # Import saat script dijalankan dari folder src.
    from config import build_vocab


# Jangan membuat ulang urutan ID di sini; gunakan vocab dari config.py.
VOCAB = build_vocab()

if not isinstance(VOCAB, dict):
    raise TypeError(
        "build_vocab() harus mengembalikan dict dengan format {token: id}."
    )

TOKEN_TO_ID: dict[str, int] = VOCAB
ID_TO_TOKEN: dict[int, str] = {idx: token for token, idx in TOKEN_TO_ID.items()}

# Ambil ID token khusus dari config.py, bukan menetapkan angka baru.
try:
    PAD_ID = TOKEN_TO_ID["<PAD>"]
    BOS_ID = TOKEN_TO_ID["<BOS>"]
    EOS_ID = TOKEN_TO_ID["<EOS>"]
except KeyError as exc:
    raise ValueError(
        "build_vocab() harus berisi token <PAD>, <BOS>, dan <EOS>."
    ) from exc

BOS_TOKEN = ID_TO_TOKEN[BOS_ID]
EOS_TOKEN = ID_TO_TOKEN[EOS_ID]
PAD_TOKEN = ID_TO_TOKEN[PAD_ID]


def _duration_token(units: float) -> str:
    """Kuantisasi durasi ke unit not 1/16, dibatasi pada 1..16."""
    value = max(1, min(16, int(round(units))))
    return f"DUR_{value}"


def _seconds_to_quarter_beats(midi: pretty_midi.PrettyMIDI, seconds: float) -> float:
    """
    Konversi waktu MIDI (detik dari awal) ke quarter-note beats,
    dengan memperhitungkan perubahan tempo pada MIDI.
    """
    seconds = max(0.0, float(seconds))
    change_times, tempi = midi.get_tempo_changes()

    if len(change_times) == 0:
        return seconds * 2.0  # pretty_midi default 120 BPM

    beats = 0.0
    for i, start_time in enumerate(change_times):
        end_time = (
            float(change_times[i + 1])
            if i + 1 < len(change_times)
            else seconds
        )
        if seconds <= start_time:
            break

        segment_end = min(seconds, end_time)
        if segment_end > start_time:
            beats += (segment_end - float(start_time)) * float(tempi[i]) / 60.0

        if seconds <= end_time:
            break

    return max(0.0, beats)


def _choose_monophonic_instrument(midi: pretty_midi.PrettyMIDI):
    """
    Pilih track bernada monofonik dengan jumlah nada terbanyak.
    Track drum dan track kosong tidak digunakan.
    """
    candidates = []

    for instrument in midi.instruments:
        if instrument.is_drum or not instrument.notes:
            continue

        notes = sorted(instrument.notes, key=lambda n: (n.start, n.end))
        furthest_end = -1.0
        monophonic = True

        for note in notes:
            if note.start < furthest_end - 1e-4:
                monophonic = False
                break
            furthest_end = max(furthest_end, note.end)

        if monophonic:
            candidates.append((len(notes), instrument))

    if not candidates:
        # Tidak meng-crash jika file tidak punya track monofonik.
        # Pemanggil encode akan melaporkan error yang jelas.
        return None

    return max(candidates, key=lambda item: item[0])[1]


def encode(midi_path: str) -> list[str]:
    """
    Baca MIDI dan hasilkan list token dengan <BOS> dan <EOS>.
    Memilih track melodi monofonik bernada dengan jumlah nada terbanyak.
    """
    path = Path(midi_path)
    if not path.is_file():
        raise FileNotFoundError(f"File MIDI tidak ditemukan: {path}")

    midi = pretty_midi.PrettyMIDI(str(path))
    instrument = _choose_monophonic_instrument(midi)

    if instrument is None:
        raise ValueError(
            f"Tidak ditemukan track bernada monofonik pada file: {path.name}"
        )

    notes = sorted(instrument.notes, key=lambda n: (n.start, n.end))
    tokens = [BOS_TOKEN]
    previous_end = 0.0

    for note in notes:
        # Catat jeda antara akhir nada sebelumnya dan awal nada berikutnya.
        gap_seconds = note.start - previous_end
        if gap_seconds > 1e-4:
            gap_beats = (
                _seconds_to_quarter_beats(midi, note.start)
                - _seconds_to_quarter_beats(midi, previous_end)
            )
            gap_units = gap_beats * 4.0
            tokens.extend(["REST", _duration_token(gap_units)])

        # Durasi nada diukur dalam unit not 1/16.
        duration_beats = (
            _seconds_to_quarter_beats(midi, note.end)
            - _seconds_to_quarter_beats(midi, note.start)
        )
        duration_units = duration_beats * 4.0

        pitch = int(note.pitch)
        pitch = max(0, min(127, pitch))
        tokens.extend([f"NOTE_{pitch}", _duration_token(duration_units)])

        previous_end = note.end

    tokens.append(EOS_TOKEN)
    return tokens


def _parse_event_token(token: str):
    """Kembalikan ('note', pitch), ('rest', None), atau None jika tidak valid."""
    if token == "REST":
        return ("rest", None)

    if token.startswith("NOTE_"):
        try:
            pitch = int(token.split("_", 1)[1])
        except (ValueError, IndexError):
            return None

        candidate = f"NOTE_{pitch}"
        if 0 <= pitch <= 127 and candidate in TOKEN_TO_ID:
            return ("note", pitch)

    return None


def _parse_duration_token(token: str):
    """Ambil durasi 1..16; None jika token bukan DUR yang valid."""
    if not token.startswith("DUR_"):
        return None
    try:
        duration = int(token.split("_", 1)[1])
    except (ValueError, IndexError):
        return None

    if 1 <= duration <= 16 and f"DUR_{duration}" in TOKEN_TO_ID:
        return duration
    return None


def decode(tokens: list[str], out_path: str, tempo: float = 120.0) -> None:
    """
    Tulis MIDI dari token. Pasangan event/durasi yang tidak valid dilewati.
    Token REST menggerakkan waktu tanpa membuat nada.
    """
    if tempo <= 0:
        raise ValueError("tempo harus lebih besar dari 0.")

    midi = pretty_midi.PrettyMIDI(initial_tempo=float(tempo))
    instrument = pretty_midi.Instrument(program=0, is_drum=False)

    # Waktu dihitung dalam quarter-note beats, kemudian dikonversi ke detik.
    current_beats = 0.0
    i = 0

    while i < len(tokens):
        token = tokens[i]

        if token in (BOS_TOKEN, EOS_TOKEN, PAD_TOKEN):
            i += 1
            continue

        event = _parse_event_token(token)
        if event is None:
            # Token tak valid (termasuk DUR berurutan) dilewati.
            i += 1
            continue

        # Event harus langsung diikuti DUR valid.
        if i + 1 >= len(tokens):
            break

        duration_units = _parse_duration_token(tokens[i + 1])
        if duration_units is None:
            i += 1
            continue

        duration_beats = duration_units / 4.0
        kind, pitch = event

        if kind == "note":
            start = current_beats * 60.0 / float(tempo)
            end = (current_beats + duration_beats) * 60.0 / float(tempo)
            instrument.notes.append(
                pretty_midi.Note(
                    velocity=100,
                    pitch=int(pitch),
                    start=start,
                    end=max(start + 1e-4, end),
                )
            )

        current_beats += duration_beats
        i += 2

    midi.instruments.append(instrument)

    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(output))


def tokens_to_ids(tokens: list[str]) -> list[int]:
    """Konversi token ke ID sesuai vocabulary resmi dari config.py."""
    ids = []
    for token in tokens:
        if token not in TOKEN_TO_ID:
            raise ValueError(f"Token tidak dikenal dalam vocabulary: {token!r}")
        ids.append(int(TOKEN_TO_ID[token]))
    return ids


def ids_to_tokens(ids: list[int]) -> list[str]:
    """Konversi ID ke token; ID di luar vocabulary dilaporkan dengan jelas."""
    tokens = []
    for idx in ids:
        try:
            token_id = int(idx)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"ID token bukan integer: {idx!r}") from exc

        if token_id not in ID_TO_TOKEN:
            raise ValueError(f"ID token tidak dikenal: {token_id}")
        tokens.append(ID_TO_TOKEN[token_id])
    return tokens


def transpose(tokens: list[str], semitones: int) -> list[str]:
    """
    Transposisi token NOTE dengan jumlah semitone.
    Pitch dijepit ke rentang MIDI 0..127 agar tetap dalam vocabulary.
    Token non-NOTE dan token yang tidak valid dipertahankan.
    """
    shifted_tokens = []

    for token in tokens:
        if not token.startswith("NOTE_"):
            shifted_tokens.append(token)
            continue

        try:
            pitch = int(token.split("_", 1)[1])
        except (ValueError, IndexError):
            shifted_tokens.append(token)
            continue

        if not 0 <= pitch <= 127:
            shifted_tokens.append(token)
            continue

        shifted_pitch = max(0, min(127, pitch + int(semitones)))
        shifted_token = f"NOTE_{shifted_pitch}"

        # Proteksi jika config.py tidak memuat token yang seharusnya wajib.
        shifted_tokens.append(
            shifted_token if shifted_token in TOKEN_TO_ID else token
        )

    return shifted_tokens


def _run_self_test(midi_path: str, output_path: str) -> None:
    """Uji sederhana encode/decode, konversi ID, dan transposisi."""
    tokens = encode(midi_path)
    ids = tokens_to_ids(tokens)
    restored_tokens = ids_to_tokens(ids)

    assert restored_tokens == tokens, "Round-trip token <-> ID gagal."
    assert tokens[0] == BOS_TOKEN and tokens[-1] == EOS_TOKEN
    assert len(TOKEN_TO_ID) == 148, (
        f"Ukuran vocabulary seharusnya 148, ditemukan {len(TOKEN_TO_ID)}."
    )

    shifted = transpose(tokens, 2)
    assert len(shifted) == len(tokens), "Transposisi mengubah panjang token."
    assert all(token in TOKEN_TO_ID for token in shifted), (
        "Transposisi menghasilkan token di luar vocabulary."
    )

    decode(tokens, output_path)
    print(f"Self-test berhasil: {midi_path}")
    print(f"Jumlah token: {len(tokens)}")
    print(f"MIDI hasil decode: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Uji tokenizer MIDI sesuai Kontrak Modul v1.0."
    )
    parser.add_argument(
        "midi_path",
        nargs="?",
        help="Path ke file MIDI untuk pengujian.",
    )
    parser.add_argument(
        "--out",
        default="outputs/tokenizer_round_trip.mid",
        help="Path MIDI hasil decode (default: outputs/tokenizer_round_trip.mid).",
    )
    args = parser.parse_args()

    if args.midi_path:
        _run_self_test(args.midi_path, args.out)
    else:
        print("Tokenizer berhasil diimpor.")
        print(f"Ukuran vocabulary: {len(TOKEN_TO_ID)}")
        print("Untuk menguji round-trip:")
        print('python -m src.tokenizer "path/ke/lagu.mid"')
