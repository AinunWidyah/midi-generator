from pathlib import Path
import csv
import pretty_midi

# ============================================================
# KONFIGURASI
# ============================================================

# Folder dataset adalah folder tempat file audit_midi.py berada.
DATA_DIR = Path(__file__).resolve().parent

# True: cari file MIDI sampai ke seluruh subfolder.
RECURSIVE = True

# Toleransi kecil untuk menghindari kesalahan floating-point (detik).
TOLERANCE_SECONDS = 1e-4

# Batas panjang konteks model sesuai KONTRAK_MODUL.md.
BLOCK_SIZE = 256

# File output audit.
SUMMARY_CSV = DATA_DIR / "audit_midi_summary.csv"
TRACKS_CSV = DATA_DIR / "audit_midi_tracks.csv"


def find_midi_files(folder: Path):
    """Cari file .mid dan .midi, tanpa memasukkan file output audit."""
    if RECURSIVE:
        candidates = list(folder.rglob("*.mid")) + list(folder.rglob("*.midi"))
    else:
        candidates = list(folder.glob("*.mid")) + list(folder.glob("*.midi"))

    excluded = {SUMMARY_CSV.resolve(), TRACKS_CSV.resolve()}
    return sorted(
        {
            path.resolve()
            for path in candidates
            if path.resolve() not in excluded
        }
    )


def note_duration_units(midi, note):
    """
    Hitung durasi nada dalam satuan not 1/16.
    1 not 1/16 = seperempat not / 4.
    Konversi menggunakan tick MIDI agar tempo map lebih diperhitungkan.
    """
    start_tick = midi.time_to_tick(note.start)
    end_tick = midi.time_to_tick(note.end)
    return max(0.0, (end_tick - start_tick) / midi.resolution * 4.0)


def inspect_track(midi, instrument):
    """Audit satu track/instrumen dan kembalikan statistiknya."""
    notes = sorted(instrument.notes, key=lambda note: (note.start, note.end))
    note_count = len(notes)

    overlap_count = 0
    furthest_end = None

    for note in notes:
        if (
            furthest_end is not None
            and note.start < furthest_end - TOLERANCE_SECONDS
        ):
            overlap_count += 1

        if furthest_end is None:
            furthest_end = note.end
        else:
            furthest_end = max(furthest_end, note.end)

    # Monofonik jika tidak ada nada yang overlap pada track ini.
    monophonic = note_count > 0 and overlap_count == 0

    units = [note_duration_units(midi, note) for note in notes]
    quantized_units = [
        min(16, max(1, int(round(value))))
        for value in units
    ]

    if units:
        units_min = round(min(units), 4)
        units_max = round(max(units), 4)
        units_mean = round(sum(units) / len(units), 4)
        in_range_count = sum(1 for value in units if 1 <= value <= 16)
        in_range_percent = round(100 * in_range_count / len(units), 2)
    else:
        units_min = ""
        units_max = ""
        units_mean = ""
        in_range_percent = ""

    # Estimasi token untuk format kontrak:
    # <BOS>, pasangan NOTE/REST + DUR, <EOS>.
    # Setiap nada = 2 token. Setiap jeda positif sebelum nada = 2 token.
    # Jeda lebih dari 16 satuan tetap dihitung sebagai satu pasangan REST/DUR,
    # karena kontrak menetapkan durasi yang lebih dari 16 dipotong menjadi 16.
    estimated_tokens = 0
    rest_events = 0

    for index, note in enumerate(notes):
        estimated_tokens += 2  # NOTE + DUR

        if index > 0:
            previous = notes[index - 1]
            gap_seconds = note.start - previous.end
            if gap_seconds > TOLERANCE_SECONDS:
                rest_events += 1
                estimated_tokens += 2  # REST + DUR

    if note_count:
        estimated_tokens += 2  # <BOS> dan <EOS>

    return {
        "note_count": note_count,
        "overlap_count": overlap_count,
        "monophonic": monophonic,
        "duration_units_min": units_min,
        "duration_units_max": units_max,
        "duration_units_mean": units_mean,
        "duration_1_to_16_percent": in_range_percent,
        "rest_events_estimate": rest_events,
        "estimated_tokens": estimated_tokens,
        "over_block_size": estimated_tokens > BLOCK_SIZE,
        "pitch_min": min((note.pitch for note in notes), default=""),
        "pitch_max": max((note.pitch for note in notes), default=""),
        "velocity_min": min((note.velocity for note in notes), default=""),
        "velocity_max": max((note.velocity for note in notes), default=""),
        "duration_units": quantized_units,
    }


def main():
    files = find_midi_files(DATA_DIR)
    summary_rows = []
    track_rows = []

    print("=" * 68)
    print("AUDIT DATASET MIDI — MIDI STRING GENERATOR")
    print("=" * 68)
    print("Folder dataset :", DATA_DIR)
    print("Total file MIDI:", len(files))
    print("Block size     :", BLOCK_SIZE)
    print()

    for file_index, path in enumerate(files, start=1):
        relative_path = str(path.relative_to(DATA_DIR))

        try:
            midi = pretty_midi.PrettyMIDI(str(path))
            instruments = midi.instruments
            tracks_with_notes = [
                (index, instrument)
                for index, instrument in enumerate(instruments)
                if instrument.notes
            ]

            file_track_stats = []
            for instrument_index, instrument in enumerate(instruments):
                stats = inspect_track(midi, instrument)
                track_name = instrument.name or f"Instrument_{instrument_index}"

                track_rows.append({
                    "file": relative_path,
                    "instrument_index": instrument_index,
                    "instrument_name": track_name,
                    "program": instrument.program,
                    "is_drum": instrument.is_drum,
                    "note_count": stats["note_count"],
                    "overlap_pairs_or_events": stats["overlap_count"],
                    "monophonic_candidate": stats["monophonic"],
                    "duration_units_min": stats["duration_units_min"],
                    "duration_units_max": stats["duration_units_max"],
                    "duration_units_mean": stats["duration_units_mean"],
                    "duration_1_to_16_percent": stats["duration_1_to_16_percent"],
                    "rest_events_estimate": stats["rest_events_estimate"],
                    "estimated_tokens": stats["estimated_tokens"],
                    "exceeds_block_size_256": stats["over_block_size"],
                    "pitch_min": stats["pitch_min"],
                    "pitch_max": stats["pitch_max"],
                    "velocity_min": stats["velocity_min"],
                    "velocity_max": stats["velocity_max"],
                })

                if stats["note_count"] > 0:
                    file_track_stats.append((instrument_index, instrument, stats))

            # Pilih track monofonik dengan nada terbanyak sebagai kandidat.
            # Jika tidak ada track monofonik, gunakan track bernada terbanyak
            # hanya untuk estimasi; hasilnya tetap ditandai agar ditinjau.
            monophonic_tracks = [
                item for item in file_track_stats if item[2]["monophonic"]
            ]
            candidate_pool = monophonic_tracks or file_track_stats
            candidate = (
                max(candidate_pool, key=lambda item: item[2]["note_count"])
                if candidate_pool else None
            )

            all_notes = [
                note
                for instrument in instruments
                for note in instrument.notes
            ]

            if candidate:
                candidate_index, candidate_instrument, candidate_stats = candidate
                candidate_name = (
                    candidate_instrument.name or f"Instrument_{candidate_index}"
                )
                candidate_is_monophonic = candidate_stats["monophonic"]
                candidate_note_count = candidate_stats["note_count"]
                candidate_tokens = candidate_stats["estimated_tokens"]
                candidate_exceeds = candidate_stats["over_block_size"]
                duration_percent = candidate_stats["duration_1_to_16_percent"]
                candidate_pitch_min = candidate_stats["pitch_min"]
                candidate_pitch_max = candidate_stats["pitch_max"]
            else:
                candidate_index = ""
                candidate_name = ""
                candidate_is_monophonic = False
                candidate_note_count = 0
                candidate_tokens = 0
                candidate_exceeds = False
                duration_percent = ""
                candidate_pitch_min = ""
                candidate_pitch_max = ""

            if not all_notes:
                status = "EMPTY_NO_NOTES"
            elif not tracks_with_notes:
                status = "NO_TRACK_WITH_NOTES"
            else:
                status = "READABLE"

            summary_rows.append({
                "file": relative_path,
                "status": status,
                "duration_seconds": round(midi.get_end_time(), 4),
                "instrument_count": len(instruments),
                "tracks_with_notes": len(tracks_with_notes),
                "total_notes_all_tracks": len(all_notes),
                "monophonic_tracks_with_notes": len(monophonic_tracks),
                "polyphonic_tracks_with_notes": sum(
                    1 for item in file_track_stats if not item[2]["monophonic"]
                ),
                "candidate_track_index": candidate_index,
                "candidate_track_name": candidate_name,
                "candidate_track_is_monophonic": candidate_is_monophonic,
                "candidate_track_note_count": candidate_note_count,
                "candidate_duration_1_to_16_percent": duration_percent,
                "candidate_pitch_min": candidate_pitch_min,
                "candidate_pitch_max": candidate_pitch_max,
                "estimated_tokens_candidate_track": candidate_tokens,
                "candidate_exceeds_block_size_256": candidate_exceeds,
                "error": "",
            })

        except Exception as error:
            summary_rows.append({
                "file": relative_path,
                "status": "FAILED_TO_READ",
                "duration_seconds": "",
                "instrument_count": "",
                "tracks_with_notes": "",
                "total_notes_all_tracks": "",
                "monophonic_tracks_with_notes": "",
                "polyphonic_tracks_with_notes": "",
                "candidate_track_index": "",
                "candidate_track_name": "",
                "candidate_track_is_monophonic": "",
                "candidate_track_note_count": "",
                "candidate_duration_1_to_16_percent": "",
                "candidate_pitch_min": "",
                "candidate_pitch_max": "",
                "estimated_tokens_candidate_track": "",
                "candidate_exceeds_block_size_256": "",
                "error": str(error),
            })

        if file_index % 100 == 0 or file_index == len(files):
            print(f"Diperiksa: {file_index}/{len(files)}")

    # Simpan ringkasan per file.
    if summary_rows:
        with open(SUMMARY_CSV, "w", newline="", encoding="utf-8-sig") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)

    # Simpan detail per instrumen.
    if track_rows:
        with open(TRACKS_CSV, "w", newline="", encoding="utf-8-sig") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=list(track_rows[0].keys()))
            writer.writeheader()
            writer.writerows(track_rows)

    # Ringkasan terminal.
    readable = sum(row["status"] == "READABLE" for row in summary_rows)
    empty = sum(row["status"] == "EMPTY_NO_NOTES" for row in summary_rows)
    failed = sum(row["status"] == "FAILED_TO_READ" for row in summary_rows)
    with_notes = [
        row for row in summary_rows
        if row["status"] == "READABLE"
    ]
    no_mono_candidate = sum(
        row["status"] == "READABLE"
        and row["monophonic_tracks_with_notes"] == 0
        for row in summary_rows
    )
    over_block = sum(
        row["status"] == "READABLE"
        and row["candidate_exceeds_block_size_256"]
        for row in summary_rows
    )

    print("\n" + "=" * 68)
    print("RINGKASAN AUDIT")
    print("=" * 68)
    print(f"Total file MIDI                      : {len(files)}")
    print(f"File terbaca dan berisi nada         : {readable}")
    print(f"File tanpa nada                       : {empty}")
    print(f"File gagal dibaca                     : {failed}")
    print(f"File tanpa track monofonik bernada    : {no_mono_candidate}")
    print(f"Kandidat track melebihi {BLOCK_SIZE} token : {over_block}")
    print("\nCSV ringkasan per file :", SUMMARY_CSV)
    print("CSV detail per track   :", TRACKS_CSV)
    print("\nCATATAN:")
    print("- Estimasi token mengikuti format kontrak dan bukan hasil tokenizer final.")
    print("- Kandidat track adalah track monofonik bernada dengan nada terbanyak.")
    print("- Jika tidak ada track monofonik, track bernada terbanyak dipakai hanya")
    print("  untuk estimasi dan ditandai candidate_track_is_monophonic = False.")
    print("- Periksa hasil CSV sebelum memutuskan filtering/preprocessing.")
    print("\nAudit selesai.")


if __name__ == "__main__":
    main()
