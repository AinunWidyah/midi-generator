from pathlib import Path
import csv
import shutil
import pretty_midi

# ============================================================
# KONFIGURASI — PERIKSA SEBELUM MENJALANKAN
# ============================================================

# Folder tempat audit_midi.py berada.
DATA_DIR = Path(__file__).resolve().parent

# True: cari MIDI sampai ke subfolder.
RECURSIVE = True

# Mode aman: True hanya membuat daftar file yang akan dikecualikan.
# Ubah menjadi False jika sudah memeriksa daftar dan benar-benar ingin
# memindahkan file dari dataset aktif.
DRY_RUN = False

# Folder karantina untuk file yang dikecualikan.
# File dipindahkan, bukan dihapus permanen, agar bisa dipulihkan.
QUARANTINE_DIR = DATA_DIR / "_excluded_non_monophonic"

# Laporan CSV.
REPORT_CSV = DATA_DIR / "excluded_midi_report.csv"

# Toleransi overlap dalam detik.
TOLERANCE_SECONDS = 1e-4


def find_midi_files():
    if RECURSIVE:
        paths = list(DATA_DIR.rglob("*.mid")) + list(DATA_DIR.rglob("*.midi"))
    else:
        paths = list(DATA_DIR.glob("*.mid")) + list(DATA_DIR.glob("*.midi"))

    excluded_roots = [QUARANTINE_DIR.resolve()]
    output_files = {REPORT_CSV.resolve()}
    result = []

    for path in paths:
        resolved = path.resolve()

        # Jangan audit file yang sudah berada di karantina atau laporan CSV.
        if any(resolved == root or root in resolved.parents for root in excluded_roots):
            continue
        if resolved in output_files:
            continue
        result.append(path)

    return sorted(set(result))


def has_overlap_in_track(instrument):
    """True jika ada dua nada bertumpang tindih pada instrumen yang sama."""
    notes = sorted(instrument.notes, key=lambda note: (note.start, note.end))
    furthest_end = None

    for note in notes:
        if furthest_end is not None and note.start < furthest_end - TOLERANCE_SECONDS:
            return True

        furthest_end = note.end if furthest_end is None else max(furthest_end, note.end)

    return False


def main():
    files = find_midi_files()
    report = []
    to_exclude = []

    print("=" * 68)
    print("PEMERIKSAAN TRACK MONOFONIK — MODE AMAN")
    print("=" * 68)
    print("Folder dataset :", DATA_DIR)
    print("Jumlah MIDI    :", len(files))
    print("DRY_RUN        :", DRY_RUN)
    print()

    for index, path in enumerate(files, start=1):
        relative = path.relative_to(DATA_DIR)

        try:
            midi = pretty_midi.PrettyMIDI(str(path))
            instruments_with_notes = [
                instrument for instrument in midi.instruments if instrument.notes
            ]

            # Lolos jika setidaknya ada satu track berisi nada yang monofonik.
            monophonic_tracks = [
                instrument for instrument in instruments_with_notes
                if not has_overlap_in_track(instrument)
            ]

            if not midi.instruments or not any(
                instrument.notes for instrument in midi.instruments
            ):
                status = "EMPTY_NO_NOTES"
                reason = "Tidak ada nada"
                exclude = True
            elif not monophonic_tracks:
                status = "EXCLUDE_NO_MONOPHONIC_TRACK"
                reason = "Tidak ditemukan track bernada yang monofonik"
                exclude = True
            else:
                status = "KEEP"
                reason = "Memiliki setidaknya satu track bernada yang monofonik"
                exclude = False

            report.append({
                "file": str(relative),
                "status": status,
                "instrument_count": len(midi.instruments),
                "tracks_with_notes": len(instruments_with_notes),
                "monophonic_tracks_with_notes": len(monophonic_tracks),
                "reason": reason,
                "action": "WOULD_MOVE_TO_QUARANTINE" if exclude and DRY_RUN
                          else "MOVE_TO_QUARANTINE" if exclude
                          else "KEEP",
                "error": "",
            })

            if exclude:
                to_exclude.append(path)

        except Exception as error:
            # File yang gagal dibaca tidak dipindahkan otomatis; perlu ditinjau.
            report.append({
                "file": str(relative),
                "status": "READ_ERROR_REVIEW",
                "instrument_count": "",
                "tracks_with_notes": "",
                "monophonic_tracks_with_notes": "",
                "reason": "Gagal dibaca; tidak dipindahkan otomatis",
                "action": "MANUAL_REVIEW",
                "error": str(error),
            })

        if index % 100 == 0 or index == len(files):
            print(f"Diperiksa: {index}/{len(files)}")

    with open(REPORT_CSV, "w", newline="", encoding="utf-8-sig") as file:
        fields = [
            "file", "status", "instrument_count", "tracks_with_notes",
            "monophonic_tracks_with_notes", "reason", "action", "error"
        ]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(report)

    print("\n" + "=" * 68)
    print("RINGKASAN")
    print("=" * 68)
    print(f"Total file diperiksa              : {len(files)}")
    print(f"File yang akan dikecualikan       : {len(to_exclude)}")
    print(f"File yang dipertahankan           : {sum(row['status'] == 'KEEP' for row in report)}")
    print(f"File gagal dibaca (perlu ditinjau): {sum(row['status'] == 'READ_ERROR_REVIEW' for row in report)}")
    print(f"Laporan                          : {REPORT_CSV}")

    if DRY_RUN:
        print("\nMODE AMAN AKTIF: tidak ada file yang dipindahkan atau dihapus.")
        print("Periksa excluded_midi_report.csv terlebih dahulu.")
        print("Jika daftar sudah benar, ubah DRY_RUN = False lalu jalankan lagi.")
        return

    QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)
    moved = 0

    for path in to_exclude:
        # Pertahankan struktur subfolder dan hindari menimpa nama yang sama.
        relative = path.relative_to(DATA_DIR)
        destination = QUARANTINE_DIR / relative
        destination.parent.mkdir(parents=True, exist_ok=True)

        if destination.exists():
            print(f"LEWATI (tujuan sudah ada): {relative}")
            continue

        shutil.move(str(path), str(destination))
        moved += 1

    print(f"\nFile dipindahkan ke karantina: {moved}")
    print(f"Lokasi karantina: {QUARANTINE_DIR}")
    print("File tidak dihapus permanen; bisa dipulihkan dari folder karantina.")


if __name__ == "__main__":
    main()
