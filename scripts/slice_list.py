# scripts/make_slice_list.py
import argparse
import csv
import os
import re
import sys
import numpy as np

def build_id_to_file(data_dir):
    """
    Map study_id -> (path, thickness_or_None) from files named either:
      - "<id>.npy"           -> thickness None (use default later)
      - "<id>_<thickness>.npy" (e.g., 1.25) -> parsed thickness (float)
    """
    id2file = {}
    re_with_thicc = re.compile(r'^(\d+)_(\d\.\d\d)\.npy$')
    re_no_thicc   = re.compile(r'^(\d+)\.npy$')

    for base, _, files in os.walk(data_dir):
        for f in files:
            m = re_with_thicc.match(f)
            if m:
                sid = int(m.group(1))
                th  = float(m.group(2))
                id2file[sid] = (os.path.join(base, f), th)
                continue
            m = re_no_thicc.match(f)
            if m:
                sid = int(m.group(1))
                id2file.setdefault(sid, (os.path.join(base, f), None))
    return id2file

def load_num_slices(npy_path):
    # Safe on memory even for large volumes
    arr = np.load(npy_path, mmap_mode="r")
    # Assume [slices, H, W] ordering
    return int(arr.shape[0])

def parse_labels_csv(labels_csv, id_col, label_col, phase_col, split_col):
    """
    Return dict: id -> (label:int, phase:str, split:str)
    Your example CSV rows look like:
      id, other_id, label, phase, split
    Default columns: 0, 2, 3, 4
    """
    id2meta = {}
    with open(labels_csv, newline="") as fh:
        reader = csv.reader(fh)
        for row in reader:
            if not row:
                continue
            # Skip header lines that don't start with an integer id
            try:
                sid = int(row[id_col])
            except Exception:
                continue
            label = row[label_col].strip()
            label = int(label) if label != "" else 0
            phase = (row[phase_col].strip() if phase_col is not None and row[phase_col] is not None else "") or "unknown"
            split = (row[split_col].strip() if split_col is not None and row[split_col] is not None else "") or "train"
            id2meta[sid] = (label, phase, split)
    return id2meta

def main():
    ap = argparse.ArgumentParser(description="Generate slice_list txt from labels.csv and .npy volumes.")
    ap.add_argument("--data_dir", required=True, help="Directory containing *.npy volumes.")
    ap.add_argument("--labels_csv", required=True, help="CSV with id, label, phase, split (your labels.csv).")
    ap.add_argument("--out_txt", required=True, help="Where to write the slice_list txt.")
    ap.add_argument("--default_thickness", type=float, default=1.25,
                    help="Thickness to assume if filename lacks thickness (e.g., '1234.npy').")
    # Column indices for flexible CSV layouts
    ap.add_argument("--id_col", type=int, default=0)
    ap.add_argument("--label_col", type=int, default=2)
    ap.add_argument("--phase_col", type=int, default=3)
    ap.add_argument("--split_col", type=int, default=4)
    # Optional: include volumes not listed in CSV as assumed negatives
    ap.add_argument("--include_unlisted", action="store_true",
                    help="Also include .npy files that aren't in the CSV (as assumed negatives).")
    ap.add_argument("--assumed_label", type=int, default=0,
                    help="Label to use for unlisted volumes if --include_unlisted is set.")
    ap.add_argument("--assumed_phase", type=str, default="unknown")
    ap.add_argument("--assumed_split", type=str, default="train")
    args = ap.parse_args()

    id2file = build_id_to_file(args.data_dir)
    if not id2file:
        print(f"No .npy volumes found under {args.data_dir}", file=sys.stderr)
        sys.exit(1)

    id2meta = parse_labels_csv(args.labels_csv, args.id_col, args.label_col, args.phase_col, args.split_col)

    lines = []
    n_ok, n_missing_file, n_missing_csv, n_total = 0, 0, 0, 0

    # First: entries present in CSV
    for sid, (label, phase, split) in id2meta.items():
        n_total += 1
        fi = id2file.get(sid)
        if not fi:
            n_missing_file += 1
            print(f"[WARN] No .npy found for id {sid}; skipping")
            continue
        npy_path, th = fi
        thickness = th if th is not None else args.default_thickness
        try:
            num_slices = load_num_slices(npy_path)
        except Exception as e:
            print(f"[WARN] Failed to load {npy_path}: {e}; skipping")
            continue
        # No per-slice labels -> empty RHS after colon
        lines.append(f"{sid},{thickness},{label},{num_slices},{phase},{split}: ")
        n_ok += 1

    # Optional: include volumes that exist but aren't in CSV
    if args.include_unlisted:
        for sid, (npy_path, th) in id2file.items():
            if sid in id2meta:
                continue
            n_total += 1
            thickness = th if th is not None else args.default_thickness
            try:
                num_slices = load_num_slices(npy_path)
            except Exception as e:
                print(f"[WARN] Failed to load {npy_path}: {e}; skipping")
                continue
            lines.append(f"{sid},{thickness},{args.assumed_label},{num_slices},{args.assumed_phase},{args.assumed_split}: ")
            n_ok += 1
            n_missing_csv += 1

    # Stable ordering helps diffs/debugging
    lines.sort(key=lambda s: int(s.split(",")[0]))

    os.makedirs(os.path.dirname(args.out_txt), exist_ok=True)
    with open(args.out_txt, "w") as fh:
        fh.write("\n".join(lines))

    print(f"Wrote {len(lines)} studies to {args.out_txt}")
    print(f"Summary: ok={n_ok}, missing_file={n_missing_file}, added_unlisted={n_missing_csv}, total_seen={n_total}")

if __name__ == "__main__":
    main()

