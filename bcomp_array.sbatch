#!/bin/bash
#SBATCH --job-name=bcomp_20parts
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/research/groups/mukhegrp/home/common/BodyComposition/Analysis/SoumitroPersonal/bodycomptool/body-composition-tool/logs/20parts_%A_%a.out
#SBATCH --array=0-19            # <-- part count. To finish in one pass, use 0-39 and set NUM_PARTS=40.
set -euo pipefail

NUM_PARTS=20                    # MUST equal the --array size above.
SERIES_TIMEOUT_SEC=1200         # kill a single hung series after 20 min, then continue.

WORKDIR="/research/groups/mukhegrp/home/common/BodyComposition/Analysis/SoumitroPersonal/bodycomptool/body-composition-tool"
MAIN_CSV="$WORKDIR/out.csv"
CHUNK_CSV="$WORKDIR/metadata_chunk_${SLURM_ARRAY_TASK_ID}.csv"
RESULTS_CSV="$WORKDIR/CTMeasurementsD_ENew.part${SLURM_ARRAY_TASK_ID}.csv"

cd "$WORKDIR"
mkdir -p logs outputD_ENew

# --- 1. Split out.csv into this task's contiguous shard ----------------------
echo "Task $SLURM_ARRAY_TASK_ID / $NUM_PARTS -> Generating chunk CSV: $CHUNK_CSV"
python - "$MAIN_CSV" "$SLURM_ARRAY_TASK_ID" "$CHUNK_CSV" "$NUM_PARTS" <<'PY'
import csv, sys
main_csv, chunk_id, out_csv, num_parts = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4])
with open(main_csv, 'r', newline='') as f:
    reader = list(csv.DictReader(f))
    if not reader:
        sys.exit(0)
    headers = list(reader[0].keys())
total_rows = len(reader)
chunk_size = total_rows // num_parts
remainder  = total_rows % num_parts
start = chunk_id * chunk_size + min(chunk_id, remainder)
end   = start + chunk_size + (1 if chunk_id < remainder else 0)
chunk_data = reader[start:end]
with open(out_csv, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=headers)
    writer.writeheader()
    writer.writerows(chunk_data)
print(f"  rows {start}:{end}  ({len(chunk_data)} of {total_rows})")
PY

# --- 2. Environment ----------------------------------------------------------
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0
export nnUNet_n_proc_DA=0
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-6}
export nnUNet_results="/research/groups/mukhegrp/home/common/BodyComposition/Analysis/SoumitroPersonal/bodycomptool/body-composition-tool/body_composition_tool/inference/.models/nnUNet/nnUNet_results"
command -v dcm2niix >/dev/null || echo "[warn] dcm2niix not on PATH (fallback conversions will fail)"

# --- 3. Row-by-row with GLOBAL RESUME + per-series TIMEOUT -------------------
echo "Processing $CHUNK_CSV row-by-row -> $RESULTS_CSV (timeout ${SERIES_TIMEOUT_SEC}s/series)"
SERIES_TIMEOUT_SEC="$SERIES_TIMEOUT_SEC" \
python - "$CHUNK_CSV" "${TMPDIR:-/tmp}/single_row_${SLURM_ARRAY_TASK_ID}.csv" "$WORKDIR/outputD_ENew" "$RESULTS_CSV" <<'PY'
import csv, subprocess, sys, os, glob, signal, time
chunk_csv, tmp_csv, out_dir, res_csv = sys.argv[1:5]
task_id = os.environ.get('SLURM_ARRAY_TASK_ID', 'UNKNOWN')
timeout = int(os.environ.get('SERIES_TIMEOUT_SEC', '1200'))
workdir = os.path.dirname(res_csv)

def norm(s):
    return s.replace(' ', '').replace('_', '').lower()
def find_col(fieldnames, target):
    for c in (fieldnames or []):
        if norm(c) == target:
            return c
    return None

# --- GLOBAL resume: union of SUCCESS series across ALL part CSVs -------------
# (reading is safe even while other tasks append; a torn last row at worst
#  means one series gets re-attempted, which is harmless.)
done = set()
for pf in glob.glob(os.path.join(workdir, "CTMeasurementsD_ENew.part*.csv")):
    try:
        with open(pf, 'r', newline='') as f:
            r = csv.DictReader(f)
            ucol = find_col(r.fieldnames, 'seriesinstanceuid')
            scol = find_col(r.fieldnames, 'processingstatus')
            if not ucol:
                continue
            for row in r:
                if scol is None or row.get(scol) == 'SUCCESS':
                    done.add(row[ucol])
    except Exception as e:
        print(f"[Task {task_id}] warn: could not fully read {os.path.basename(pf)}: {e}", flush=True)
print(f"[Task {task_id}] resume: {len(done)} series already completed (all parts), will skip", flush=True)

with open(chunk_csv, 'r', newline='') as f:
    reader = list(csv.DictReader(f))
    if not reader:
        sys.exit(0)
    headers = list(reader[0].keys())
meta_uid = find_col(headers, 'seriesinstanceuid')
total = len(reader)

for idx, row in enumerate(reader, 1):
    uid = row.get(meta_uid) if meta_uid else None
    if uid and uid in done:
        print(f"[Task {task_id}] series {idx}/{total} already done -> skip", flush=True)
        continue

    print(f"\n[Task {task_id}] Processing series {idx}/{total}...", flush=True)
    with open(tmp_csv, 'w', newline='') as tf:
        w = csv.DictWriter(tf, fieldnames=headers)
        w.writeheader()
        w.writerow(row)

    # own process group so a hang can be killed tree-wide (nnU-Net workers too)
    proc = subprocess.Popen(
        ["python", "-m", "body_composition_tool",
         "--metadata_path", tmp_csv,
         "--output_dir",    out_dir,
         "--results_csv",   res_csv],
        start_new_session=True,
    )
    try:
        rc = proc.wait(timeout=timeout)
        if rc != 0:
            print(f"[ERROR] Tool failed on series {idx}/{total} (rc={rc}). Moving on.", flush=True)
    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT] series {idx}/{total} exceeded {timeout}s -> killing and moving on.", flush=True)
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(proc.pid, sig)
            except ProcessLookupError:
                break
            time.sleep(10)
PY

echo "Done with chunk $SLURM_ARRAY_TASK_ID"
