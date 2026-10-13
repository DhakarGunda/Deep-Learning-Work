# St. Jude body composition — code inventory

What was recovered from working history, and what is still only on the cluster.

## Recovered in full

**`bcomp_array.sbatch`** — the production inference harness. A 20-way SLURM array
(scales to 40) that shards `out.csv` by contiguous row ranges, runs the body
composition tool one series at a time, and writes to a per-task results CSV.
Three pieces of real engineering in it:

- **Per-task output files.** The original design had all tasks appending to one
  shared `CTMeasurementsD_ENew.csv`. Concurrent writes from 10 tasks tore rows
  mid-line and corrupted the file (`EOF inside string` on read). Per-shard
  files removed the shared writer entirely.
- **Global resume.** Each task reads *every* `part*.csv` and skips any series
  already marked `SUCCESS` anywhere — not just its own file. This is what makes
  the part count safe to change between passes, so each resubmission strictly
  chips away at what's left instead of restarting.
- **Per-series timeout with process-group kill.** `subprocess.Popen(...,
  start_new_session=True)` plus `os.killpg` on SIGTERM then SIGKILL. A plain
  `proc.kill()` would orphan the nnU-Net worker processes, which hold the GPU.

**`nnunet_v2_pipeline.sh`** — the training/inference command sequence, from MD.ai
annotation JSON through fingerprinting, planning, preprocessing, 5-fold training,
configuration selection, and prediction. Includes the patient-grouped
`splits_final.json` step that prevents slice-level leakage across folds.

## Reconstructable from history but not recovered verbatim

- `mdai_bodycomp2nii_merged.py` — MD.ai JSON → nnU-Net dataset converter. Schema
  abstraction, five-gate case validation, whitespace-normalized label matching,
  `--allow-single-level`, self-diagnosing `rejected.csv`.
- `split_missing.py` / `merge_parts.py` / `merge_D_ENew.py` — round-robin
  sharding and priority dedup on merge (`SUCCESS=2 > SKIPPED=1 > blank=0`),
  with a coverage check against `out.csv`.
- `yr_array.sbatch` — the YR-dataset variant. Converts the LSF
  `SIGUSR2`/`bsub < $SELF` auto-resume idiom to SLURM-native
  `--signal=B:USR1@600` + `--requeue` + a `scontrol requeue` trap, with the main
  process backgrounded so the trap fires promptly.
- Cohort-split tooling — `PatientAge` DICOM tag parsing with Y/M/W/D unit
  handling and a `StudyDate − PatientBirthDate` fallback; L1/L3 mask presence
  verified by regex with a negative lookahead (`vertebrae.*?l0*1(?!\d)`) so L12
  and L13 don't false-match.

## Only on the cluster

- The TotalSegmentator custom fork (env-var-controlled vertebral weights via
  `USE_DEFAULT_TSEG_VERTEBRAL_MODEL` / `TSEG_VERTEBRAL_MODEL_PATH`).
- The CCC / Wilcoxon / bootstrap analysis scripts and the forest-plot code.
- The NIfTI→DICOM-SEG (highdicom), NIfTI→RTSTRUCT (rt-utils), and
  NIfTI→MD.ai uploader converters.
- The custom OHIF viewer extension.
- Model weights, all datasets, all results.

