#!/bin/bash
# nnU-Net v2 end-to-end: annotations -> trained model -> predictions.
# Body composition segmentation (L1/L3 tissue compartments), St. Jude, summer 2026.

# ---- 0. environment (in ~/.bashrc or the sbatch script) ----
export nnUNet_raw=/research/groups/mukhegrp/.../nnUNet_raw
export nnUNet_preprocessed=/research/groups/mukhegrp/.../nnUNet_preprocessed
export nnUNet_results=/research/groups/mukhegrp/.../nnUNet_results
export nnUNet_n_proc_DA=8            # data-augmentation workers; drop to 4 if RAM-bound

# ---- 1. annotations -> raw dataset (once per source dataset) ----
# mdai_bodycomp2nii_merged.py: MD.ai annotation JSON -> nnU-Net raw layout.
# Schema abstraction, five-gate case validation, whitespace-normalized label
# matching, --allow-single-level, self-diagnosing rejected.csv.
python mdai_bodycomp2nii_merged.py \
  --json mdai_..._D_X8J5m3_....json \
  --dicom-root .../training_data/D_X8J5m3/images \
  --schema nnunet --max-leak 50 \
  --out-dir $nnUNet_raw/Dataset511_UAB_Ped

# ---- 2. pool the sources ----
python merge_nnunet_datasets.py \
  --src $nnUNet_raw/Dataset511_UAB_Ped --src $nnUNet_raw/Dataset512_... \
  --out-dir $nnUNet_raw/Dataset520_BodyCompMerged \
  --dataset-name Dataset520_BodyCompMerged

# ---- 3. fingerprint + plan + preprocess (2d only; cases are single-slice) ----
nnUNetv2_plan_and_preprocess -d 520 -c 2d --verify_dataset_integrity

# ---- 4. install patient-grouped folds (AFTER step 3 creates the dir) ----
# Patient-level grouping prevents slices from one patient landing in both
# train and val, which would leak and inflate validation Dice.
cp splits_final.json $nnUNet_preprocessed/Dataset520_BodyCompMerged/splits_final.json

# ---- 5. train (one GPU per fold; run as a 5-way SLURM array) ----
nnUNetv2_train 520 2d $SLURM_ARRAY_TASK_ID --npz

# ---- 6. pick config / postprocessing ----
nnUNetv2_find_best_configuration 520 -c 2d --disable_ensembling

# ---- 7. inference ----
nnUNetv2_predict -i $nnUNet_raw/Dataset520_BodyCompMerged/imagesTs \
                 -o /path/preds -d 520 -c 2d -f 0 1 2 3 4
nnUNetv2_apply_postprocessing -i /path/preds -o /path/preds_pp
