#!/usr/bin/env bash
set -euo pipefail

# Run from the project root, after activating the existing project environment.
# All four runs use the same images, plate split, seed, and effective batch.
MDS_ROOT="${RXRX3_MDS_ROOT:?Set RXRX3_MDS_ROOT to a complete MDS conversion}"
RUN_ROOT="${PHENOSTATE_COMPARISON_ROOT:-runs/comparison_patch8_50epochs}"
BATCH_SIZE="${PHENOSTATE_BATCH_SIZE:-1}"
ACCUMULATION_STEPS="${PHENOSTATE_ACCUMULATION_STEPS:-64}"
NUM_WORKERS="${PHENOSTATE_NUM_WORKERS:-0}"

for MODEL in mae ca_mae diffmae cnn_mae; do
  CONFIG="configs/${MODEL}.json"
  if [[ "$MODEL" == "mae" ]]; then CONFIG="configs/mae_comparison.json"; fi
  RUN_DIR="$RUN_ROOT/$MODEL"
  RESUME_ARGS=()
  if [[ "${PHENOSTATE_RESUME:-0}" == "1" && -f "$RUN_DIR/checkpoints/last.ckpt" ]]; then
    RESUME_ARGS=(--resume "$RUN_DIR/checkpoints/last.ckpt")
  fi
  phenostate-train --mds-root "$MDS_ROOT" --output "$RUN_DIR" \
    --model-config "$CONFIG" --epochs 50 --batch-size "$BATCH_SIZE" \
    --accumulation-steps "$ACCUMULATION_STEPS" --num-workers "$NUM_WORKERS" --seed 2026 \
    --checkpoint-every 500 --log-every 50 "${RESUME_ARGS[@]}"

  # Reuse only exports whose source and completeness checks pass.
  for SPLIT in val test; do
    OUTPUT="$RUN_DIR/embedding_${SPLIT}.parquet"
    phenostate-embed --checkpoint "$RUN_DIR/checkpoints/best.ckpt" \
      --mds-root "$MDS_ROOT" --split "$SPLIT" --output "$OUTPUT" \
      --batch-size "$BATCH_SIZE" --skip-existing
  done
done
