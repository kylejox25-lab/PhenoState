#!/usr/bin/env bash
set -euo pipefail

# Export existing frozen checkpoints only; never start or resume training.
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

MDS_ROOT="${RXRX3_MDS_ROOT:?Set RXRX3_MDS_ROOT to the complete MDS directory}"
RUN_ROOT="${PHENOSTATE_COMPARISON_ROOT:-runs/comparison_patch8_50epochs}"
EMBED_ROOT="${PHENOSTATE_EMBEDDING_ROOT:?Set PHENOSTATE_EMBEDDING_ROOT to the embedding output directory}"
PYTHON_COMMAND="${PHENOSTATE_PYTHON:-python}"
BATCH_SIZE="${PHENOSTATE_BATCH_SIZE:-1}"
NUM_WORKERS="${PHENOSTATE_NUM_WORKERS:-4}"
read -r -a MODELS <<< "${PHENOSTATE_MODELS:-mae ca_mae diffmae cnn_mae}"
read -r -a SPLITS <<< "${PHENOSTATE_SPLITS:-train val test}"

if [[ -z "${MODELS[*]-}" || -z "${SPLITS[*]-}" ]]; then
  echo "Model and split lists must not be empty" >&2
  exit 1
fi
for SPLIT in "${SPLITS[@]}"; do
  case "$SPLIT" in
    train|val|test) ;;
    *) echo "Invalid split: $SPLIT" >&2; exit 1 ;;
  esac
done

# Check every checkpoint before starting expensive exports.
CHECKPOINTS=()
for MODEL in "${MODELS[@]}"; do
  CHECKPOINT="$RUN_ROOT/$MODEL/checkpoints/best.ckpt"
  if [[ ! -f "$CHECKPOINT" ]]; then
    CHECKPOINT="$RUN_ROOT/$MODEL/checkpoints/last.ckpt"
  fi
  if [[ ! -f "$CHECKPOINT" ]]; then
    echo "Missing checkpoint for $MODEL under $RUN_ROOT/$MODEL" >&2
    exit 1
  fi
  CHECKPOINTS+=("$CHECKPOINT")
done

for INDEX in "${!MODELS[@]}"; do
  MODEL="${MODELS[$INDEX]}"
  echo "Model: $MODEL; checkpoint: ${CHECKPOINTS[$INDEX]}"
  for SPLIT in "${SPLITS[@]}"; do
    "$PYTHON_COMMAND" -m phenostate.infer \
      --checkpoint "${CHECKPOINTS[$INDEX]}" \
      --mds-root "$MDS_ROOT" --split "$SPLIT" \
      --output "$EMBED_ROOT/$MODEL/embedding_${SPLIT}.parquet" \
      --batch-size "$BATCH_SIZE" --num-workers "$NUM_WORKERS" \
      --skip-existing
  done
done
