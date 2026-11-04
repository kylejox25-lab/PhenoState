#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

EMB="${PHENOSTATE_EMBEDDING_ROOT:-../data/embeddings}"
DATA="${RXRX3_CORE_ROOT:-/home/xkj2006/CellPainting/rxrx3_core}"
PUBLIC="${PHENOSTATE_PUBLIC_EMBEDDING_ROOT:-$DATA}"
ANNOTATIONS="${PHENOSTATE_EFAAR_ANNOTATIONS_DIR:-../data/EFAAR_benchmarking/efaar_benchmarking/benchmark_annotations}"
OUT="${PHENOSTATE_PAPER_RESULTS_DIR:-results/downstream_val_all/paper_7_models}"
PYTHON_COMMAND="${PHENOSTATE_PYTHON:-python}"
read -r -a SPLITS <<< "${PHENOSTATE_PAPER_SPLITS:-train val}"
if (( ${#SPLITS[@]} == 0 )); then
  printf 'Provide at least one split in PHENOSTATE_PAPER_SPLITS\n' >&2
  exit 1
fi
for SPLIT in "${SPLITS[@]}"; do
  case "$SPLIT" in
    train|val|test) ;;
    *) printf 'Invalid split: %s\n' "$SPLIT" >&2; exit 1 ;;
  esac
done

ARGS=()
REQUIRED=("$DATA/metadata_rxrx3_core.csv")
for FILE in CORUM.txt HuMAP.txt Reactome.txt StringDB.txt compound_gene_interactions.csv; do
  REQUIRED+=("$ANNOTATIONS/$FILE")
done

for SPEC in mae:mae_patch8_head8 ca_mae:ca_mae diffmae:diffmae cnn_mae:cnn_mae; do
  ARGS+=(--model "${SPEC%%:*}")
  for SPLIT in "${SPLITS[@]}"; do
    FILE="$EMB/${SPEC#*:}/embedding_${SPLIT}.parquet"
    ARGS+=("$FILE")
    REQUIRED+=("$FILE")
  done
done

for SPEC in \
  OpenPhenom:OpenPhenom_rxrx3_core_embeddings.parquet \
  MAE-L-8:MAE-L_8_RPI-93M_rxrx3_core_embeddings.parquet \
  MAE-G-8:MAE-G_8_PP-16M_trimmed_rxrx3_core_embeddings.parquet; do
  FILE="$PUBLIC/${SPEC#*:}"
  ARGS+=(--embedding "${SPEC%%:*}" "$FILE")
  REQUIRED+=("$FILE")
done

MISSING=0
for FILE in "${REQUIRED[@]}"; do
  if [[ ! -f "$FILE" ]]; then
    printf 'Missing input: %s\n' "$FILE" >&2
    MISSING=1
  fi
done
if (( MISSING )); then
  exit 1
fi

exec "$PYTHON_COMMAND" -m phenostate.paper_benchmarks \
  "${ARGS[@]}" \
  --metadata "$DATA/metadata_rxrx3_core.csv" \
  --annotations-dir "$ANNOTATIONS" \
  --output-dir "$OUT" \
  "$@"
