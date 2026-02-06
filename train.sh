#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR"
export OMP_NUM_THREADS=1

# Data root
DATA_ROOT="/root/autodl-tmp/kaggle-happywhale-1st-place-solution-charmq/happywhale_data"
PREP_ROOT_TRAIN="${DATA_ROOT}/preprocessed_768/preprocessed_768"
PREP_ROOT_TEST="${DATA_ROOT}/preprocessed_768/test_images"

# Source weights (will be normalized inside dataset)
P_ORIGINAL=${P_ORIGINAL:-0.10}
P_FULLBODY_YOURS=${P_FULLBODY_YOURS:-0.30}
P_FULLBODY_CHARMQ=${P_FULLBODY_CHARMQ:-0.30}
P_BACKFIN_YOURS=${P_BACKFIN_YOURS:-0.20}
P_BACKFIN_CHARMQ=${P_BACKFIN_CHARMQ:-0.10}

TEST_SOURCE=${TEST_SOURCE:-original}

# -------------------------
# 1) Smoke test: B0 + low res + 200 steps
# -------------------------
python -m run.train \
  out_dir=/root/autodl-tmp/hw_runs/smoke_b0_384_200steps \
  dataset.type=happy_whale \
  dataset.phase=train \
  dataset.use_preprocessed=true \
  dataset.preprocessed_root="$PREP_ROOT_TRAIN" \
  dataset.preprocessed_test_root="$PREP_ROOT_TEST" \
  dataset.train_source_weights.original="$P_ORIGINAL" \
  dataset.train_source_weights.fullbody_yours="$P_FULLBODY_YOURS" \
  dataset.train_source_weights.fullbody_charmq="$P_FULLBODY_CHARMQ" \
  dataset.train_source_weights.backfin_yours="$P_BACKFIN_YOURS" \
  dataset.train_source_weights.backfin_charmq="$P_BACKFIN_CHARMQ" \
  dataset.test_source="$TEST_SOURCE" \
  preprocessing.h_resize_to=384 \
  preprocessing.w_resize_to=384 \
  model.base_model=tf_efficientnet_b0 \
  training.num_gpus=1 \
  training.epoch=1 \
  training.max_steps=200 \
  training.batch_size=32 \
  training.batch_size_test=64 \
  training.num_workers=16 \
  training.use_amp=false \
  training.accumulate_grad_batches=1 \
  training.gradient_clip_val=0.5 \
  training.freeze_bn_stats=true

# -------------------------
# 2) Full training: EffV2-M + 768 + acc=2
# -------------------------
python -m run.train \
  out_dir=/root/autodl-tmp/hw_runs/train_effv2m_768_acc2 \
  dataset.type=happy_whale \
  dataset.phase=train \
  dataset.use_preprocessed=true \
  dataset.preprocessed_root="$PREP_ROOT_TRAIN" \
  dataset.preprocessed_test_root="$PREP_ROOT_TEST" \
  dataset.train_source_weights.original="$P_ORIGINAL" \
  dataset.train_source_weights.fullbody_yours="$P_FULLBODY_YOURS" \
  dataset.train_source_weights.fullbody_charmq="$P_FULLBODY_CHARMQ" \
  dataset.train_source_weights.backfin_yours="$P_BACKFIN_YOURS" \
  dataset.train_source_weights.backfin_charmq="$P_BACKFIN_CHARMQ" \
  dataset.test_source="$TEST_SOURCE" \
  preprocessing.h_resize_to=768 \
  preprocessing.w_resize_to=768 \
  model.base_model=tf_efficientnetv2_m \
  training.num_gpus=1 \
  training.epoch=20 \
  training.batch_size=12 \
  training.batch_size_test=64 \
  training.num_workers=16 \
  training.use_amp=false \
  training.accumulate_grad_batches=2 \
  training.gradient_clip_val=0.5 \
  training.freeze_bn_stats=true
