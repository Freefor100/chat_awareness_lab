#!/usr/bin/env bash
# Phase 3/4 正式组排队脚本（可断点续跑：单元格已存在则跳过）
# 用法: bash formal_run.sh [model1 model2 ...]   默认全部主模型
set -u
export CHATLAB_QUANT=int4_bnb
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODELS=${@:-qwen3.5-2b qwen3.5-4b ministral-3-3b-instruct}
SAMPLING="plain,A0,A1,A6,A9,A10,extract"
NOSTOP="plain,A9,A1"

for m in $MODELS; do
  echo "===== [$m] attack matrix (E5/E6/E7, n=100) $(date +%T) ====="
  CHATLAB_SAMPLING_SUBSET="$SAMPLING" CHATLAB_NOSTOP_SUBSET="$NOSTOP" \
    CHATLAB_MAX_NEW_TOKENS=192 \
    .venv/bin/python -m src.run_stage attack --models "$m" \
      --experiments E5 E6 E7 --sample-size formal 2>&1 | grep -v Warning | tail -1

  echo "===== [$m] reproduction (E1/E2/E4, n=100) $(date +%T) ====="
  CHATLAB_SAMPLING_SUBSET=none CHATLAB_NOSTOP_SUBSET="*" \
    CHATLAB_MAX_NEW_TOKENS=384 \
    .venv/bin/python -m src.run_stage attack --models "$m" \
      --experiments E1 E2 E4 --sample-size formal 2>&1 | grep -v Warning | tail -1
done

echo "===== E8: base pairs (0.8b-base / 2b-base, E5/E6 subset) $(date +%T) ====="
for m in qwen3.5-0.8b-base qwen3.5-2b-base; do
  CHATLAB_SAMPLING_SUBSET=none CHATLAB_NOSTOP_SUBSET=none \
    CHATLAB_MAX_NEW_TOKENS=192 \
    .venv/bin/python -m src.run_stage attack --models "$m" \
      --experiments E5 E6 --sample-size formal 2>&1 | grep -v Warning | tail -1
done
echo "ALL DONE $(date +%T)"
