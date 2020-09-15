#!/bin/bash

TRAIN_DATASET="div2k_linear_train"
VAL_DATASET="div2k_linear_val"
METADATA_DIR="generated/metadata/kpn_timg/avg2"
LOG_DIR="generated/models/logs/kpn_timg/avg2/trained_on_${TRAIN_DATASET}_logtimg"
CKPT_DIR="generated/models/ckpt/kpn_timg/avg2/trained_on_${TRAIN_DATASET}_logtimg"
VAL_DIR="generated/models/val/kpn_timg/avg2/trained_on_${TRAIN_DATASET}_logtimg"

python -m scripts.base.kpn_timg.train_logtimg \
	"${METADATA_DIR}/${TRAIN_DATASET}.json" \
	"${METADATA_DIR}/${VAL_DATASET}.json" \
	--Kout 5 \
	--crop-size 128 \
	--dloader-workers 4 \
	--epochs 3 \
	--batch-size 5 \
	--lr 1e-4 \
	--log-dir "${LOG_DIR}" \
	--checkpoint-dir "${CKPT_DIR}" \
	--val-output-dir "${VAL_DIR}"

