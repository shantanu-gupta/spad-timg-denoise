#!/bin/bash

TRAIN_DATASET="div2k_linear_train"
DATA_NAME="trained_on_${TRAIN_DATASET}_logtimg"

OUTPUT_DIR="./generated/output/kpn_timg/avg1/${DATA_NAME}"
mkdir -p "${OUTPUT_DIR}"

CKPT="./generated/models/ckpt/kpn_timg/avg1/${DATA_NAME}/state_final.tar"
EXAMPLE_IMG="./generated/data/DIV2K_linear/DIV2K_valid_HR/0812.png"

FLOAT_IMG="${OUTPUT_DIR}/grayscale.tiff"
LOGTIMG="${OUTPUT_DIR}/logtimg.npy"
DENOISED_LOGTIMG="${OUTPUT_DIR}/logtimg_kpn.npy"
FINAL_OUTPUT="${OUTPUT_DIR}/final_output.tiff"

python -m scripts.base.convert_to_float_img \
	"${EXAMPLE_IMG}" "${FLOAT_IMG}"
python -m scripts.base.create_timg_from_float_img \
	"${FLOAT_IMG}" \
	--logtimg "${LOGTIMG}" \
	--tmin 1e-3 \
	--tmax 1e6
python -m scripts.base.kpn_timg.run_logtimg \
	"${LOGTIMG}" \
	"${DENOISED_LOGTIMG}" \
	--model "${CKPT}" \
	--crop-granularity 128
python -m scripts.base.spad_convert \
	"${DENOISED_LOGTIMG}" \
	"${FINAL_OUTPUT}" \
	--mapping logtimg-to-radiance
