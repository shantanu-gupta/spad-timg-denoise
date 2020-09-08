#!/bin/bash

ORIG_DATA_DIR="datasets"
ORIG_DIV2K_TRAIN_FILELIST="filelists/div2k_train_imgs.list"
ORIG_DIV2K_VAL_FILELIST="filelists/div2k_val_imgs.list"

GEN_DATA_DIR="generated/data"
LINEAR_DIV2K_TRAIN_FILELIST="filelists/div2k_linear_train_imgs.list"
LINEAR_DIV2K_VAL_FILELIST="filelists/div2k_linear_val_imgs.list"
sed 's/^DIV2K/DIV2K_linear/g' "${ORIG_DIV2K_TRAIN_FILELIST}" \
	> "${LINEAR_DIV2K_TRAIN_FILELIST}"
sed 's/^DIV2K/DIV2K_linear/g' "${ORIG_DIV2K_VAL_FILELIST}" \
	> "${LINEAR_DIV2K_VAL_FILELIST}"
mkdir -p "${GEN_DATA_DIR}/DIV2K_linear"
mkdir -p "${GEN_DATA_DIR}/DIV2K_linear/DIV2K_train_HR"
mkdir -p "${GEN_DATA_DIR}/DIV2K_linear/DIV2K_valid_HR"

GENCMDS_DIR="generated/cmds/linearize_div2k"
mkdir -p "${GENCMDS_DIR}"
paste "${ORIG_DIV2K_TRAIN_FILELIST}" "${LINEAR_DIV2K_TRAIN_FILELIST}" \
	| awk -v orig_data_dir="${ORIG_DATA_DIR}" \
		-v gen_data_dir="${GEN_DATA_DIR}" \
		'{print "convert " \
			orig_data_dir "/" $1 \
			" -depth 16 " \
			" -grayscale Rec709Luminance " \
			gen_data_dir "/"  $2 "[16]"}' \
	> "${GENCMDS_DIR}/div2k_train.cmdlist"

paste "${ORIG_DIV2K_VAL_FILELIST}" "${LINEAR_DIV2K_VAL_FILELIST}" \
	| awk -v orig_data_dir="${ORIG_DATA_DIR}" \
		-v gen_data_dir="${GEN_DATA_DIR}" \
		'{print "convert " \
			orig_data_dir "/" $1 \
			" -depth 16 " \
			" -grayscale Rec709Luminance " \
			gen_data_dir "/"  $2 "[16]"}' \
	> "${GENCMDS_DIR}/div2k_val.cmdlist"

cat "${GENCMDS_DIR}/div2k_train.cmdlist" | parallel --jobs 8
cat "${GENCMDS_DIR}/div2k_val.cmdlist" | parallel --jobs 8
