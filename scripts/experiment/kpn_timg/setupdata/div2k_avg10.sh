#!/bin/bash

# NOTE: I/O can be pretty slow, so it is usually a good idea to be conservative
# with how many jobs are spawned so that we don't get stuck eventually
# It is better to have more jobs for scripts which have more computation going
# on in them, like create_burst_from_image

TRAIN_DATASET="div2k_linear_train"
VAL_DATASET="div2k_linear_val"
FILELIST_DIR="filelists"
METADATA_DIR="generated/metadata/kpn_timg/avg10"
GENDATA_DIR="generated/data/kpn_timg/avg10"
GENCMDS_DIR="generated/cmds/kpn_timg/avg10"

mkdir -p "${METADATA_DIR}"
mkdir -p "${GENDATA_DIR}"
mkdir -p "${GENCMDS_DIR}"

echo "create_metadata"
python -m scripts.base.kpn_timg.create_metadata \
	"${FILELIST_DIR}/${VAL_DATASET}_imgs.list" \
	"${METADATA_DIR}/${VAL_DATASET}.json" \
	--orig-data-dir ./generated/data \
	--gen-data-dir "${GENDATA_DIR}/${VAL_DATASET}" \
	--spatial-downscale 4 \
	--num-timgs 1 \
	--tmin 1e-3 \
	--tmax 1e7 \
	--max-photon-rate 1.0 \
	--num-avg 10 \
	--avg-fn GM

rm -i "${GENCMDS_DIR}/${VAL_DATASET}_mkdir.cmdlist"
rm -i "${GENCMDS_DIR}/${VAL_DATASET}_cp_orig.cmdlist"
rm -i "${GENCMDS_DIR}/${VAL_DATASET}_convert_to_float_img.cmdlist"
rm -i "${GENCMDS_DIR}/${VAL_DATASET}_spad_convert.cmdlist"
rm -i "${GENCMDS_DIR}/${VAL_DATASET}_create_timg_from_float_img.cmdlist"

echo "gen_mkdir"
python -m scripts.gencmds.kpn_timg.gen_mkdir \
	"${METADATA_DIR}/${VAL_DATASET}.json" \
	"${GENCMDS_DIR}/${VAL_DATASET}_mkdir.cmdlist"
echo "gen_cp_orig"
python -m scripts.gencmds.kpn_timg.gen_cp_orig \
	"${METADATA_DIR}/${VAL_DATASET}.json" \
	"${GENCMDS_DIR}/${VAL_DATASET}_cp_orig.cmdlist"
echo "gen_convert_to_float_img"
python -m scripts.gencmds.kpn_timg.gen_convert_to_float_img \
	"${METADATA_DIR}/${VAL_DATASET}.json" \
	"${GENCMDS_DIR}/${VAL_DATASET}_convert_to_float_img.cmdlist"
echo "gen_spad_convert"
python -m scripts.gencmds.kpn_timg.gen_spad_convert \
	"${METADATA_DIR}/${VAL_DATASET}.json" \
	"${GENCMDS_DIR}/${VAL_DATASET}_spad_convert.cmdlist"
echo "gen_create_timg_from_float_img"
python -m scripts.gencmds.kpn_timg.gen_create_timg_from_float_img \
	"${METADATA_DIR}/${VAL_DATASET}.json" \
	"${GENCMDS_DIR}/${VAL_DATASET}_create_timg_from_float_img.cmdlist"
echo "exec_mkdir"
cat "${GENCMDS_DIR}/${VAL_DATASET}_mkdir.cmdlist" | parallel --jobs 1
echo "exec_cp_orig"
cat "${GENCMDS_DIR}/${VAL_DATASET}_cp_orig.cmdlist" | parallel --jobs 20
echo "exec_convert_to_float_img"
cat "${GENCMDS_DIR}/${VAL_DATASET}_convert_to_float_img.cmdlist" | parallel --jobs 2
echo "exec_spad_convert"
cat "${GENCMDS_DIR}/${VAL_DATASET}_spad_convert.cmdlist" | parallel --jobs 2
echo "exec_create_timg_from_float_img"
cat "${GENCMDS_DIR}/${VAL_DATASET}_create_timg_from_float_img.cmdlist" | parallel --jobs 8

echo "create_metadata"
python -m scripts.base.kpn_timg.create_metadata \
	"${FILELIST_DIR}/${TRAIN_DATASET}_imgs.list" \
	"${METADATA_DIR}/${TRAIN_DATASET}.json" \
	--orig-data-dir ./generated/data \
	--gen-data-dir "${GENDATA_DIR}/${TRAIN_DATASET}" \
	--spatial-downscale 4 \
	--num-timgs 4 \
	--tmin 1e-3 \
	--tmax 1e7 \
	--max-photon-rate 1.0 \
	--num-avg 10 \
	--avg-fn GM

rm -i "${GENCMDS_DIR}/${TRAIN_DATASET}_mkdir.cmdlist"
rm -i "${GENCMDS_DIR}/${TRAIN_DATASET}_cp_orig.cmdlist"
rm -i "${GENCMDS_DIR}/${TRAIN_DATASET}_convert_to_float_img.cmdlist"
rm -i "${GENCMDS_DIR}/${TRAIN_DATASET}_spad_convert.cmdlist"
rm -i "${GENCMDS_DIR}/${TRAIN_DATASET}_create_timg_from_float_img.cmdlist"

echo "gen_mkdir"
python -m scripts.gencmds.kpn_timg.gen_mkdir \
	"${METADATA_DIR}/${TRAIN_DATASET}.json" \
	"${GENCMDS_DIR}/${TRAIN_DATASET}_mkdir.cmdlist"
echo "gen_cp_orig"
python -m scripts.gencmds.kpn_timg.gen_cp_orig \
	"${METADATA_DIR}/${TRAIN_DATASET}.json" \
	"${GENCMDS_DIR}/${TRAIN_DATASET}_cp_orig.cmdlist"
echo "gen_convert_to_float_img"
python -m scripts.gencmds.kpn_timg.gen_convert_to_float_img \
	"${METADATA_DIR}/${TRAIN_DATASET}.json" \
	"${GENCMDS_DIR}/${TRAIN_DATASET}_convert_to_float_img.cmdlist"
echo "gen_spad_convert"
python -m scripts.gencmds.kpn_timg.gen_spad_convert \
	"${METADATA_DIR}/${TRAIN_DATASET}.json" \
	"${GENCMDS_DIR}/${TRAIN_DATASET}_spad_convert.cmdlist"
echo "gen_create_timg_from_float_img"
python -m scripts.gencmds.kpn_timg.gen_create_timg_from_float_img \
	"${METADATA_DIR}/${TRAIN_DATASET}.json" \
	"${GENCMDS_DIR}/${TRAIN_DATASET}_create_timg_from_float_img.cmdlist"
echo "exec_mkdir"
cat "${GENCMDS_DIR}/${TRAIN_DATASET}_mkdir.cmdlist" | parallel --jobs 1
echo "exec_cp_orig"
cat "${GENCMDS_DIR}/${TRAIN_DATASET}_cp_orig.cmdlist" | parallel --jobs 20
echo "exec_convert_to_float_img"
cat "${GENCMDS_DIR}/${TRAIN_DATASET}_convert_to_float_img.cmdlist" | parallel --jobs 2
echo "exec_spad_convert"
cat "${GENCMDS_DIR}/${TRAIN_DATASET}_spad_convert.cmdlist" | parallel --jobs 2
echo "exec_create_timg_from_float_img"
cat "${GENCMDS_DIR}/${TRAIN_DATASET}_create_timg_from_float_img.cmdlist" | parallel --jobs 2

