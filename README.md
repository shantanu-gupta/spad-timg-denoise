# spad-timg-denoise

The Python implementations are in the `src/py/` directory. The scripts in `scripts/base/` provide an interface to that code so that you can run it in a command-line. The code in `scripts/gencmds/` uses these scripts, in turn, to generate and/or pre-process all the training data for the KPN.

## Setting up simulated training data
See `scripts/experiment/kpn_timg/setupdata/`. They set up a training dataset of linear radiance images using the DIV2K dataset.

## Training the KPN
See `scripts/experiment/kpn_timg/train/`.

## Running a trained KPN denoiser on a simulated example image
See the files in `scripts/experiment/kpn_timg/run/`. They take you from a PNG file representing (linear) image radiances, to a simulated log-timestamp image, to denoising it through the KPN, then mapping it back to a radiance value. You need to train a model using the previous steps (or use the trained models within this repo in `generated/models/ckpt/kpn_timg/`) for this to work.
