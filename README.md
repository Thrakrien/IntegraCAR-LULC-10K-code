# IntegraCAR-LULC-10K: Segmentation Training

This repository contains code to prepare masks and train land use and land cover (LULC) semantic segmentation models for the paper **“IntegraCAR-LULC-10K: A High-Resolution Optical Satellite Dataset for LULC Segmentation in the Brazilian Rural Environmental Registry”** (SIBGRAPI 2026). The public [IntegraCAR-LULC-500](https://huggingface.co/datasets/laicsiifes/IntegraCAR-LULC-500) dataset is a 500-image benchmark subset of the larger 10,000-tile collection described in the paper.

## Dataset and classes

IntegraCAR-LULC-500 contains 2048 × 2048 pixel tiles at a spatial resolution of 0.5 m/pixel from Espírito Santo, Brazil. The optical images and annotations are derived from GeoBases data for 2019–2020. The dataset includes 50 image-mask pairs per microregion: 300 for training, 100 for validation, and 100 for testing. On Hugging Face, images and masks are stored in separate splits (`satellite_train`, `mask_train`, `satellite_val`, `mask_val`, `satellite_test`, `mask_test`) and paired by the `filename` field. See the [dataset card](https://huggingface.co/datasets/laicsiifes/IntegraCAR-LULC-500) for data provenance, metadata, and annotation limitations.

**The published dataset uses a different class ID order from `convert_mask.py` and the legends in `train_code.py`:**

| Class | IntegraCAR-LULC-500 ID | ID after `convert_mask.py --classes 5` |
| --- | ---: | ---: |
| Vegetação | 0 | 1 |
| Agropastoril | 1 | 2 |
| Infraestrutura | 2 | 0 |
| Água | 3 | 4 |
| Macega | 4 | 3 |

The dataset card's usage example shows indexed masks with values from 0 to 4; it also mentions single-band and RGB mask representations. Check the values in your export. **Do not run `convert_mask.py` on masks that are already indexed.** To train with the class interpretation used by this code, remap their IDs according to the table before placing them in `masks_dir`. Training directly on the Hugging Face IDs still gives the model five classes, but the training visualizations will use incorrect class legends. The converter below is intended for the **original RGB masks** with the colors defined in `convert_mask.py`; a different RGB representation may use a different palette.

## 1. Install the environment with `uv`

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and run this command from the repository root:

```bash
uv sync --locked
```

The project requires Python 3.10 or newer (`.python-version` specifies 3.10). This command creates or updates `.venv` from `pyproject.toml` and `uv.lock`. Training uses PyTorch and selects CUDA when available, or CPU otherwise. With `encoder_weights: imagenet`, the first run also needs to download the encoder weights.

## 2. Prepare images and convert RGB masks

Training reads local GeoTIFF images and masks with **matching filenames**. For example:

```text
data/
├── satellite_images/
│   └── ES-...tif
├── masks_rgb/
│   └── ES-...tif
└── masks_indexed/       # created by the converter
    └── ES-...tif
```

Each split file (`data-segments/train_sample.txt`, `data-segments/validation_sample.txt`, or another split) contains one `.tif` filename per line. Every listed file must exist in `satellite_images/` and, after conversion, in `masks_indexed/`. The `*_sample.txt` files select a sample; `train.txt` and `validation.txt` are other split lists included in the repository. The image data is not included in this code repository.

**Run this conversion before training if the source masks are RGB:**

```bash
uv run --locked python convert_mask.py \
  --input-dir data/masks_rgb \
  --output-dir data/masks_indexed \
  --classes 5
```

The script reads `.tif` files directly inside `--input-dir`, maps known RGB colors to class IDs, and writes **single-band `uint8` GeoTIFF masks** to `--output-dir` while preserving filenames and spatial metadata. `--classes 5` matches `num_classes: 5` in the training configuration. `--classes 25` generates detailed IDs from 0 to 24, but this mode requires adapting the training code, including `num_classes`, `ignore_index`, and the legends. Use `--overwrite` only when you intend to replace existing output files. The input and output directories must differ; unknown RGB colors raise an error to prevent silently incorrect masks.

If the masks are already indexed in the desired class order, place them directly in the directory specified by `masks_dir` and continue with step 3.

## 3. Configure training

Create `configs/local.json` with your data paths. It overrides only the listed options and keeps the remaining defaults from `src/training_config.py`:

```json
{
  "train_txt": "data-segments/train_sample.txt",
  "val_txt": "data-segments/validation_sample.txt",
  "images_dir": "data/satellite_images",
  "masks_dir": "data/masks_indexed",
  "experiment_name": "integracar-unet-local"
}
```

Run all commands from the **repository root**. Adjust the JSON paths to your data; the absolute paths in the example YAML files and default configuration are specific to the machine where they were created. To use the full split lists, set `train_txt` to `data-segments/train.txt` and `val_txt` to `data-segments/validation.txt` after checking that all corresponding files are available.

By default, the code trains a U-Net with an EfficientNet-B5 encoder, five classes, 256 × 256 pixel patches, a 32-pixel step, and 100 epochs. This step produces many patches per image. Adjust `patch_step`, `batch_size`, `num_workers`, and `num_epochs` in the JSON to match your available memory and time. For other step sizes, `(image_size - patch_size)` must be divisible by both `patch_step` and `inference_stride`.

## 4. Run training with `nohup`

After preparing the masks and JSON file, run:

```bash
nohup uv run --locked python -u train_code.py --config configs/local.json \
  > train.log 2>&1 < /dev/null &
echo $!
```

`nohup` lets the process continue after the terminal closes; `&` runs it in the background. `python -u` sends unbuffered output to `train.log`, and `echo $!` prints the process ID. To follow training:

```bash
tail -f train.log
```

The script writes each run to `experiments/<experiment_name>_<timestamp>/`. The directory contains `config.json`, `metrics.csv`, `last_checkpoint.pth`, a best-model checkpoint (`best_model.pth`, when produced), and figures and predictions. To resume from a checkpoint:

```bash
nohup uv run --locked python -u train_code.py \
  --resume experiments/<run_name>/last_checkpoint.pth \
  > train_resume.log 2>&1 < /dev/null &
```

When resuming without `--config`, the code uses the configuration saved in the checkpoint and continues in the same experiment directory. Check that its data paths are still valid. To change options, also pass `--config configs/local.json`.

## Citation

If you use the dataset or this code in research, cite the associated paper:

```bibtex
@inproceedings{integracar_lulc_2026,
  title     = {IntegraCAR-LULC-10K: A High-Resolution Optical Satellite Dataset for LULC Segmentation in the Brazilian Rural Environmental Registry},
  author    = {Albertino, Calebe and Lima, Gabriel M. B. and Oliveira, Vin{\'{i}}cius R. and Ribeiro, Arthur R. V. and Oliveira, Eliza K. S. and Souza, Eduardo H. P. and Dalvi, Otavio G. and Komati, Karin S. and Andrade, Jefferson O. and Boldt, Francisco A. and Paix{\~{a}}o, Thiago M.},
  booktitle = {Proceedings of the 39th SIBGRAPI Conference on Graphics, Patterns and Images (SIBGRAPI)},
  year      = {2026}
}
```
