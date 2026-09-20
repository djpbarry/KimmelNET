# AGENTS.md

Guidance for AI agents working in the KimmelNET repository.

## Project Overview

KimmelNET is a deep-learning model (TensorFlow/Keras) that predicts the age of zebrafish embryos (hours post fertilisation, HPF) from 2D brightfield images. It is a **regression** task (a CNN predicting a single continuous HPF value), not a classifier. The reference paper is Jones, Renshaw & Barry, *Automated staging of zebrafish embryos with deep learning*, Life Science Alliance 7(1), DOI 10.26508/lsa.202302351.

The primary user-facing entry point is the Jupyter notebook `zebrafish_age_estimator.ipynb` (also deployable via Binder). The standalone `.py` scripts in the repo root are the training/evaluation/analysis code that produced the published model.

## Environment & Dependencies

- Python 3.10 (per README badge).
- Dependencies are in `requirements.txt`: `tensorflow`, `matplotlib`, `numpy`, `pandas`, `scipy`, `scikit-learn`, `scikit-image`, `notebook`.
- `requirements.txt` uses **unpinned** version specifiers. This matters: the notebook uses `TFSMLayer` (from `keras.layers`) and `serving_default` endpoints, whereas several `.py` scripts use `keras.models.load_model(...)` and older preprocessing APIs (`layers.experimental.preprocessing.*`), which only work on specific TensorFlow versions.
- There is **no** setup.py/pyproject.toml, no test framework, no linter, no CI, and no Makefile. There are no automated tests and no build step.

## Key Commands

There is no build or test command. Useful references:

```bash
# Install environment
conda create --name kimmelnet pip
conda activate kimmelnet
python -m pip install -r requirements.txt

# Run the interactive estimator
jupyter notebook zebrafish_age_estimator.ipynb

# Load the published model directly from Python
python -c "from tensorflow import keras; m = keras.models.load_model('KimmelNet_Model/published_model_multi_gpu_custom_augmentation_trained_model'); m.summary()"
```

## Architecture & Data Flow

The canonical data layout is a directory tree where each subdirectory is named by a numeric HPF label and contains PNG images of embryos at that age:

```
test_data/
  4.5/image_0.png ...
  6.0/...
  50.0/...
```

The label for each image is derived by `parse_image` from the name of its **parent directory** (`float(parts[-2])` after splitting the path on the OS separator). This convention is central and repeated in every script. Image filenames are otherwise ignored.

Standard pipeline (each script re-implements it rather than sharing a module):
1. `glob` for `*/ *.png` files under a data root.
2. Filter out a hardcoded blocklist of known-bad well IDs (a long `... not in r and ...` chain).
3. Build a `tf.data.Dataset` via `tf.data.Dataset.list_files(...)`, `shuffle`, `map(parse_image)`, `batch`, `prefetch(AUTOTUNE)`, `cache`.
4. Resize images to `image_size = (224, 268)`, crop to `cropped_image_size = (224, 224)` via `layers.CenterCrop`.
5. The model is a sequential CNN: `Conv2D`/`MaxPooling2D` stacks → `Flatten` → `Dropout(0.5)` → `Dense(1)` (single linear output), compiled with `loss="mean_squared_error"`.

## Files & Roles

| File | Purpose |
|------|---------|
| `definitions.py` | Shared constants used by several scripts (`name`, `test_source_folder`). Note: declares `test_source_folder: str = "..."` as an *annotated* variable which is legal but non-idiomatic at module level with no assignment style. |
| `zebrafish_age_estimator.ipynb` | Primary user-facing estimator; loads model via `TFSMLayer(..., call_endpoint='serving_default')`. |
| `train_model.py` | Full training script. Uses `tf.distribute.MirroredStrategy()`. Takes dataset index from `sys.argv[1]`. Saves model + a copy of its own source + training log + sample-image plot. |
| `transfer_learning.py` | Fine-tunes an existing model. Args: `sys.argv[1]`=model dir, `[2]`=epochs, `[3]`=layers to retrain, `[4]`=run suffix. Freezes all but the last N layers. |
| `hypertune_model.py` | Hyperparameter tuning via `keras_tuner.BayesianOptimization`. |
| `test_data_augmentation.py` | Generates augmented training data (histogram equalisation, contrast/saturation rescale, noise) out to a new folder tree. Takes `sys.argv[1]` as output suffix. |
| `generate_saliency_maps.py` | Guided-backprop saliency maps via a `@tf.custom_gradient` `guidedRelu`. Saves `.tiff` gradient maps. |
| `plot_results.py` | Post hoc analysis/plotting over prediction CSVs. Args: `sys.argv[3]`=plot filename suffix. Runs 10,000-sample bootstrap loops. |
| `IJ_Macros/*.ijm` | FIJI/ImageJ macros to organise raw images into the required folder structure. `OrganiseImages.ijm` requires the **Bio-Formats** plugin. |
| `KimmelNet_Model/.../` | The published, pre-trained model in TensorFlow SavedModel format (`saved_model.pb`, `variables/`, `fingerprint.pb`). |

## Conventions & Patterns

- **Repeated boilerplate**: the `parse_image` function and the `filtered_*_files` exclusion list are copy-pasted nearly verbatim across `train_model.py`, `transfer_learning.py`, `test_data_augmentation.py`, `generate_saliency_maps.py`, and the estimator notebook. If you edit one, check the others for consistency, but note there is deliberately no shared module for this.
- **Hardcoded paths**: many scripts contain absolute HPC paths (`/nemo/stp/lm/working/barryd/hpc/...`, `z:/working/...`) and lab-specific dataset names (`Zebrafish_Train_Regression`, `Zebrafish_Test_Princeton_Regression`, `20232803 ZF 15 mins 25`). These are environment-specific and not portable. Where they are active (rather than commented-out notes), they are marked with a `HPC/lab-specific path` comment at the top of the variable. Do not assume they are valid in general use.
- **Output convention**: scripts write results under an `outputs/` directory with a `name + timestamp` suffix, and frequently persist a copy of their own source (`name_source.py`) and `model.summary()` text alongside outputs for reproducibility.
- **Model naming**: names are assembled from `definitions.name` + descriptive prefixes (e.g. `published_model_multi_gpu_custom_augmentation_trained_model`). The `name` field in `definitions.py` historically encoded which training configuration (e.g. multi-GPU, custom augmentation) generated a model.
- **Color scheme** in `plot_results.py` uses discrete named tuple variables (`lred`, `dblue`, etc.) rather than inline colours.

## Gotchas & Non-Obvious Details

- **Image dimensions are (height, width) = (224, 268), then center-cropped to 224×224.** The README documents 268×224 width×height, but code uses `image_size = (224, 268)` in the `(height, width)` order expected by TensorFlow resize. Keep this order; swapping them silently corrupts input.
- **Labels are floats parsed from folder names** via `float(parts[-2])`, so folder names must be valid floats (e.g. `4.5`, not `4,5` or `4.5h`).
- **The `layers.experimental.preprocessing` namespace** was historically used in `hypertune_model.py` but has been replaced with the non-experimental `layers.*` equivalents (`RandomFlip`, `Rescaling`, `CenterCrop`).
- **The augmented-data pipeline** (`test_data_augmentation.py`) saves images to folder names like `str(4.5 + i*0.25)`, producing floating-point folder names with potential `4.5` vs `4.75` style labels — the folder name is what becomes the label downstream.
- **`Transfer_learning.py` retrains layer slice `model.layers[5:-layers_to_train]`**; the index `5` is a magic number tied to the number of preprocessing layers in the base model.
- **GPU/HPC assumptions**: `train_model.py`, `transfer_learning.py`, and `test_data_augmentation.py` all call `tf.distribute.MirroredStrategy()` and assume a multi-GPU HPC environment. They will still run (with a warning) on a single GPU or CPU, but `batch_size` values (256/512) are scaled for that hardware.
- **The model is NOT binary**: it outputs a single linear (Dense(1)) regression value. Prediction is a raw HPF estimate, compare against the Kimmel equation's expected slope (1.0 for 28.5°C wild-type, 0.805 for 25.0°C) which is hardcoded in `plot_results.py`.

## Testing

There is no test suite. To sanity-check changes, the practical approach is to run a small slice of the notebook or one of the scripts against the bundled `test_data/` (a small sample of 50 labelled images) and confirm it loads the model and produces predictions. Be aware that `train_model.py` and `transfer_learning.py` require HPC-scale data not present in the repo and won't meaningfully run locally without external datasets.
