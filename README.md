# Automated staging of zebrafish embryos with KimmelNet

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/djpbarry/KimmelNET/main?labpath=zebrafish_age_estimator.ipynb) [![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/) ![Commit activity](https://img.shields.io/github/commit-activity/y/djpbarry/KimmelNET?style=plastic) ![GitHub](https://img.shields.io/github/license/djpbarry/KimmelNET?color=green&style=plastic)

![Overview](https://github.com/djpbarry/KimmelNET/blob/main/images/Overview.png)

This repository contains the Python implementation of automated zebrafish staging as described in the following paper:

- Barry DJ, Jones RA and Renshaw MJ. Automated staging of zebrafish embryos with KimmelNet. *bioRxiv*, 2023. doi: https://doi.org/10.1101/2023.01.13.523922

# Overview

KimmelNet has been trained to predict the age (hours post fertilisation) of individual zebrafish embryos. When tested on 2D brightfield images of zebrafish embryos (such as those shown above), the predictions generated agree closely with those expected from the Kimmel equation.

# Get Started

The quickest and easiest way to try KimmelNet is to [try it on Binder](https://mybinder.org/v2/gh/djpbarry/KimmelNET/main?labpath=zebrafish_age_estimator.ipynb). This will allow you to run KimmelNet on the images stored in this repo's test_data directory.

# Run KimmelNet On Your Own Data

To test KimmelNet on your own data, the easiest thing to do is download this repo and replace the "test_data" with your own images, using a similar folder structure. You can then use the [Jupyter Notebook](https://github.com/djpbarry/KimmelNET/blob/main/zebrafish_age_estimator.ipynb) to run KimmelNet on your own images.

A step-by-step guide is presented below. **You only need to perform steps 1 and 2 once.** Every subsequent time you want to run KimmelNet, skip straight to step 3.

## Step 1: Install a Python Distribution

We recommend using conda as it's relatively straightforward and makes the management of different Python environments simple. You can install conda from [here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html#regular-installation) (miniconda will suffice).

## Step 2: Set Up Environment

Once conda is installed, open a terminal (Mac) or command line (Windows) and run the following series of commands:

```
conda create --name kimmelnet pip
conda activate kimmelnet
python -m pip install -r <path to this repo>/requirements.txt
```
where you need to replace `<path to this repo>` with the location on your file system where you downloaded this repo. You will be presented with a list of packages to be downloaded and installed. The following prompt will appear:
```
Proceed ([y]/n)?
```
Hit Enter and all necessary packages will be downloaded and installed - this may take some time. When complete, you can deactivate the environment you have created with the following command.

```
conda deactivate
```
You have successfully set up an environment to run KimmelNet!

## Step 3: Prepare your images

KimmelNet processes images of size 268 x 224 pixels (width x height). don't worry if your images are a different size - they will be scaled automatically. However, if the aspect ratio of your images is substantially different to this, then KimmelNet may not perform terribly well.

Organise your images such that they can conform to the same folder structure as the test_data in this repo. Each image *must* be saved in a folder corresponding to the hours post-fertilisation when the image was captured. For example:
```
test_data
|
|-- 4.5
|   |
|   |   image_0.png
|   |   image_1.png
|   ⋮
|   |   image_n.png
|-- 6.0
|   |
|   |   image_0.png
|   |   image_1.png
|   ⋮
|   |   image_n.png
⋮
⋮
|-- 50.0
|   |
|   |   image_0.png
|   |   image_1.png
|   ⋮
|   |   image_n.png
```
The names of the individual images are unimportant.

Organising your images to conform to this structure can be done manually. However, this might be impractical for large number of images. We have therefore provided a FIJI script ([IJ_Macros/OrganiseImages.ijm](https://github.com/djpbarry/KimmelNET/blob/main/IJ_Macros/OrganiseImages.ijm)) to automatically convert your images and store them in the above folder structure.

## Step 4: Run KimmelNet

The following commands will launch a Jupyter notebook allowing you to run KimmelNet on your own images:
```
conda activate kimmelnet
jupyter notebook <path to this repo>/zebrafish_age_estimator.ipynb
```

The Jupyter Notebook should open in your browser - follow the step-by-step instructions in the notebook to run the code. If you are not familiar with Jupyter Notebooks, you can find a detailed introduction [here](https://jupyter-notebook.readthedocs.io/en/latest/notebook.html#introduction).

# Train KimmelNet On Your Own Data

To train KimmelNet on your own images, you first need to organise your training data into the same folder structure used in this repo's "test_data". You can then use the [provided Python script](https://github.com/djpbarry/KimmelNET/blob/main/train_model.py) to run the training, by simply changing the `train_path` variable to specify the location of your training data - here's the first few lines of `train_model.py` with the most important variables highlighted:

```python
import glob
import os
from datetime import datetime

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers

import definitions

# Training images will be resized to the following dimensions, then cropped
image_size = (224, 268)
cropped_image_size = (224, 224)

# Consult the Tensorflow documentation for information on the following variables
batch_size = 256
epochs = 1200
buffer_size = 4

# Change the following variable to specify the path to your training data
train_path = "path/to/your/training/data"
```
The variables `batch_size`, `epochs` and `buffer_size` will the influence the training process. For more information on what each of these variables do, [consult the Tensorflow documentation](https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit).

# Use KimmelNet In Your Own Python Scripts

To use KimmelNet independently of the scripts and notebooks in this repo, you can load and display the model summary with the following simple commands:
```python
from tensorflow import keras

model = keras.models.load_model('KimmelNet_Model/published_model_multi_gpu_custom_augmentation_trained_model')
model.summary()
```
