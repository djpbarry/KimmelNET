# Automated staging of zebrafish embryos with KimmelNet

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/djpbarry/KimmelNET/main?labpath=zebrafish_age_estimator.ipynb) [![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/) ![Commit activity](https://img.shields.io/github/commit-activity/y/djpbarry/KimmelNET?style=plastic) ![GitHub](https://img.shields.io/github/license/djpbarry/KimmelNET?color=green&style=plastic)

![Overview](https://github.com/djpbarry/KimmelNET/blob/main/images/Overview.png)

This repository contains the Python implementation of automated zebrafish staging as described in the following paper:

Barry DJ, Jones RA, Renshaw MJ. Automated staging of zebrafish embryos with KimmelNET.

# Overview

KimmelNet has been trained to predict the age (hours post fertilisation) of individual zebrafish embryos. When tested on 2D brightfield images of zebrafish embryos (such as those shown above), the predictions generated agree closely with those expected from the Kimmel equation.

# Get Started

The quickest and easiest way to try KimmelNet is to [try it on Binder](https://mybinder.org/v2/gh/djpbarry/KimmelNET/main?labpath=zebrafish_age_estimator.ipynb). This will allow you to run KimmelNet on the images stored in this repo's test_data directory.

# Run KimmelNet On Your Own Data

To test KimmelNet on your own data, the easiest thing to do is clone this repo and replace the "test_data" with your own images, using a similar folder structure. You can then use the [Jupyter Notebook](https://github.com/djpbarry/KimmelNET/blob/main/zebrafish_age_estimator.ipynb) to run KimmelNet on your own images.

# Train KimmelNet On Your Own Data

To train KimmelNet on your own images, you first need to organise your training data into the same folder structure used in this repo's "test_data". You can then use the [provided Python script](https://github.com/djpbarry/KimmelNET/blob/main/train_model.py) to run the training, by simply changing the `train_path` variable to specify the location of your training data:

```python
import glob
import os
from datetime import datetime

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers

import definitions

image_size = (224, 268)
cropped_image_size = (224, 224)
batch_size = 256
epochs = 1200
buffer_size = 4

train_path = "path/to/your/training/data"
```
The variables `batch_size`, `epochs` and `buffer_size` will the influence the training process. For more information on what each of these variables do, [consult the Tensorflow documentation](https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit).

# Use KimmelNet In Your Own Python Scripts

To use KimmelNet independently of the scripts and notebooks in this repo, you can load and display the model summary with the following simple commands:
```python
from tensorflow import keras

model = keras.models.load_model('./simple_regression_trained_model')
model.summary()
```
