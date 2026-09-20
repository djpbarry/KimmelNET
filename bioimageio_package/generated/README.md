# KimmelNET — Zebrafish Embryo Age Prediction

## Description

KimmelNET predicts the age (hours post fertilisation, HPF) of individual zebrafish
embryos from 2D brightfield microscopy images. It is a convolutional neural network
trained to regress a single continuous HPF value, following the staging framework
described by Kimmel et al.

The model takes a single grayscale image, resized to 224 x 268 pixels, and outputs a
scalar prediction of the embryo's developmental age in hours post fertilisation.

## Intended Use

- Staging zebrafish embryos from brightfield images, for developmental biology
  experiments and timelapse analysis.
- Suitable for images similar to the training data: individual embryos imaged in
  brightfield at the magnification used in the source study.

## Known Limitations

Trained on brightfield images; may underperform on very different aspect ratios or
acquisition settings. Predictions are most reliable within the training age range
(~4.5 to 50 hpf).

## Validation

The model was validated against images of wild-type embryos staged at two incubation
temperatures (25.0 C and 28.5 C). A representative withheld image (true label 24 hpf)
yields a prediction of 22.1 hpf.

## Citation

Jones RA, Renshaw MJ and Barry DJ. Automated staging of zebrafish embryos with deep
learning. _Life Science Alliance_, 7 (1): e202302351. DOI:
[10.26508/lsa.202302351](https://doi.org/10.26508/lsa.202302351)

Source code: https://github.com/djpbarry/KimmelNET
