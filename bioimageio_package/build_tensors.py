"""Generate test tensors and cover image for the KimmelNET bioimage.io package.

Runs the *real* model on a representative `test_data/` image to produce a
genuine `test_output.npy` (no hand-crafted values), plus a cover image.
"""
import glob
import os

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

ROOT = ".."
MODEL_PATH = os.path.join(
    ROOT, "KimmelNet_Model",
    "published_model_multi_gpu_custom_augmentation_trained_model",
)
OUT = "generated"

# Pick a representative mid-range embryo (label 24 -> ~24 hpf).
files = sorted(glob.glob(os.path.join(ROOT, "test_data", "24", "*.png")))
src = files[0]
label = float(os.path.split(os.path.dirname(src))[1])
print("using", src, "label", label)

img = tf.image.decode_png(tf.io.read_file(src))          # (H, W, 1) uint8
img = tf.image.resize(img, (224, 268))                   # (224, 268, 1)
img = tf.cast(img, tf.float32)                            # raw 0-255

# --- test_input.npy: exactly the declared input tensor (batch, 224, 268, 1) ---
test_input = img.numpy()[None]                            # (1, 224, 268, 1)
np.save(os.path.join(OUT, "test_input.npy"), test_input.astype("float32"))

# --- run the real model ---
gd = tf.saved_model.load(MODEL_PATH)
sig = gd.signatures["serving_default"]
out = sig(input_1=tf.constant(test_input))["dense"].numpy()  # (1, 1)
test_output = out.astype("float32")
np.save(os.path.join(OUT, "test_output.npy"), test_output)
print("prediction:", test_output.ravel()[0])

# --- cover image (1:1, input + prediction) ---
fig, ax = plt.subplots(1, 2, figsize=(4, 2), dpi=150)
ax[0].imshow(test_input[0, :, :, 0], cmap="gray", vmin=0, vmax=255)
ax[0].set_title("Input", fontsize=8)
ax[0].axis("off")
ax[1].axis("off")
ax[1].set_title(f"Predicted {test_output.ravel()[0]:.1f} hpf", fontsize=8)
ax[1].imshow(np.full((224, 268), 255), cmap="gray", vmin=0, vmax=255)
ax[1].text(134, 112, f"{test_output.ravel()[0]:.1f}", ha="center", va="center", fontsize=20)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "cover0.png"), bbox_inches="tight")
plt.close(fig)
print("done")
