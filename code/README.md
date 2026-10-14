### Data Augmentation (**augment_data.py**)

This script generates **20 augmented versions of each image** across four classes: Full Cable, Left End, Right End, and No Cable. With `SAVE_ORIGINALS = True`, each input produces **21 output images**, including the original.

The augmentations simulate changes in lighting, contrast, color, blur, noise, image quality, and camera position. Output images retain the class folder structure, and `README_augmentations.txt` records the applied transformations.

Before running the script, set `DATA_ROOT` to the input dataset folder and `OUTPUT_ROOT` to the desired output folder. Existing output images are skipped unless `OVERWRITE = True`. Optional JPEG compression is currently disabled (`JPEG_ONLY_COUNT = 0`).

### CNN Accelerator (**ds_cnn.cpp**)

This file implements the CNN accelerator in C++ for Vitis HLS. It processes a **60 × 80 grayscale image** through convolution, ReLU, max pooling, depthwise and pointwise convolution, average pooling, and global average pooling. A final classifier produces four class scores (logits), and **argmax** selects the predicted class.

The implementation uses **16-bit fixed-point data** (`ap_fixed<16,8>`) and **24-bit accumulators** (`ap_fixed<24,12>`). Trained weights are supplied through `ds_cnn_weights_data.h`.

The top-level function, `ds_cnn`, reads the input image through an **AXI memory-mapped interface** and exposes control, the prediction, and logits through **AXI-Lite**. HLS directives request pipelining, array partitioning, and FPGA memory resources for hardware implementation.
