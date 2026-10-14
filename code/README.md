### Data Augmentation (**augment_data.py**)

This script generates **20 augmented versions of each image** across four classes: Full Cable, Left End, Right End, and No Cable. With `SAVE_ORIGINALS = True`, each input produces **21 output images**, including the original.

The augmentations simulate changes in lighting, contrast, color, blur, noise, image quality, and camera position. Output images retain the class folder structure, and `README_augmentations.txt` records the applied transformations.

Before running the script, set `DATA_ROOT` to the input dataset folder and `OUTPUT_ROOT` to the desired output folder. Existing output images are skipped unless `OVERWRITE = True`. Optional JPEG compression is currently disabled (`JPEG_ONLY_COUNT = 0`).

### CNN Accelerator (**ds_cnn.cpp**)

This file implements the CNN accelerator in C++ for Vitis HLS. It processes a **60 × 80 grayscale image** through convolution, ReLU, max pooling, depthwise and pointwise convolution, average pooling, and global average pooling. A final classifier produces four class scores (logits), and **argmax** selects the predicted class.

The implementation uses **16-bit fixed-point data** (`ap_fixed<16,8>`) and **24-bit accumulators** (`ap_fixed<24,12>`). Trained weights are supplied through `ds_cnn_weights_data.h`.

The top-level function, `ds_cnn`, reads the input image through an **AXI memory-mapped interface** and exposes control, the prediction, and logits through **AXI-Lite**. HLS directives request pipelining, array partitioning, and FPGA memory resources for hardware implementation.


### CNN Test Bench (**ds_cnn_tb.cpp**)

This file tests the CNN implementation in Vitis HLS simulation. It loads **60 × 80 grayscale images stored as text files**, applies the training normalization `(x / 255 − 0.5) / 0.25`, converts the values to fixed-point format, and calls `ds_cnn`.

The test bench compares predictions with the labels for **Full Cable (0), Left End (1), Right End (2), and No Cable (3)**. It prints per-image predictions and logits, overall accuracy, and confusion matrices showing counts and row-normalized results (**rows = true labels, columns = predictions**).

Before running, set `BASE_DIR` to the test dataset path. By default, `SAMPLES_PER_CLASS = 1` selects one random image per class; set it to `0` to test all images. Set `VERBOSE_PER_FILE = false` to disable per-image output.

### CNN Definitions and Weight Declarations (**ds_cnn_weights.h**)

This header defines the CNN’s **16-bit fixed-point data type** (`ap_fixed<16,8>`) and network dimensions: a **60 × 80 input**, **16 convolution channels**, **8 pointwise output channels**, and **4 classes**. It also specifies the spatial dimensions after max pooling (**30 × 40**) and average pooling (**5 × 5**).

The `weights` namespace declares the weight and bias arrays for the standard convolution, depthwise convolution, pointwise convolution, and classifier. The actual parameter values are supplied separately through `ds_cnn_weights_data.h`.

An include guard prevents repeated inclusion of the header.


### CNN Weight Data (**ds_cnn_weights_data.h**)

This header defines the CNN’s weight and bias arrays by including the C-style parameter files exported during Python training from `weights_cstyle_folded`.

It supplies parameters for the standard convolution, depthwise convolution, pointwise convolution, and classifier, using the fixed-point type and array dimensions defined in `ds_cnn_weights.h`. Batch normalization is folded into the parameters of the first three layers.

Keep `weights_cstyle_folded` alongside this header and preserve the exported filenames. The parameters are included at compile time for simulation and hardware synthesis.


