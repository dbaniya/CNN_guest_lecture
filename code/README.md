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


### Bare-Metal FPGA Application (**helloworld.c**)

This application runs on the Zynq ARM processor and controls the CNN accelerator. It receives **60 × 80 images as 16-bit Q8.8 values** over UART at **4,000,000 baud (4 Mbaud)**. Each image contains **9,600 payload bytes** (60 × 80 × 2).

The application stores the image in DDR memory, flushes the data cache, and starts the accelerator through its **AXI-Lite control interface**. After inference completes, it reads the predicted class and four logits.

Execution time is measured using the processor timer, while an **AXI GPIO pulse** supports oscilloscope or logic analyzer measurements. The measured interval includes software control overhead and excludes UART image transfer.

Results are returned to the computer in a **15-byte binary reply** containing framing markers, the prediction, elapsed time in microseconds, and four fixed-point logits. Ensure the UART baud rate, accelerator register offsets, and device identifiers match your hardware design and Python sender script.


### FPGA Image Testing over UART (**send_images_uart.py**)

This script sends test images from the computer to the FPGA over UART at **4,000,000 baud (4 Mbaud)**. It converts images to grayscale, resizes them to **60 × 80**, applies the training normalization, and encodes the pixels as **16-bit Q8.8 values**.

Each image is transmitted with framing markers and a payload length, followed by **9,600 image bytes**. The script reads the FPGA’s **15-byte reply**, containing the predicted class, measured inference time, and four fixed-point logits.

It prints per-image results, overall accuracy, and a confusion matrix (**rows = true labels, columns = predictions**). By default, it tests up to **1,000 images per class**.

Before running, update `ROOT` and `SERIAL_PORT`, and ensure `BAUD_RATE` matches the Vitis application. The dataset must contain the folders `Full Cable`, `Left End`, `Right End`, and `No Cable`. Close any other serial terminal using the selected port.


### CNN Training and Weight Export (**train_cnn.py**)

This script trains a lightweight CNN using **TensorFlow’s Keras API** to classify Full Cable, Left End, Right End, and No Cable images. It loads images in grayscale, resizes them to **60 × 80**, applies normalization `(x / 255 − 0.5) / 0.25`, and creates a stratified **70% training / 30% validation split**.

The network combines standard, depthwise, and pointwise convolution with batch normalization, ReLU, pooling, and a four-class softmax output. Training uses **Adam**, early stopping, and learning-rate reduction.

Outputs include saved models, training history, loss and accuracy plots, validation confusion matrices, and a classification report. The script also exports **C-style weights and biases**, folding batch normalization into the first three convolution layers for Vitis HLS deployment.

Before running, update `ROOT` and `run_dir` to your dataset and output paths. Adjust the training settings as needed; the provided script currently uses `EPOCHS = 2`.




