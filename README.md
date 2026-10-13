# Lightweight CNN for Embedded Vision
## From convolution to an efficient image classifier

This lab presents a practical workflow for training and deploying a custom-designed CNN on an FPGA to classify images into four classes: Full Cable, Right End, Left End, and No Cable. The images are organized into three datasets—training, validation, and test—and all three datasets are augmented. The images are then converted to grayscale and normalized before the CNN is trained, validated, and tested using TensorFlow’s Keras API. Loss curves and confusion matrices are used to assess its performance. Next, the trained model parameters are exported for accelerator implementation and verification in Vitis HLS. Finally, the generated RTL is integrated into a Vivado design, and the FPGA is programmed to verify the model’s predictions on hardware.

## Requirements

- Python 3.12 with `venv` and `pip`
- A Python virtual environment with the following packages installed:
  - TensorFlow
  - NumPy
  - OpenCV (`opencv-python`)
  - Matplotlib
  - scikit-learn
  - Albumentations
  - pyserial
- AMD/Xilinx development tools: Vitis, Vitis HLS, and Vivado (**version 2020.2 preferred**, as used in the course instructions)
- PYNQ-Z2 FPGA board with Zynq-7020 (`xc7z020clg400-1`)
- USB Type-A to Micro-USB Type-B cable for FPGA programming, UART communication, and power
- Course datasets and source files from the course GitHub repository

## Disclaimer

The instructions were prepared using Vitis, Vitis HLS, and Vivado **2020.2**. Newer versions may have different menus, project workflows, or settings. Adjust the steps as needed for your installed software versions.

## Instructions

### 1. Create and activate a Python virtual environment

Open a terminal in your project folder and create a virtual environment named `cnn_env` using Python 3.12:

```bash
/usr/bin/python3.12 -m venv cnn_env
```

Activate the virtual environment:

```bash
source cnn_env/bin/activate
```

Verify the Python version:

```bash
python --version
```

The output should show `Python 3.12.x`. Keep the virtual environment activated when installing packages or running the Python scripts.

### 2. Install the required Python packages

With the `cnn_env` virtual environment activated, upgrade `pip`:

```bash
python -m pip install --upgrade pip
```

Install the required packages:

```bash
python -m pip install tensorflow numpy opencv-python matplotlib scikit-learn albumentations pyserial
```

The packages are used for:

- **TensorFlow:** building, training, and running the CNN
- **NumPy:** numerical arrays and operations
- **OpenCV (`opencv-python`):** image loading and preprocessing
- **Matplotlib:** plotting training curves and confusion matrices
- **scikit-learn:** evaluating classification performance
- **Albumentations:** augmenting the image datasets
- **pyserial:** sending test images to the FPGA and receiving predictions over UART

Verify the installed packages:

```bash
python -m pip list
```
