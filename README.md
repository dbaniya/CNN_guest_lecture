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

### 3. Install Vitis, Vitis HLS, and Vivado

Download the **Unified Installer for version 2020.2** from the [AMD download page](https://www.amd.com/en/support/downloads/adaptive-socs-and-fpgas/development-tools/2020-2.html).

Run the installer and select **Vitis** as the installation product. Ensure that **Vitis, Vitis HLS, and Vivado** are included, along with device support for **Zynq-7000**.

Complete the installation. **Version 2020.2 is preferred** because it matches the tools used in this lab.

**Version note:** You may use a newer release, but menus, tool names, and project workflows may differ.

### 4. Download the training dataset

Download `new_training_data` from the dataset link provided in the course materials. If the download is compressed, extract it into your project folder.

Note the location of the `new_training_data` folder, as you will use this path in the data augmentation script.

### 5. Augment the training dataset

Open the data augmentation script and set `DATA_ROOT` to the absolute path of your `new_training_data` folder:

```python
DATA_ROOT = Path("/absolute/path/to/new_training_data")
OUTPUT_ROOT = DATA_ROOT.parent / f"{DATA_ROOT.name}_aug"
```

Replace `/absolute/path/to/new_training_data` with the actual dataset path. `OUTPUT_ROOT` creates an output folder named `new_training_data_aug` alongside the original dataset folder.

Save and run the script with the `cnn_env` virtual environment activated. After it finishes, confirm that `new_training_data_aug` contains the augmented images.

### 6. Train the CNN

Download `train_cnn.py` from the course GitHub repository and save it in your project folder.

Open the script and set `ROOT` to the absolute path of the augmented dataset folder:

```python
ROOT = Path("/absolute/path/to/new_training_data_aug")
```

Replace `/absolute/path/to/new_training_data_aug` with the actual folder path, then save the script.

With the `cnn_env` virtual environment activated, run:

```bash
python train_cnn.py
```

Wait for training and evaluation to complete.

### 7. Inspect the training outputs

After training finishes, open the run folder created under the path specified by `RUNS_DIR` in the training script.

The run folder should contain the following subfolders:

- `models`: saved trained models
- `plots`: training curves and confusion matrices
- `reports`: evaluation results and classification reports
- `weights_cstyle_folded`: exported C-style weights with batch normalization folded into the convolution parameters

Review the plots and reports to assess the model’s performance. Keep the `weights_cstyle_folded` folder for the Vitis HLS implementation.

### 8. Create a new Vitis HLS project

Open **Vitis HLS** and select **Create New Project**.

Enter a project name and choose a location for the project, then click **Next** to continue through the project wizard.


### 9. Set the top-level function

In the Vitis HLS project settings, set **Top Function** to `ds_cnn`.

This name must exactly match the top-level function in the provided C++ source code. Vitis HLS uses this function as the entry point for generating the CNN hardware accelerator.

### 10. Select the target FPGA part

During the **Vitis HLS project setup**, go to the **Solution Configuration** window and select the following FPGA part:

`xc7z020clg400-1`

This is the Zynq-7020 device used on the **PYNQ-Z2** board.
