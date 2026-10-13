#!/usr/bin/env python3
"""
FULL SCRIPT (ONE FILE) — TinyCNN GRAYSCALE with:
  - Conv3x3 + BN + ReLU + MaxPool
  - DW3x3 + BN + ReLU
  - PW1x1 (16->BOTTLENECK_CH) + BN + ReLU
  - AvgPool to 5x5 bins (pool_size=(6,8))
  - Cls 1x1 (BOTTLENECK_CH->4) **NO BN on logits**
  - GAP -> softmax (training only; HLS uses argmax on logits)

AND:
  - Saves history JSON+CSV
  - Saves confusion matrices (png + npy + csv)
  - Saves classification report
  - Saves models (.keras, .h5, .weights.h5)
  - Exports **BN-folded** C-style weights for conv1/dw/pw
  - Exports direct C-style weights+bias for cls_1x1 (no BN)

Run output:
  runs/tinycnn_bn_DW_PW_convGAP_5x5_4class_60x80_<seed>/
    models/
    plots/
    reports/
    weights_cstyle_folded/
    run_info.txt

Normalization:
  - load grayscale in [0,1]
  - then X = (X - 0.5) / 0.25   (match your deployment!)

Bins / HLS constants:
  After MaxPool: (30,40,*)
  With pool_size=(6,8) => OH=5, OW=5
"""
import os
from xml.parsers.expat import model

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # CPU-only
os.environ["PYTHONHASHSEED"] = "42"

# REMOVE THIS LINE
# os.environ["TF_DETERMINISTIC_OPS"] = "1"

import random
from pathlib import Path
import json
import csv

import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf

seed = 42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

print("GPUs:", tf.config.list_physical_devices("GPU"))

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, DepthwiseConv2D, BatchNormalization, Activation,
    MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report


# ===================== EDIT THESE =====================
ROOT = Path("/home/dinesh/Journal/Dataset/new_training_data_aug")

CLASS_FOLDERS = [
    "Full Cable", # 0
    "Left End",   # 1
    "Right End",  # 2
    "No Cable",   # 3
]

IMG_H, IMG_W = 60, 80
VAL_SPLIT = 0.30
BATCH = 16
EPOCHS = 2
LR = 3e-4

USE_GPU0_ONLY = True

BOTTLENECK_CH = 8

# 5x5 bins: after MaxPool -> (30,40). (6,8) => (30/6=5, 40/8=5)
BINS_POOL_SIZE = (6, 8)

# RUNS_DIR = Path("runs")
# RUN_NAME_PREFIX = "tinycnn_bn_DW_PW_convGAP_5x5_4class_60x80"

run_dir = Path("/home/dinesh/Journal/dw_pw_cnn_trained_model_gl")

# ======================================================

seed = 42
print("\n🧪 Using fixed seed:", seed)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# ---------------------- Data ----------------------
def load_all_images_cv2():
    all_images = []
    all_labels = []
    resized_count = 0
    total_count = 0

    for label, cname in enumerate(CLASS_FOLDERS):
        d = ROOT / cname
        if not d.is_dir():
            print(f"[WARN] Missing folder: {d}")
            continue

        files = sorted(p for p in d.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS)
        print(f"[LOAD] {cname}: {len(files)} images")

        for p in files:
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            total_count += 1

            if img.shape != (IMG_H, IMG_W):
                img = cv2.resize(img, (IMG_W, IMG_H), interpolation=cv2.INTER_LINEAR)
                resized_count += 1

            img = img.astype(np.float32) / 255.0   # [0,1]
            all_images.append(img)
            all_labels.append(label)

    all_images = np.asarray(all_images, dtype=np.float32)
    all_labels = np.asarray(all_labels, dtype=np.int64)

    print("[TOTAL] samples:", all_images.shape[0])
    print(f"[RESIZE] resized {resized_count}/{total_count}")

    # (N,H,W) -> (N,H,W,1)
    all_images = np.expand_dims(all_images, axis=-1)

    # Shuffle whole dataset (reproducible)
    all_images, all_labels = shuffle(all_images, all_labels, random_state=seed)
    return all_images, all_labels


# ---------------------- Model ----------------------
def build_model(num_classes: int):
    """
    EXACT ARCHITECTURE:
      Conv3x3 -> BN -> ReLU -> MaxPool
      DW3x3   -> BN -> ReLU
      PW1x1   -> BN -> ReLU
      AvgPool to 5x5 bins
      cls 1x1 (NO BN) -> GAP -> softmax
    """
    BN_MOM = 0.9
    BN_EPS = 1e-3

    return Sequential([
        # (60,80,1) -> (60,80,16)
        Conv2D(16, (3, 3), padding="same", use_bias=False,
               input_shape=(IMG_H, IMG_W, 1), name="conv1"),
        BatchNormalization(momentum=BN_MOM, epsilon=BN_EPS, name="bn1"),
        Activation("relu", name="relu1"),

        # -> (30,40,16)
        MaxPooling2D((2, 2), name="pool1"),

        # DW: (30,40,16) -> (30,40,16)
        DepthwiseConv2D((3, 3), padding="same", use_bias=False, name="dw3x3"),
        BatchNormalization(momentum=BN_MOM, epsilon=BN_EPS, name="bn_dw"),
        Activation("relu", name="relu_dw"),

        # PW: (30,40,16) -> (30,40,BOTTLENECK_CH)
        Conv2D(BOTTLENECK_CH, (1, 1), padding="same", use_bias=False, name="pw1x1"),
        BatchNormalization(momentum=BN_MOM, epsilon=BN_EPS, name="bn_pw"),
        Activation("relu", name="relu_pw"),

        # bins: (30,40,CH) -> (5,5,CH) because pool=(6,8)
        AveragePooling2D(pool_size=BINS_POOL_SIZE, name="pool_bins_5x5"),

        # cls logits per bin: (5,5,CH) -> (5,5,num_classes)
        # NO BN here; keep logits linear
        Conv2D(num_classes, (1, 1), padding="same", use_bias=True, name="cls_1x1"),

        # GAP over 5x5 bins -> (num_classes,)
        GlobalAveragePooling2D(name="gap_logits"),

        # training only
        Activation("softmax", name="softmax"),
    ])


# ---------------------- Saving helpers ----------------------
def save_history(history, out_dir: Path, prefix: str = "history") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    hist = history.history

    # JSON
    with open(out_dir / f"{prefix}.json", "w") as f:
        json.dump(hist, f, indent=2)

    # CSV
    keys = list(hist.keys())
    rows = zip(*[hist[k] for k in keys]) if keys else []
    with open(out_dir / f"{prefix}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys)
        w.writerows(rows)


def plot_training_curves(history, out_dir: Path, prefix: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    hist = history.history
    epochs_x = range(1, len(hist.get("loss", [])) + 1)

    # Loss
    fig = plt.figure()
    plt.plot(epochs_x, hist.get("loss", []), marker="o", label="Train Loss")
    plt.plot(epochs_x, hist.get("val_loss", []), marker="x", label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_loss_vs_epoch.png")
    plt.show(); plt.close(fig)

    # Acc
    fig = plt.figure()
    plt.plot(epochs_x, hist.get("accuracy", []), marker="o", label="Train Acc")
    plt.plot(epochs_x, hist.get("val_accuracy", []), marker="x", label="Val Acc")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Accuracy")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_accuracy_vs_epoch.png")
    plt.show(); plt.close(fig)


def plot_confusion_matrices(y_true, y_pred, class_names, out_dir: Path, prefix: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = list(range(len(class_names)))

    # Counts
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    np.save(out_dir / f"{prefix}_counts.npy", cm)
    np.savetxt(out_dir / f"{prefix}_counts.csv", cm, delimiter=",", fmt="%d")

    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)\
        .plot(values_format="d", cmap="Blues", ax=ax, colorbar=False)
    ax.set_title("Confusion Matrix (Val Counts)")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_counts.png")
    plt.show(); plt.close(fig)

    # Normalized
    cmn = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    np.save(out_dir / f"{prefix}_normalized.npy", cmn)
    np.savetxt(out_dir / f"{prefix}_normalized.csv", cmn, delimiter=",", fmt="%.6f")

    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(confusion_matrix=cmn, display_labels=class_names)\
        .plot(values_format=".2f", cmap="Blues", ax=ax, colorbar=False)
    ax.set_title("Confusion Matrix (Val Normalized)")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_normalized.png")
    plt.show(); plt.close(fig)


# ---------------------- BN folding + export ----------------------
def fold_bn_scale_bias(gamma, beta, mean, var, eps):
    denom = np.sqrt(var + eps)
    scale = gamma / denom
    bias = beta - mean * scale
    return scale.astype(np.float32), bias.astype(np.float32)


def fold_bn_into_conv2d(W, b, scale, bias):
    # W: (kh,kw,Cin,Cout), scale/bias: (Cout,)
    if b is None:
        b = np.zeros((W.shape[-1],), dtype=np.float32)
    Wf = W * scale.reshape(1, 1, 1, -1)
    bf = b * scale + bias
    return Wf.astype(np.float32), bf.astype(np.float32)


def fold_bn_into_depthwise(W, b, scale, bias):
    # W: (kh,kw,Cin,depth_mult), depth_mult must be 1
    kh, kw, Cin, dm = W.shape
    if dm != 1:
        raise RuntimeError(f"Depthwise depth_multiplier must be 1 for exporter; got {dm}")
    if b is None:
        b = np.zeros((Cin,), dtype=np.float32)

    Wf = W * scale.reshape(1, 1, Cin, 1)
    bf = b * scale + bias
    return Wf.astype(np.float32), bf.astype(np.float32)


def save_weights_c_style_bn_folded(model, out_dir: Path):
    """
    Writes C-style arrays (curly braces) to .txt files:

      conv1 (BN-folded): (3,3,1,16) -> (9,16)
      dw3x3 (BN-folded): (3,3,16,1) -> (16,9)  (per-channel 3x3 taps)
      pw1x1 (BN-folded): (1,1,16,8) -> (16,8)
      cls_1x1 (NO BN):   (1,1,8,4)  -> (8,4) + bias(4)
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    def write_c_array_2d(fname: Path, W: np.ndarray):
        with open(fname, "w") as f:
            f.write("{")
            for i in range(W.shape[0]):
                f.write("{")
                for j in range(W.shape[1]):
                    f.write(f"{float(W[i, j])}")
                    if j < W.shape[1] - 1:
                        f.write(", ")
                f.write("}")
                if i < W.shape[0] - 1:
                    f.write(",\n")
            f.write("}")

    def write_c_array_1d(fname: Path, v: np.ndarray):
        with open(fname, "w") as f:
            f.write("{")
            for i in range(len(v)):
                f.write(f"{float(v[i])}")
                if i < len(v) - 1:
                    f.write(", ")
            f.write("}")

    # conv1 + bn1
    W1 = model.get_layer("conv1").get_weights()[0]  # no bias
    gamma, beta, mean, var = model.get_layer("bn1").get_weights()
    scale, bias = fold_bn_scale_bias(gamma, beta, mean, var, model.get_layer("bn1").epsilon)
    W1f, b1f = fold_bn_into_conv2d(W1, None, scale, bias)
    W1_out = W1f.reshape(3 * 3 * 1, 16)
    write_c_array_2d(out_dir / "layer_1_conv1_WEIGHTS.txt", W1_out)
    write_c_array_1d(out_dir / "layer_1_conv1_BIAS.txt", b1f)

    # dw3x3 + bn_dw
    Wd = model.get_layer("dw3x3").get_weights()[0]  # (3,3,16,1)
    gamma, beta, mean, var = model.get_layer("bn_dw").get_weights()
    scale, bias = fold_bn_scale_bias(gamma, beta, mean, var, model.get_layer("bn_dw").epsilon)
    Wdf, bdf = fold_bn_into_depthwise(Wd, None, scale, bias)

    kh, kw, Cin, dm = Wdf.shape
    Wd_out = np.zeros((Cin, kh * kw), dtype=np.float32)
    for c in range(Cin):
        Wd_out[c, :] = Wdf[:, :, c, 0].reshape(kh * kw)
    write_c_array_2d(out_dir / "layer_2_dw3x3_WEIGHTS.txt", Wd_out)
    write_c_array_1d(out_dir / "layer_2_dw3x3_BIAS.txt", bdf)

    # pw1x1 + bn_pw
    Wp = model.get_layer("pw1x1").get_weights()[0]  # (1,1,16,8)
    gamma, beta, mean, var = model.get_layer("bn_pw").get_weights()
    scale, bias = fold_bn_scale_bias(gamma, beta, mean, var, model.get_layer("bn_pw").epsilon)
    Wpf, bpf = fold_bn_into_conv2d(Wp, None, scale, bias)
    Wp_out = Wpf.reshape(16, BOTTLENECK_CH)
    write_c_array_2d(out_dir / "layer_3_pw1x1_WEIGHTS.txt", Wp_out)
    write_c_array_1d(out_dir / "layer_3_pw1x1_BIAS.txt", bpf)

    # cls_1x1 (NO BN; has bias)
    Wc, bc = model.get_layer("cls_1x1").get_weights()
    Wc_out = Wc.reshape(BOTTLENECK_CH, len(CLASS_FOLDERS))
    write_c_array_2d(out_dir / "layer_4_cls_1x1_WEIGHTS.txt", Wc_out)
    write_c_array_1d(out_dir / "layer_4_cls_1x1_BIAS.txt", bc)

    print(f"[EXPORT OK] Saved to: {out_dir}")
    print(f"  conv1:  W {W1_out.shape}, b {b1f.shape}")
    print(f"  dw3x3:  W {Wd_out.shape}, b {bdf.shape}   (Cin,9)")
    print(f"  pw1x1:  W {Wp_out.shape}, b {bpf.shape}")
    print(f"  cls1x1: W {Wc_out.shape}, b {bc.shape}   (NO BN)")


# ---------------------- Main ----------------------
def main():
    if USE_GPU0_ONLY:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    gpus = tf.config.list_physical_devices("GPU")
    print("[DEVICE]", gpus[0].name if gpus else "CPU")

    # Bins check: after MaxPool => 30x40
    if (30 % BINS_POOL_SIZE[0]) != 0 or (40 % BINS_POOL_SIZE[1]) != 0:
        raise RuntimeError(
            f"BINS_POOL_SIZE={BINS_POOL_SIZE} must divide (30,40) exactly."
        )
    bins_h = 30 // BINS_POOL_SIZE[0]
    bins_w = 40 // BINS_POOL_SIZE[1]
    print(f"[BINS] pool_size={BINS_POOL_SIZE} -> bins={bins_h}x{bins_w} (OH={bins_h}, OW={bins_w})")

    # # Run folders
    # run_dir = RUNS_DIR / f"{RUN_NAME_PREFIX}_{seed}"
    # models_dir = run_dir / "models"
    # plots_dir = run_dir / "plots"
    # reports_dir = run_dir / "reports"
    # cstyle_dir = run_dir / "weights_cstyle_folded"

    # Run folders
    models_dir = run_dir / "models"
    plots_dir = run_dir / "plots"
    reports_dir = run_dir / "reports"
    cstyle_dir = run_dir / "weights_cstyle_folded"


    for d in (models_dir, plots_dir, reports_dir, cstyle_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Metadata
    with open(run_dir / "run_info.txt", "w") as f:
        f.write(f"seed={seed}\n")
        f.write(f"root={ROOT}\n")
        f.write(f"classes={CLASS_FOLDERS}\n")
        f.write(f"img_h={IMG_H} img_w={IMG_W}\n")
        f.write(f"val_split={VAL_SPLIT}\n")
        f.write(f"batch={BATCH}\n")
        f.write(f"epochs={EPOCHS}\n")
        f.write(f"lr={LR}\n")
        f.write(f"bottleneck_ch={BOTTLENECK_CH}\n")
        f.write(f"bins_pool_size={BINS_POOL_SIZE}\n")
        f.write(f"bins_h={bins_h} bins_w={bins_w}\n")

    # Load data
    X, y = load_all_images_cv2()
    if X.shape[0] == 0:
        raise RuntimeError("No images loaded. Check ROOT and folder names.")

    # Normalize (match deployment)
    X = (X - 0.5) / 0.25

    # Stratified split
    x_train, x_val, y_train, y_val = train_test_split(
        X, y,
        test_size=VAL_SPLIT,
        stratify=y,
        random_state=seed,
        shuffle=True
    )
    print("[SPLIT] Train:", x_train.shape[0], "Val:", x_val.shape[0])

    # Build + train
    model = build_model(num_classes=len(CLASS_FOLDERS))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.summary()

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
    ]

    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH,
        callbacks=callbacks,
        verbose=1
    )

    # Save history + plots
    save_history(history, out_dir=reports_dir, prefix="history")
    # plot_training_curves(history, out_dir=plots_dir, prefix=RUN_NAME_PREFIX)
    plot_training_curves(history, out_dir=plots_dir, prefix="dw_pw_cnn")



    # Eval + confusion
    y_val_pred = np.argmax(model.predict(x_val, verbose=0), axis=1)
    
    # plot_confusion_matrices(
    #     y_true=y_val, y_pred=y_val_pred,
    #     class_names=CLASS_FOLDERS,
    #     out_dir=plots_dir,
    #     prefix=f"{RUN_NAME_PREFIX}_confusion_val"
    # )

    plot_confusion_matrices(
        y_true=y_val,
        y_pred=y_val_pred,
        class_names=CLASS_FOLDERS,
        out_dir=plots_dir,
        prefix="dw_pw_cnn_confusion_val"
    )

    # Report
    report = classification_report(y_val, y_val_pred, target_names=CLASS_FOLDERS)
    print("\nClassification report:\n", report)
    with open(reports_dir / "classification_report.txt", "w") as f:
        f.write(report)

    # Save models
    model.save(models_dir / "dwpw_cnn.keras")
    model.save(models_dir / "dwpw_cnn.h5")
    model.save_weights(models_dir / "dwpw_cnn.weights.h5")

    print("\n✅ Saved models into:", models_dir)

    # Export BN-folded weights for HLS
    save_weights_c_style_bn_folded(model, out_dir=cstyle_dir)
    print("\n✅ Saved BN-folded C-style weights into:", cstyle_dir)

    print("\n✅ DONE. All outputs saved under:", run_dir)
    print(f"[REMINDER] HLS bins: OH={bins_h}, OW={bins_w} (pool_size={BINS_POOL_SIZE}).")
    print("[NOTE] cls_1x1 has NO BN; exported directly with bias.")


if __name__ == "__main__":
    main()


 