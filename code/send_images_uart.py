#!/usr/bin/env python3

from pathlib import Path
import struct
import time

import cv2
import numpy as np
import serial


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path("/home/dinesh/Journal/Dataset/new_testing_data_aug")

SERIAL_PORT = "/dev/ttyUSB1"
BAUD_RATE = 4000000

MAX_IMAGES_PER_FOLDER = 1000

IMG_H = 60
IMG_W = 80
N_PIXELS = IMG_H * IMG_W
RX_BYTES = N_PIXELS * 2   # int16 payload

FRAME_START = b"\xA6\x6A"

CLASS_FOLDERS = [
    "Full Cable",
    "Left End",
    "Right End",
    "No Cable",
]


# ============================================================
# IMAGE PREPROCESSING
# Match training:
#   x = u8 / 255
#   z = (x - 0.5) / 0.25 = 4*x - 2
# Send Q8.8:
#   q = round(z * 256)
#     = round((1024*u8)/255 - 512)
# ============================================================

def image_to_q8_8_60x80(image_path: Path) -> np.ndarray:
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    if img.shape != (IMG_H, IMG_W):
        img = cv2.resize(img, (IMG_W, IMG_H), interpolation=cv2.INTER_LINEAR)

    img_u8 = img.astype(np.uint8).reshape(-1)

    px = img_u8.astype(np.int32)
    q = ((1024 * px + 127) // 255 - 512).astype(np.int32)
    q = np.clip(q, -32768, 32767).astype(np.int16)

    assert q.shape[0] == N_PIXELS

    return q


# ============================================================
# UART COMMUNICATION
# FPGA expects:
#   0xA6 0x6A <len u16 LE> <4800 int16 little-endian>
#
# FPGA replies:
#   0xA5 <pred> <time_us u32 LE> 0x5A
# ============================================================

def send_image_and_get_prediction(ser, q_img):
    payload = q_img.astype("<i2", copy=False).tobytes(order="C")

    if len(payload) != RX_BYTES:
        raise RuntimeError(f"Bad payload size: {len(payload)} != {RX_BYTES}")

    packet = FRAME_START + struct.pack("<H", len(payload)) + payload

    ser.reset_input_buffer()
    ser.write(packet)
    ser.flush()

    # Wait for binary reply start 0xA5.
    # This skips any FPGA ASCII/debug prints such as:
    #   accel(sw): 1.234 ms
    deadline = time.time() + 5.0
    found_start = False

    while time.time() < deadline:
        b = ser.read(1)

        if not b:
            continue

        if b[0] == 0xA5:
            found_start = True
            break

    if not found_start:
        raise RuntimeError("Timeout waiting for FPGA reply start 0xA5")

    pred_byte = ser.read(1)
    time_bytes = ser.read(4)
    logit_bytes = ser.read(8)
    end = ser.read(1)

    if (
        len(pred_byte) != 1
        or len(time_bytes) != 4
        or len(logit_bytes) != 8
        or len(end) != 1
    ):        
        raise RuntimeError("Incomplete FPGA reply")

    if end[0] != 0x5A:
        raise RuntimeError(f"Bad FPGA reply end: {end}")

    pred = pred_byte[0]
    time_us = struct.unpack("<I", time_bytes)[0]
    time_ms = time_us / 1000.0
    logits = struct.unpack("<hhhh", logit_bytes)

    return pred, time_us, time_ms, logits


# ============================================================
# LABELS
# ============================================================

def label_from_folder(folder_name: str) -> int:
    return CLASS_FOLDERS.index(folder_name)


# ============================================================
# MAIN
# ============================================================

def main():
    print()
    print("=====================================")
    print(" FPGA UART CNN TEST")
    print(" Q8.8 int16 60x80 image sender")
    print("=====================================")
    print()

    print(f"Dataset root : {ROOT}")
    print(f"Serial port  : {SERIAL_PORT}")
    print(f"Baud rate    : {BAUD_RATE}")
    print(f"Images/folder: {MAX_IMAGES_PER_FOLDER}")
    print(f"Image size   : {IMG_H} x {IMG_W}")
    print(f"TX bytes/img : {RX_BYTES}")
    print()

    ser = serial.Serial(
        port=SERIAL_PORT,
        baudrate=BAUD_RATE,
        timeout=5,
        write_timeout=5,
        xonxoff=False,
        rtscts=False,
        dsrdtr=False,
    )

    time.sleep(0.2)
    ser.reset_input_buffer()
    ser.reset_output_buffer()

    total = 0
    correct = 0
    confusion = np.zeros((4, 4), dtype=int)

    try:
        for class_name in CLASS_FOLDERS:
            class_dir = ROOT / class_name

            if not class_dir.exists():
                print(f"Skipping missing folder: {class_dir}")
                continue

            true_label = label_from_folder(class_name)

            image_files = sorted(
                list(class_dir.glob("*.jpg")) +
                list(class_dir.glob("*.jpeg")) +
                list(class_dir.glob("*.png")) +
                list(class_dir.glob("*.bmp"))
            )[:MAX_IMAGES_PER_FOLDER]

            print()
            print(f"Processing class: {class_name}")
            print(f"Images selected: {len(image_files)}")
            print()

            for image_path in image_files:
                q_img = image_to_q8_8_60x80(image_path)

                pred, time_us, time_ms, logits = send_image_and_get_prediction(ser, q_img)

                total += 1
                correct += int(pred == true_label)

                if 0 <= pred < 4:
                    confusion[true_label, pred] += 1

                    print(
                    f"{image_path.name}: "
                    f"true={true_label} "
                    f"pred={pred} "
                    f"logits={logits} "
                    f"time={time_us} us "
                    f"({time_ms:.6f} ms) "
                    f"{'OK' if pred == true_label else 'WRONG'}"
                )

    finally:
        ser.close()

    print()
    print("=====================================")
    print(" FINAL RESULTS")
    print("=====================================")
    print()

    print(f"Total   : {total}")
    print(f"Correct : {correct}")

    if total > 0:
        print(f"Accuracy: {correct / total * 100:.2f}%")

    print()
    print("Confusion matrix")
    print("(rows=true, cols=pred)")
    print()
    print(confusion)
    print()


if __name__ == "__main__":
    main()