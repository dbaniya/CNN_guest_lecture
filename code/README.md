### Data Augmentation (**augment_data.py**)

This script generates **20 augmented versions of each image** across four classes: Full Cable, Left End, Right End, and No Cable. With `SAVE_ORIGINALS = True`, each input produces **21 output images**, including the original.

The augmentations simulate changes in lighting, contrast, color, blur, noise, image quality, and camera position. Output images retain the class folder structure, and `README_augmentations.txt` records the applied transformations.

Before running the script, set `DATA_ROOT` to the input dataset folder and `OUTPUT_ROOT` to the desired output folder. Existing output images are skipped unless `OVERWRITE = True`. Optional JPEG compression is currently disabled (`JPEG_ONLY_COUNT = 0`).
