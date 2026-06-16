# Semantic Scene Parsing on Cityscapes

A production-ready semantic segmentation project using **DeepLabV3 + ResNet-50**
trained on the **Cityscapes** dataset to classify 19 urban scene classes at the
pixel level.

---

## Project Structure

```
semantic-scene-parsing/
├── app.py                      # Streamlit web application
├── requirements.txt            # Pinned Python dependencies
├── README.md                   # This file
│
├── src/
│   ├── __init__.py
│   ├── label_map.py            # Cityscapes ID → trainId mapping & colorization
│   ├── dataset.py              # PyTorch Dataset class for Cityscapes
│   ├── prepare_dataset.py      # Dataset verification & class statistics
│   ├── model.py                # DeepLabV3 model builder
│   ├── train.py                # Full training pipeline
│   ├── evaluate.py             # Evaluation: Pixel Acc, mIoU, per-class IoU
│   └── inference.py            # Single-image production inference
│
├── data/
│   └── cityscapes/             # ← Place dataset here (see Setup below)
│       ├── leftImg8bit/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       └── gtFine/
│           ├── train/
│           ├── val/
│           └── test/
│
├── checkpoints/                # Saved model checkpoints (auto-created)
│   ├── best_model.pth
│   └── logs/                   # TensorBoard logs
│
├── outputs/                    # Inference and eval visualizations
└── sample_images/              # Optional: demo images for the Streamlit app
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

> **GPU recommended.** Training on CPU is possible but will take ~10× longer.
> Minimum: 8 GB VRAM for batch_size=8 at 512×256.

### 2. Download the Cityscapes Dataset

Cityscapes requires a **free account** (academic use).

1. Register at https://www.cityscapes-dataset.com/
2. Download these two archives:
   - `leftImg8bit_trainvaltest.zip` (~11 GB) — RGB images
   - `gtFine_trainvaltest.zip` (~241 MB) — annotation labels
3. Extract both into `data/cityscapes/`:

```bash
unzip leftImg8bit_trainvaltest.zip -d data/cityscapes/
unzip gtFine_trainvaltest.zip      -d data/cityscapes/
```

### 3. Verify Dataset

```bash
python src/prepare_dataset.py --data_root data/cityscapes

# Optional: compute class pixel statistics (takes ~1 min, improves training)
python src/prepare_dataset.py --data_root data/cityscapes --check_stats
```

Expected output:
```
[OK]  data/cityscapes/leftImg8bit/train
[OK]  data/cityscapes/leftImg8bit/val
[OK]  data/cityscapes/gtFine/train
[OK]  data/cityscapes/gtFine/val

train:  2975 images  [OK]
val  :   500 images  [OK]
test :  1525 images  [OK]
```

### 4. Train the Model

```bash
python src/train.py \
    --data_root  data/cityscapes \
    --output_dir checkpoints \
    --epochs     50 \
    --batch_size 8 \
    --lr         0.001
```

**Expected training time:**
| Hardware       | ~Time per epoch | ~Total (50 ep) |
|----------------|-----------------|----------------|
| RTX 3080       | ~4 min          | ~3.5 hours     |
| RTX 4090       | ~2 min          | ~1.7 hours     |
| CPU only       | ~60 min         | ~50 hours      |

Monitor training with TensorBoard:
```bash
tensorboard --logdir checkpoints/logs
```

### 5. Evaluate

```bash
python src/evaluate.py \
    --checkpoint checkpoints/best_model.pth \
    --data_root  data/cityscapes \
    --save_vis \
    --save_results
```

**Expected results after 50 epochs at 512×256:**
```
Pixel Accuracy : ~93%
Mean IoU (mIoU): ~68-72%
```

Per-class IoU (approximate):
| Class          | IoU   |
|----------------|-------|
| road           | ~97%  |
| sky            | ~94%  |
| building       | ~90%  |
| vegetation     | ~89%  |
| car            | ~93%  |
| person         | ~78%  |
| bicycle        | ~70%  |
| motorcycle     | ~55%  |
| train          | ~50%  |
| rider          | ~55%  |

> Rare classes (train, motorcycle, rider) score lower due to fewer training pixels,
> even with class-weighted loss.

### 6. Run Single-Image Inference

```bash
python src/inference.py \
    --checkpoint checkpoints/best_model.pth \
    --image      path/to/your_image.jpg \
    --output_dir outputs/
```

Produces:
- `outputs/your_image_overlay.png` — color segmentation blended on original
- `outputs/your_image_segmentation.png` — pure color map
- `outputs/your_image_coverage.json` — per-class pixel percentages

### 7. Launch the Web App

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

The app works in **demo mode** without a trained checkpoint (shows illustrative
output). For real predictions, train the model first and enter the checkpoint
path in the sidebar.

---

## Key Technical Decisions

### Why DeepLabV3?
DeepLabV3's **Atrous Spatial Pyramid Pooling (ASPP)** uses dilated convolutions
at rates [6, 12, 18] to capture multi-scale context without resolution loss.
This is critical for segmenting both small objects (traffic lights, cyclists)
and large uniform regions (road, sky) in the same image.

### Why 512×256?
This is exactly ¼ of Cityscapes' native 2048×1024 resolution, preserving the
2:1 aspect ratio. Memory scales with resolution², so ¼-scale = 1/16th the
feature map memory — enabling batch_size=8 on a consumer GPU instead of
batch_size=1 at full resolution.

### Class Imbalance Handling
Road and building pixels account for >50% of all labelled pixels. Without
correction, the model ignores rare classes entirely.
**Strategy:** Median-frequency weighting:
```
w_c = median(all_class_frequencies) / frequency_c
```
This upweights rare classes (motorcycle, rider: ~10×) without over-amplifying
extremely rare ones to destabilizing levels.

### Ignore Label (255)
Pixels labelled 255 are excluded from loss computation and metrics. These
represent ambiguous regions (ego vehicle, unlabelled areas) that would
confuse the model if included.

---

## File-by-File Summary

| File | Purpose |
|------|---------|
| `src/label_map.py` | Defines all 34 raw Cityscapes IDs, maps them to 19 trainIds, provides color palette |
| `src/dataset.py` | PyTorch Dataset: loads images+labels, resizes to 512×256, remaps labels, augments |
| `src/prepare_dataset.py` | Validates dataset structure, counts images, computes class pixel stats |
| `src/model.py` | Builds DeepLabV3-ResNet50, replaces 21-class VOC head with 19-class Cityscapes head |
| `src/train.py` | Training loop: weighted loss, SGD + PolynomialLR, checkpoint saving, TensorBoard |
| `src/metrics.py` | Confusion-matrix-based Pixel Accuracy, Per-Class IoU, Mean IoU |
| `src/evaluate.py` | Runs full val set evaluation, saves metric table and prediction visualizations |
| `src/inference.py` | Production single-image inference with overlay generation and coverage stats |
| `app.py` | Streamlit UI: image upload, real-time overlay, coverage chart, download buttons |

---

## License

This project is for educational and portfolio purposes.
Cityscapes dataset is subject to its own [terms of use](https://www.cityscapes-dataset.com/license/).
