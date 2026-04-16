🇷🇺Русская версия: [README_ru.md](README_ru.md)
# Background Removal and Replacement

Computer Vision project for **image background removal and replacement** using modern segmentation models, model benchmarking, weighted ensembling, and a Gradio web interface.

## Overview

This project explores a practical background removal pipeline built around two segmentation models:

- **BiRefNet**
- **RMBG**

The work includes:

- inference pipeline for foreground mask prediction,
- quantitative evaluation on **DUTS-TE**,
- comparison using **IoU**, **Dice**, and **MAD** metrics,
- weighted ensemble of model predictions,
- background replacement with a solid color, transparency, or another image,
- interactive **Gradio** web application.

The project is designed as a portfolio-ready CV case that combines **research**, **engineering**, and **product packaging**.

---

## Key Results

Experiments on DUTS-TE showed that **BiRefNet** is the strongest standalone model in this project and provides the most stable segmentation quality.

Average results on a subset of test images:

- **BiRefNet**: IoU `0.9337`, Dice `0.9651`, MAD `0.0098`
- **RMBG**: IoU `0.9288`, Dice `0.9622`, MAD `0.0114`

A weighted ensemble was also evaluated. The best configuration on the explored subset was:

- **Ensemble**: `0.85 * BiRefNet + 0.15 * RMBG`

On a larger evaluation subset, this ensemble slightly improved **IoU** and **Dice** over the standalone baseline, while **BiRefNet** still remained slightly better in **MAD**.

### Practical conclusion

- Use **BiRefNet** as the default model for a robust and reliable pipeline.
- Use **Ensemble (0.85 / 0.15)** when maximizing overlap-based metrics is the priority.

---

## Features

- Background removal from a user-uploaded image
- Choice of segmentation model:
  - `BiRefNet`
  - `RMBG`
  - `Ensemble`
- Background replacement modes:
  - solid color
  - transparent PNG
  - custom background image
- Output preview for:
  - predicted mask
  - final composited result
- Gradio-based local web application

---

## Project Structure

```text
remove_background/
├── app/
│   └── app.py
├── data/
├── models/
├── notebooks/
├── src/
│   ├── __init__.py
│   ├── compositing.py
│   ├── ensemble.py
│   ├── matting.py
│   ├── metrics.py
│   ├── pipeline.py
│   └── utils/
│       ├── __init__.py
│       └── io.py
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Pipeline

The inference pipeline consists of the following steps:

1. Load the input image
2. Predict the foreground mask using one of the supported models
3. Optionally combine masks using a weighted ensemble
4. Post-process the mask
5. Composite the foreground with:
   - a solid background,
   - a transparent background,
   - or a custom background image
6. Return the mask and final result

The core orchestration logic is implemented in:

```python
src/pipeline.py
```

---

## Models

### BiRefNet
Used as the main high-quality segmentation model.

### RMBG
Used both as a standalone baseline and as a complementary model in the ensemble.

Model weights are downloaded locally and stored in the `models/` directory.

---

## Installation

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd remove_background
```

### 2. Create and activate a virtual environment

**Conda**

```bash
conda create -n bg-remove python=3.10
conda activate bg-remove
```

or **venv**

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the App

From the project root:

```bash
python -m app.app
```

After launch, Gradio will open a local interface, usually at:

```text
http://127.0.0.1:7861
```

---

## How to Use

1. Upload an image
2. Select a model:
   - `birefnet`
   - `rmbg`
   - `ensemble`
3. Select a background mode:
   - `solid`
   - `transparent`
   - `image`
4. If needed:
   - choose a solid color,
   - or upload a custom background image
5. Click **Run**
6. Review the predicted mask and the final composited image

---

## Example Use Cases

- profile photo cleanup
- marketplace or e-commerce product images
- quick content creation for social media
- CV / image editing demos
- prototyping segmentation-based photo tools

---

## Research Notes

This project was built not only as an application, but also as a small experimental study.

What was done:

- compared two segmentation models,
- evaluated them on a benchmark dataset,
- measured quality with standard segmentation metrics,
- tested whether ensembling improves the final result,
- packaged the final pipeline as an interactive application.

This makes the project a strong example of combining:

- **computer vision experimentation**,
- **model evaluation**, and
- **ML productization**.

---

## Limitations

- Evaluation was performed on selected subsets of DUTS-TE rather than the full benchmark.
- Ensemble weights were selected empirically.
- The current version focuses on quality, not inference speed benchmarking.
- Background image replacement currently uses resizing; more advanced fitting strategies can be added later.

---

## Future Improvements

- full benchmark on a larger evaluation set
- inference speed comparison
- mask smoothing / edge refinement options in UI
- batch processing
- deployment to Hugging Face Spaces
- downloadable outputs from the interface
- support for more advanced background placement modes

---

## Tech Stack

- Python
- PyTorch
- Transformers
- TorchVision
- NumPy
- Pillow
- Hugging Face Hub
- Gradio

---

## Author

**Maksim Novikov**

Portfolio project in **Computer Vision / Image Segmentation / ML Engineering**.

