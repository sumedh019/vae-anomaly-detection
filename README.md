# VAE-Based Industrial Anomaly Detection

> **Final-Year B.Tech AIML Project** — A Streamlit dashboard for inspecting industrial product images using a Variational Autoencoder (VAE).

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-red)
![Status](https://img.shields.io/badge/Model-Not%20Trained-orange)

---

## 🚀 Quick Start (Windows)

```bash
# 1. Create a virtual environment
python -m venv venv
venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the dashboard
streamlit run app.py
```

The dashboard opens at **http://localhost:8501**.

---

## 📁 Project Structure

```
vae-anomaly-detection/
├── app.py                        # Entry point — Overview page
├── requirements.txt
├── .gitignore
├── README.md
│
├── pages/                        # Streamlit multipage navigation
│   ├── 1_🔍_Image_Inspection.py  # Upload & inspect images
│   ├── 2_📊_Model_Performance.py # Evaluation charts (placeholder)
│   └── 3_📋_Inspection_History.py# Browse past inspections
│
├── src/                          # Reusable business logic
│   ├── config.py                 # Constants & colour palette
│   ├── anomaly_detector.py       # Detection pipeline wrapper
│   ├── image_processing.py       # Image utilities (diff, heatmap)
│   └── session_state.py          # Streamlit state helpers
│
└── tests/                        # Unit tests (pytest)
    └── test_image_processing.py
```

---

## 📖 Pages

| Page | Description |
|------|-------------|
| **🏭 Overview** | KPI cards (total inspections, anomalies, rate) and quick-start guide. |
| **🔍 Image Inspection** | Upload an image → view Original, Reconstructed, Difference Map, and Anomaly Heatmap in a 2×2 grid. Anomaly score with adjustable threshold. |
| **📊 Model Performance** | ROC curve, training loss, and classification metrics — all placeholder until the model is trained. |
| **📋 Inspection History** | Table of past inspections with CSV export and clear button. |

---

## ⚠️ Important Note

The VAE model has **not been trained yet**. All detection results are **placeholder outputs**:

- The "reconstructed" image is a Gaussian-blurred copy of the original.
- The anomaly score is derived from simple pixel-difference heuristics.
- The performance charts contain randomly generated data.

These placeholders demonstrate the dashboard layout and will be replaced once a real VAE model is integrated.

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

---

## 🛠️ Next Steps

1. Select and download an industrial dataset (e.g., MVTec AD).
2. Implement and train the VAE model (PyTorch / TensorFlow).
3. Replace `src/anomaly_detector.py` placeholder logic with real inference.
4. Populate the Model Performance page with actual evaluation metrics.

---

## 📜 License

This project is for academic purposes (B.Tech AIML final-year project).


## Calibrating inspection verdicts

The inspection page now withholds a verdict until a threshold is calibrated,
unless you explicitly enable an experimental manual threshold. The old `0.5`
default is not a validated decision boundary. A score is reconstruction error,
not a probability that a product is defective; heatmap colours alone are not a
verdict. “Below threshold” does not certify a defect-free product.

With your trained bottle checkpoint at `models/vae_bottle.pth`, use at least 20
held-out known-good images from the same product category and capture conditions:

```bash
python -m src.calibration --normal-dir /path/to/held-out/bottle/good
```

This writes `models/vae_bottle.calibration.json`, bound to the checkpoint hash
and deterministic scoring version. Recalibrate after replacing the checkpoint.
The default targets a 5% false-positive rate on the calibration sample; this is
not a guarantee on future images. Evaluate normal and defective images on a
separate set before relying on decisions. Do not calibrate on training images
or select a threshold merely to make one uploaded defect pass.

Inference now decodes the latent mean rather than sampling randomly. Previous
evaluation CSV scores and thresholds must be regenerated. The committed legacy
evaluation CSV has ROC-AUC approximately 0.409 and cannot establish reliable
defect detection; a better-trained model may be required. No checkpoint or
MVTec dataset is included in this repository.
